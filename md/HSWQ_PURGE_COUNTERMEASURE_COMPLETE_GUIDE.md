# Complete Technical Guide: HSWQ Purge Countermeasure (Distorch)

**Baseline (start point):** `a5f25d5` -- `chore: bump version to 2.4.3`
**Tip covered by this guide:** `f5c30e9` on `main` (and intermediate purge commits after the baseline)
**Scope:** Distorch **Purge VRAM V2** + **Memory Manager** residual cleanup. HSWQ Loader may also peel Z Image contamination on SDXL load; Distorch still clears every HSWQ surface on purge so leftovers cannot survive across models.

This document answers four questions in order:

1. What was wrong
2. Which files were added / modified
3. Full text of the added / modified purge code
4. What that code means

---

## 1. What was wrong

### 1.1 Product symptoms

Four related failure modes appear around Distorch purge / Memory Manager:

**A. Same-process second prompt (poison after HSWQ nuclear kill)**

- First HSWQ generation (INT8 and/or NVFP4, including Z-Image ConvRot and Detailer paths) succeeds.
- Method 3 empties live CUDA storage while **HSWQ** toggle is ON.
- Module-level pools, Hadamard caches, ConvRot arms, bake bookkeeping, and TC forward caches still point at **dead** storage.
- The next prompt in the **same** ComfyUI process fails or produces garbage (`quantize_nvfp4` / cuBLAS INT8 seeing `PyCapsule`, noise, half-purged modules).

**B. Cross-model contamination (SDXL NVFP4 -> Z Image NVFP4 -> SDXL NVFP4)**

- SDXL NVFP4 first gen OK.
- Z Image NVFP4 second gen OK (ZI installs `comfy_parity` + ZI LoRA bake hooks).
- Return to SDXL NVFP4 third gen produces **noise** because ZI parity / bake hooks and HSWQ globals were not peeled on purge.
- Loader-side peel on SDXL load helps, but Distorch purge must still remove every residual so the next model never inherits ZI state.

**C. Soft purge / Memory Manager left ~9-12 GB after Krea2 NVFP4 (HSWQ toggle OFF)**

- Soft path called `model_unload()` and reported "Unloaded N model(s)" / Memory Manager printed "GPU memory cleared".
- MultiGPU Dynamic / NVFP4 often left ~9 GB CUDA resident (Task Manager ~10+ GB; monitor `cuda:0|~9`).
- Root causes:
  - Soft path did not force-empty leftover CUDA param/buffer storage after unload.
  - Memory Manager / SafeMemoryManager called `free_memory(0, ...)` which is a **no-op** in ComfyUI.
- HSWQ nuclear path must **not** run when the HSWQ toggle is OFF. Soft residue is fixed on the soft path + Memory Manager only.

**D. CPU wipe broke CLIP / ZI TE after Ollama + soft/HSWQ purge**

- After Ollama purge (and soft / HSWQ storage kill), CLIP / text-encoder reload failed:
  - `Embedding.weight` became non-2D
  - `CLIPTextEncode` raised `RuntimeError: 'weight' must be 2-D`
  - reload logged `0.00 MB`
- Cause: `empty(0)` was applied to **CPU** tensors that are ComfyUI's reload source after `model_unload()`.
- Fix: `_force_empty_cuda_storage` / `_kill_tensor_storage` wipe **CUDA only**; pinned host may unpin; CPU weights are never replaced with `empty(0)`.

### 1.2 State at baseline `a5f25d5`

At `a5f25d5`, Purge VRAM V2 already had a large HSWQ block (PinCache drain, Detailer / PromptExecutor sweep, PINNED_MEMORY unregister, Method 3 nuclear kill, Method 2c kitchen + basic `clear_nvfp4_runtime_pools`). That was **not** enough for A/B. Soft path and Memory Manager gaps (C) and CPU wipe (D) were addressed later (`f5c30e9`).

| Gap | Effect |
|---|---|
| `_is_hswq_int8_nn` was **INT8-only** | Pure NVFP4 / ZI ConvRot modules skipped for Method 1/3 |
| No full scan for `_hswq_nvfp4_*`, `_hswq_convrot`, ZI bake keys, any residual `_hswq_*` | Arms and parity tensors survived Method 3 |
| `_kill_module_vram` only nulled INT8 bake keys | NVFP4 parity H, w_plain, alpha, Conv2d `_hswq_convrot`, act scales remained |
| Method 2c = kitchen + narrow NVFP4 API | Missed inplace pool dicts, INT8 Hadamard, ZI `zi_nvfp4_hadamard`, stack peel |
| No `restore_nvfp4_tc_product_stack` / `uninstall_zimage_nvfp4_lora_bake` on purge | SDXL after ZI kept poisoned TC / bake hooks |
| Soft unload without CUDA force-empty + `free_memory(0)` | Krea2 / MultiGPU ~9 GB left with HSWQ OFF |
| Storage kill wiped CPU tensors | CLIP / ZI TE broken after Ollama-adjacent purge |

### 1.3 Failure mechanisms

```text
A/B (HSWQ ON):
1st gen OK
  -> HSWQ arms modules + kitchen / nvfp4_runtime / Hadamard / (ZI) parity+bake
  -> Distorch Method 3 replaces CUDA storage (dead)
  -> pools + attrs + hooks still reference dead storage or ZI stack
  -> 2nd gen same model OR next model (e.g. SDXL after ZI) -> noise / TypeError

C (HSWQ OFF, soft / Memory Manager):
Krea2 NVFP4 gen OK
  -> soft model_unload / "GPU memory cleared"
  -> CUDA storage still held (~9GB)
  -> free_memory(0) no-op
  -> Task Manager still ~10+ GB

D (Ollama + storage kill):
purge touches loaded models
  -> empty(0) on CPU Embedding.weight (reload source)
  -> CLIPTextEncode RuntimeError weight must be 2-D / reload 0.00 MB
```

Correct Distorch policy:

1. HSWQ toggle ON -> full residual surface clear + ZI peel after nuclear kill.
2. Soft path always (toggle-independent) -> force-empty **CUDA** leftovers + `unload_all_models` + `free_memory(1e30)`.
3. Never `empty(0)` CPU tensors that ComfyUI needs to reload.
4. HSWQ nuclear Methods 0-2c run **only** when the HSWQ toggle is ON (no auto-arm).

### 1.4 Commits after `a5f25d5` (purge line)

| Commit | Role |
|---|---|
| `65d6b2c` | Detect ZI ConvRot NVFP4; clear parity Hadamard |
| `ea3e2a3` | Clear ZI NVFP4 bake keys on Distorch HSWQ purge |
| `4e60fbd` | Clear `nvfp4_hadamard` global cache in Method 2c |
| `6e7ca80` | Align NVFP4 purge clear with Loader `_clear_one` attrs |
| `4c08922` | Always clear all HSWQ pools / attrs / arms after Distorch purge |
| `f9d90d1` | Full surface: INT8 Conv2d `_hswq_convrot`, INT8 Hadamard, Detailer, any `_hswq_*` |
| `bea2867` | Spell **Distorch** in comments / `CATEGORY` (class id `DisTorchPurgeVRAMV2` kept for workflows) |
| `d604c15` | Method 2c peels ZI parity/bake; broadens NVFP4/Hadamard/forward clears |
| `f5c30e9` | Soft CUDA force-empty + Memory Manager `free_memory(1e30)`; CUDA-only `empty(0)` so CPU TE survives Ollama/soft/HSWQ |

---

## 2. Files added / modified

No new production modules. Keep twins in sync:

| File | Role |
|---|---|
| `nodes/purge_vram.py` | Canonical node (`DisTorchPurgeVRAMV2`, category `Distorch/Memory`) |
| `purge_vram.py` | Compatibility / deploy twin of the same node logic |
| `nodes/memory_manager.py` | Twin of Memory Manager / SafeMemoryManager |
| `memory_manager.py` | Compatibility twin -- `free_memory(1e30)` on all CUDA devices |
| `__init__.py` | Prefer `nodes.purge_vram`; comment spelling `Distorch` |

Also relevant:

| Item | Note |
|---|---|
| HSWQ Loader | May peel on SDXL load; Distorch still owns purge-time residual cleanup |
| `pyproject.toml` / Registry | Untouched for this work |

Toggles:

```python
"HSWQ": ("BOOLEAN", {"default": False, "tooltip": "Purge HSWQ residual VRAM (whole HSWQ path: models, PinCache, Detailer caches)"}),
"Ollama": ("BOOLEAN", {"default": False, "tooltip": "Unload Ollama server models (comfyui-ollama / describer keep_alive)"}),
```

Legacy workflow key `HSWQ INT8` is still accepted in `purge()`. Soft CUDA force-empty runs whenever `purge_models` is used, independent of HSWQ.

---

## 3. Full text of the added / modified purge code

### 3.1 Soft path -- CUDA force-empty + hard free (always with `purge_models`)

**Source:** `nodes/purge_vram.py` at tip `f5c30e9`, **lines 125-327** (inclusive).

```python
                # Aggressive unload: MultiGPU Dynamic / NVFP4 (Krea2) often leave
                # ~9GB CUDA after model_unload()==True. Soft path alone is not enough.
                if hasattr(comfy.model_management, "current_loaded_models"):
                    current_loaded_models = comfy.model_management.current_loaded_models
                    unloaded_count = 0
                    bytes_force_killed = 0

                    def _force_empty_cuda_storage(t) -> int:
                        # NVFP4 / MultiGPU Dynamic: free leftover CUDA only.
                        # Never wipe CPU tensors — after model_unload() they are
                        # ComfyUI's reload source. Wiping to empty(0) made CLIP
                        # Embedding.weight non-2D (Ollama purge → CLIPTextEncode
                        # RuntimeError: 'weight' must be 2-D; reload logged 0.00 MB).
                        if t is None:
                            return 0
                        freed = 0
                        try:
                            data = getattr(t, "data", t)
                            if data is None:
                                return 0
                            nbytes = int(getattr(data, "nbytes", 0) or 0)
                            is_cuda = False
                            try:
                                is_cuda = bool(getattr(data, "is_cuda", False))
                                if not is_cuda:
                                    dev = getattr(data, "device", None)
                                    is_cuda = getattr(dev, "type", None) == "cuda"
                            except Exception:
                                pass
                            if not is_cuda:
                                return 0
                            dtype = getattr(data, "dtype", torch.float32)
                            empty = torch.empty(0, dtype=dtype, device="cpu")
                            if hasattr(t, "data"):
                                t.data = empty
                            freed = nbytes
                        except Exception:
                            pass
                        return freed

                    def _force_kill_nn_cuda(module) -> int:
                        if module is None:
                            return 0
                        freed = 0
                        try:
                            if hasattr(module, "to") and callable(module.to):
                                try:
                                    module.to("cpu")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            for _n, p in list(module.named_parameters()):
                                freed += _force_empty_cuda_storage(p)
                        except Exception:
                            pass
                        try:
                            for _n, b in list(module.named_buffers()):
                                freed += _force_empty_cuda_storage(b)
                        except Exception:
                            pass
                        return freed

                    def _unwrap_nn_soft(obj):
                        cur = obj
                        for _ in range(8):
                            if cur is None:
                                return None
                            try:
                                if isinstance(cur, torch.nn.Module):
                                    return cur
                            except Exception:
                                pass
                            nxt = getattr(cur, "model", None)
                            if nxt is None or nxt is cur:
                                nxt = getattr(cur, "diffusion_model", None)
                            if nxt is None or nxt is cur:
                                try:
                                    return cur if isinstance(cur, torch.nn.Module) else None
                                except Exception:
                                    return None
                            cur = nxt
                        try:
                            return cur if isinstance(cur, torch.nn.Module) else None
                        except Exception:
                            return None

                    # Mark unused, unload, kill CUDA storage, then remove from registry
                    for i in range(len(current_loaded_models) - 1, -1, -1):
                        loaded_model = current_loaded_models[i]
                        if loaded_model is None:
                            try:
                                current_loaded_models.pop(i)
                            except Exception:
                                pass
                            continue
                        try:
                            try:
                                loaded_model.currently_used = False
                            except Exception:
                                pass
                            try:
                                if hasattr(loaded_model, "partially_unload") and callable(loaded_model.partially_unload):
                                    try:
                                        loaded_model.partially_unload(None, 1e30)
                                    except Exception:
                                        loaded_model.partially_unload(torch.device("cpu"), 1e30)
                            except Exception:
                                pass
                            try:
                                if hasattr(loaded_model, "model_unload") and callable(loaded_model.model_unload):
                                    loaded_model.model_unload()
                                    unloaded_count += 1
                            except Exception as e:
                                print(f"Error unloading model: {e}")
                            try:
                                inner = getattr(loaded_model, "model", None)
                                nn = _unwrap_nn_soft(inner)
                                if nn is not None:
                                    bytes_force_killed += _force_kill_nn_cuda(nn)
                                elif inner is not None:
                                    bytes_force_killed += _force_kill_nn_cuda(_unwrap_nn_soft(inner))
                            except Exception:
                                pass
                            try:
                                current_loaded_models.pop(i)
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"Error force-unloading model[{i}]: {e}")
                            try:
                                current_loaded_models.pop(i)
                            except Exception:
                                pass

                    if unloaded_count > 0:
                        print(f"Unloaded {unloaded_count} model(s)")
                    if bytes_force_killed > 0:
                        print(
                            f"Force-killed ~{bytes_force_killed / (1024 ** 3):.2f} GB CUDA storage "
                            f"from loaded models (MultiGPU/NVFP4 soft-unload residue)"
                        )

                    # Pre-cleanup again before second cleanup_models() call
                    if hasattr(comfy.model_management, "current_loaded_models"):
                        current_loaded_models = comfy.model_management.current_loaded_models
                        pre_cleaned_2 = 0
                        for i in range(len(current_loaded_models) - 1, -1, -1):
                            loaded_model = current_loaded_models[i]
                            if loaded_model is not None:
                                try:
                                    if hasattr(loaded_model, "real_model"):
                                        real_model = loaded_model.real_model
                                        if real_model is None or not callable(real_model):
                                            current_loaded_models.pop(i)
                                            pre_cleaned_2 += 1
                                        else:
                                            try:
                                                if real_model() is None:
                                                    current_loaded_models.pop(i)
                                                    pre_cleaned_2 += 1
                                            except (TypeError, AttributeError):
                                                current_loaded_models.pop(i)
                                                pre_cleaned_2 += 1
                                except Exception:
                                    pass
                        
                        if pre_cleaned_2 > 0:
                            print(f"Pre-cleaned {pre_cleaned_2} problematic model(s) before second cleanup_models()")
                    
                    # Cleanup again after unloading
                    if hasattr(comfy.model_management, "cleanup_models"):
                        try:
                            comfy.model_management.cleanup_models()
                        except Exception as e:
                            print(f"Error in cleanup_models: {e}")

                # Hard free: unload_all + free_memory(1e30). free_memory(0) does nothing.
                try:
                    mm = comfy.model_management
                    if hasattr(mm, "unload_all_models") and callable(mm.unload_all_models):
                        mm.unload_all_models()
                        print("unload_all_models() issued")
                    if torch.cuda.is_available() and hasattr(mm, "free_memory") and callable(mm.free_memory):
                        for di in range(torch.cuda.device_count()):
                            try:
                                mm.free_memory(1e30, torch.device(f"cuda:{di}"))
                            except Exception as e:
                                print(f"free_memory(cuda:{di}) warning: {e}")
                        print("free_memory(1e30) issued for all CUDA devices")
                except Exception as e:
                    print(f"Hard free after purge_models warning: {e}")
                
                # Soft empty cache (if available)
                if hasattr(comfy.model_management, "soft_empty_cache") and callable(comfy.model_management.soft_empty_cache):
                    try:
                        comfy.model_management.soft_empty_cache()
                    except Exception as e:
                        print(f"Error in soft_empty_cache: {e}")
                    
            except Exception as e:
                print(f"Error purging models: {e}")
```

### 3.2 HSWQ nuclear path (only when HSWQ toggle ON)

**Source:** `nodes/purge_vram.py` at tip `f5c30e9`, **lines 2092-3566** (inclusive).
Root `purge_vram.py` carries the same block and must stay aligned.

```python
        # Purge HSWQ whole path (models / INT8 / PinCache / Detailer / pins / kitchen)
        # + orphaned ComfyUI cudaHostRegister / CUDA tensors (Task Manager dedicated GPU mem)
        # HSWQ nuclear runs ONLY when the HSWQ toggle is explicitly ON — never auto-arm.
        if purge_hswq_int8:
            try:
                print("HSWQ INT8/NVFP4: Starting purge process...")
                hswq_cleared = 0
                bytes_killed = 0
                pins_unregistered = 0
                cuda_tensors_killed = 0
                patchers_unloaded = 0

                def _sys_modules():
                    # Never use a local "import sys" in purge_vram — it shadows module-level sys.
                    return list(sys.modules.items())

                def _mem_diag(tag: str) -> None:
                    try:
                        import comfy.model_management as mm
                        total_pin = int(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0)
                        pin_entries = len(getattr(mm, "PINNED_MEMORY", {}) or {})
                        print(
                            f"HSWQ INT8/NVFP4: [{tag}] TOTAL_PINNED_MEMORY="
                            f"{total_pin / (1024 * 1024):.1f} MB entries={pin_entries}"
                        )
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: [{tag}] pin diag failed: {e}")
                    if torch.cuda.is_available():
                        try:
                            for di in range(torch.cuda.device_count()):
                                alloc = torch.cuda.memory_allocated(di)
                                reserved = torch.cuda.memory_reserved(di)
                                free_b, total_b = torch.cuda.mem_get_info(di)
                                used_sys = max(0, total_b - free_b)
                                print(
                                    f"HSWQ INT8/NVFP4: [{tag}] cuda:{di} "
                                    f"allocated={alloc / (1024 ** 3):.2f}GB "
                                    f"reserved={reserved / (1024 ** 3):.2f}GB "
                                    f"sys_used={used_sys / (1024 ** 3):.2f}GB"
                                )
                        except Exception as e:
                            print(f"HSWQ INT8/NVFP4: [{tag}] cuda diag failed: {e}")

                def _drain_hswq_pin_cache() -> int:
                    drained = 0

                    def _call_purge(mod, mod_name: str) -> int:
                        fn = getattr(mod, "purge_pin_cache", None)
                        if callable(fn):
                            got = int(fn() or 0)
                            print(
                                f"HSWQ INT8/NVFP4: PinCache purged via {mod_name}: "
                                f"{got / (1024 * 1024):.1f} MB"
                            )
                            return got
                        pool = getattr(mod, "_PIN_BUFFER_POOL", None)
                        total = int(getattr(mod, "_PIN_CACHE_TOTAL", 0) or 0)
                        drain = getattr(mod, "_drain_pool", None)
                        if callable(drain):
                            setattr(mod, "_active", False)
                            setattr(mod, "_depth", 0)
                            drain()
                            print(
                                f"HSWQ INT8/NVFP4: PinCache _drain_pool via {mod_name}: "
                                f"{total / (1024 * 1024):.1f} MB"
                            )
                            return total
                        if pool is not None:
                            pool.clear()
                            setattr(mod, "_PIN_CACHE_TOTAL", 0)
                            print(f"HSWQ INT8/NVFP4: PinCache pool cleared via {mod_name}")
                            return total
                        return 0

                    found = False
                    for mod_name, mod in _sys_modules():
                        if mod is None or "hswq_pin_cache" not in str(mod_name):
                            continue
                        try:
                            drained += _call_purge(mod, str(mod_name))
                            found = True
                        except Exception as e:
                            print(f"HSWQ INT8/NVFP4: PinCache purge via {mod_name} failed: {e}")
                    if found:
                        return drained

                    # Force-import: Detailer scope may have ended (deactivate drained
                    # tracking) or module never stayed in sys.modules under expected name.
                    try:
                        import importlib.util
                        pkg_dir = os.path.dirname(os.path.abspath(__file__))
                        # purge_vram.py at DistTorch root → custom_nodes is parent
                        cn_root = os.path.dirname(pkg_dir)
                        if os.path.basename(pkg_dir) == "nodes":
                            cn_root = os.path.dirname(os.path.dirname(pkg_dir))
                        candidates = [
                            os.path.join(
                                cn_root,
                                "ComfyUI-nunchaku-unofficial-loader",
                                "nodes",
                                "hswq_pin_cache.py",
                            ),
                            os.path.join(
                                cn_root,
                                "comfyui-nunchaku-unofficial-loader",
                                "nodes",
                                "hswq_pin_cache.py",
                            ),
                        ]
                        for pin_py in candidates:
                            if not os.path.isfile(pin_py):
                                continue
                            print(f"HSWQ INT8/NVFP4: Force-import PinCache from {pin_py}")
                            spec = importlib.util.spec_from_file_location(
                                "hswq_pin_cache_force_purge", pin_py
                            )
                            if spec is None or spec.loader is None:
                                continue
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            sys.modules["hswq_pin_cache_force_purge"] = mod
                            drained += _call_purge(mod, pin_py)
                            return drained
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: PinCache force-import failed: {e}")

                    print("HSWQ INT8/NVFP4: PinCache module not loaded (nothing to drain)")
                    return drained

                def _purge_detailer_segs_and_executor_cache() -> int:
                    """Drop Impact SEGS / large IMAGE held in PromptExecutor caches now.

                    Do NOT call PromptExecutor.reset() mid-prompt: reset() replaces
                    CacheSet with a fresh RAMPressureCache that has never run
                    set_prompt(), so cache_key_set is missing and the next
                    caches.outputs.get() raises AttributeError.

                    MultiGPU correctly only sets free_memory (reset after prompt).
                    Here we clear .cache / .subcaches in place so the current
                    prompt's cache_key_set / initialized state stay valid.
                    """
                    freed_hint = 0
                    cleared_entries = 0
                    executor_n = 0
                    try:
                        for obj in gc.get_objects():
                            if type(obj).__name__ != "PromptExecutor":
                                continue
                            executor_n += 1
                            try:
                                caches = getattr(obj, "caches", None)
                                if caches is None:
                                    continue
                                for cache in getattr(caches, "all", None) or []:
                                    try:
                                        cdict = getattr(cache, "cache", None)
                                        if isinstance(cdict, dict) and cdict:
                                            cleared_entries += len(cdict)
                                            cdict.clear()
                                        sub = getattr(cache, "subcaches", None)
                                        if isinstance(sub, dict) and sub:
                                            cleared_entries += len(sub)
                                            sub.clear()
                                        for attr in (
                                            "timestamps",
                                            "used_generation",
                                            "children",
                                        ):
                                            bag = getattr(cache, attr, None)
                                            if isinstance(bag, dict) and bag:
                                                bag.clear()
                                    except Exception as e:
                                        print(
                                            f"HSWQ INT8/NVFP4: in-place cache clear "
                                            f"failed: {e}"
                                        )
                            except Exception as e:
                                print(
                                    f"HSWQ INT8/NVFP4: PromptExecutor cache clear "
                                    f"failed: {e}"
                                )
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: PromptExecutor scan failed: {e}")
                    print(
                        f"HSWQ INT8/NVFP4: PromptExecutor in-place cache clear "
                        f"executors={executor_n} entries={cleared_entries}"
                    )

                    impact_cleared = 0
                    for mod_name, mod in _sys_modules():
                        if mod is None:
                            continue
                        n = str(mod_name).replace("\\", "/")
                        if "impact/core" not in n and not n.endswith("impact.core"):
                            if "impact.core" not in n:
                                continue
                        d = getattr(mod, "__dict__", None)
                        if not isinstance(d, dict):
                            continue
                        for attr in (
                            "preview_bridge_cache",
                            "preview_bridge_last_mask_cache",
                            "preview_bridge_image_id_map",
                            "preview_bridge_image_name_map",
                        ):
                            bag = d.get(attr)
                            if isinstance(bag, dict) and bag:
                                impact_cleared += len(bag)
                                bag.clear()
                    if impact_cleared:
                        print(
                            f"HSWQ INT8/NVFP4: Impact preview/SEG bridge caches cleared "
                            f"entries={impact_cleared}"
                        )

                    # HSWQ Batched Detailer / Impact Detailer instances holding SEGS crops
                    detailer_n = 0
                    detailer_tensors = 0
                    try:
                        for obj in gc.get_objects():
                            try:
                                tn = type(obj).__name__
                                if (
                                    "Detailer" not in tn
                                    and "SEGS" not in tn
                                    and "Segs" not in tn
                                ):
                                    continue
                                detailer_n += 1
                                d = getattr(obj, "__dict__", None)
                                if not isinstance(d, dict):
                                    continue
                                for attr, val in list(d.items()):
                                    try:
                                        if torch.is_tensor(val):
                                            nbytes = int(getattr(val, "nbytes", 0) or 0)
                                            if nbytes < 1024 * 1024:
                                                continue
                                            freed_hint += _kill_tensor_storage(val)
                                            detailer_tensors += 1
                                        elif isinstance(val, (list, tuple)):
                                            for item in val:
                                                if not torch.is_tensor(item):
                                                    continue
                                                nbytes = int(
                                                    getattr(item, "nbytes", 0) or 0
                                                )
                                                if nbytes < 1024 * 1024:
                                                    continue
                                                freed_hint += _kill_tensor_storage(item)
                                                detailer_tensors += 1
                                    except Exception:
                                        pass
                            except Exception:
                                continue
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: Detailer object sweep failed: {e}")
                    if detailer_n or detailer_tensors:
                        print(
                            f"HSWQ INT8/NVFP4: Detailer/SEGS object sweep "
                            f"objects={detailer_n} tensors={detailer_tensors}"
                        )

                    # Kill large CUDA / pinned tensors still reachable (SEG crops etc.)
                    tensor_killed = 0
                    try:
                        for obj in gc.get_objects():
                            try:
                                if not torch.is_tensor(obj):
                                    continue
                                nbytes = int(getattr(obj, "nbytes", 0) or 0)
                                if nbytes < 4 * 1024 * 1024:
                                    continue
                                pinned = False
                                try:
                                    pinned = bool(obj.is_pinned())
                                except Exception:
                                    pass
                                on_cuda = False
                                try:
                                    on_cuda = bool(obj.is_cuda)
                                except Exception:
                                    pass
                                if not pinned and not on_cuda:
                                    continue
                                freed_hint += _kill_tensor_storage(obj)
                                tensor_killed += 1
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: SEGS tensor sweep failed: {e}")
                    print(
                        f"HSWQ INT8/NVFP4: Detailer SEGS/cache sweep "
                        f"tensors_touched={tensor_killed} "
                        f"approx={freed_hint / (1024 * 1024):.1f} MB"
                    )
                    return freed_hint

                def _reset_comfy_kitchen_cuda_caches() -> None:
                    """Drop ALL HSWQ residual pools / module caches after nuclear kill.

                    Covers the whole HSWQ surface — not NVFP4 alone:

                    - INT8 (Linear protect ConvRot + Conv2d ``_hswq_convrot`` + LoRA bake)
                    - NVFP4 / ZI ConvRot (parity H, TC arms, runtime pools)
                    - Detailer path leaves SEGS via PromptExecutor (cleared separately)
                    - PinCache / kitchen / Hadamard globals (INT8 + NVFP4)

                    Independent of HSWQ Loader clear API presence or return values.
                    Loader ``clear_*`` is best-effort; local in-place + gc always run.
                    """
                    cleared = []

                    def _safe_hasattr(obj, name: str) -> bool:
                        # Some third-party modules (e.g. seedvr2 compatibility wrappers)
                        # raise ImportError from __getattr__; bare hasattr aborts purge.
                        try:
                            return hasattr(obj, name)
                        except Exception:
                            return False

                    def _safe_getattr(obj, name: str, default=None):
                        try:
                            return getattr(obj, name, default)
                        except Exception:
                            return default

                    def _drop_attr(obj, name: str) -> bool:
                        if not _safe_hasattr(obj, name):
                            return False
                        try:
                            delattr(obj, name)
                            return True
                        except Exception:
                            try:
                                setattr(obj, name, None)
                                return True
                            except Exception:
                                return False

                    def _clear_dict_attr(mod, attr: str) -> int:
                        bag = _safe_getattr(mod, attr, None)
                        if isinstance(bag, dict) and bag:
                            n = len(bag)
                            bag.clear()
                            return n
                        return 0

                    def _empty_cuda_tensor(t) -> None:
                        if t is None or not torch.is_tensor(t):
                            return
                        try:
                            data = getattr(t, "data", t)
                            if not bool(getattr(data, "is_cuda", False)):
                                return
                            empty = torch.empty(0, dtype=data.dtype, device=data.device)
                            t.data = empty
                        except Exception:
                            pass

                    # Known residual names (INT8 + NVFP4 + bake + forward caches).
                    # Stray walk below also drops every other ``_hswq_*`` on Modules.
                    _hswq_drop_attrs = (
                        # NVFP4 / ZI ConvRot
                        "_hswq_nvfp4_parity_H",
                        "_hswq_nvfp4_H",
                        "_hswq_nvfp4_w_plain",
                        "_hswq_nvfp4_alpha",
                        "_hswq_nvfp4_no_cudagraph",
                        "_hswq_nvfp4_convrot",
                        "_hswq_nvfp4_convrot_groupsize",
                        "_hswq_nvfp4_convrot_parity",
                        "_hswq_nvfp4",
                        "_hswq_nvfp4_act_scale",
                        "_hswq_nvfp4_scale_placeholder",
                        "_hswq_nvfp4_scale_from_ckpt",
                        "_hswq_zi_nvfp4_baked_keys",
                        "_hswq_zi_nvfp4_baked_uuid",
                        # INT8 Linear protect ConvRot
                        "_hswq_int8_convrot",
                        "_hswq_int8_convrot_groupsize",
                        "_hswq_int8_baked_keys",
                        "_hswq_int8_baked_uuid",
                        # INT8 Conv2d ConvRot (comfy_quant_int8 QuantConv2d)
                        "_hswq_convrot",
                        "_hswq_convrot_groupsize",
                    )

                    # --- comfy_kitchen ---
                    try:
                        import comfy_kitchen.backends.cuda as ck_cuda
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: comfy_kitchen cuda import skipped: {e}")
                        ck_cuda = None
                    if ck_cuda is not None:
                        for attr in ("_cublas_workspaces", "_empty_cuda_tensors"):
                            n = _clear_dict_attr(ck_cuda, attr)
                            if n:
                                cleared.append(f"{attr}={n}")

                    # --- Peel Z Image parity / bake hooks so SDXL TC is not poisoned ---
                    # Loader also does this on SDXL load; Distorch must do it on purge so
                    # the next model (SDXL after ZI) never inherits comfy_parity / ZI bake.
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        nlow = str(name).replace("\\", "/").lower()
                        if not (
                            "nvfp4" in nlow
                            or "zimage_nvfp4" in nlow
                            or "comfy_quant_nvfp4" in nlow
                        ):
                            continue
                        for api_name in (
                            "_clear_zimage_parity_contamination_for_sdxl",
                            "restore_nvfp4_tc_product_stack",
                            "uninstall_zimage_nvfp4_lora_bake",
                        ):
                            fn = _safe_getattr(mod, api_name, None)
                            if not callable(fn):
                                continue
                            try:
                                ret = fn()
                                cleared.append(f"{api_name}@{name}={ret!r}")
                                print(
                                    "HSWQ INT8/NVFP4: HSWQ stack peel "
                                    f"{api_name} via {name} -> {ret!r}"
                                )
                            except Exception as e_peel:
                                print(
                                    f"HSWQ INT8/NVFP4: {api_name} failed "
                                    f"({name}): {e_peel}"
                                )

                    # --- Peel ZI INT8-protect load arm on ops._load_quantized_module ---
                    # SDXL INT8 ConvRot and ZI INT8 protect share conf shape
                    # (int8_tensorwise + convrot). Loader peel may leave
                    # _hswq_int8_protect_in_load / _hswq_int8_protect_arm_v2
                    # (arm freevar is ``cur``, not orig_load) or PRODUCT wrapping
                    # that arm — then _arm_int8_protect_convrot_after_stock_load
                    # fires on SDXL load (Params.convrot=False / VER=8 bake).
                    try:
                        import comfy.ops as _ops_peel_load

                        def _closure_load_cell(fn, name: str):
                            try:
                                cells = fn.__closure__ or ()
                                for n, c in zip(fn.__code__.co_freevars, cells):
                                    if n == name:
                                        return c.cell_contents
                            except Exception:
                                return None
                            return None

                        def _is_foreign_int8_protect_load(fn) -> bool:
                            return bool(
                                getattr(fn, "_hswq_nvfp4_comfy_only", False)
                                or getattr(fn, "_hswq_int8_protect_in_load", False)
                                or getattr(fn, "_hswq_int8_protect_arm_v2", False)
                                or getattr(fn, "_hswq_int8_decode_patched", False)
                                or (
                                    getattr(fn, "_hswq_nvfp4_full_load", False)
                                    and not getattr(
                                        fn, "_hswq_nvfp4_product_tc", False
                                    )
                                )
                            )

                        def _next_load_under(fn):
                            for name in (
                                "cur",
                                "orig_load",
                                "original_load",
                                "_orig_load",
                            ):
                                nxt = _closure_load_cell(fn, name)
                                if nxt is not None:
                                    return nxt
                            return getattr(fn, "_hswq_nvfp4_orig_load", None)

                        _peeled_load_n = 0
                        _cur_l = getattr(
                            _ops_peel_load, "_load_quantized_module", None
                        )
                        _seen_l: set[int] = set()
                        while (
                            _cur_l is not None
                            and id(_cur_l) not in _seen_l
                            and _peeled_load_n < 16
                        ):
                            _seen_l.add(id(_cur_l))
                            if getattr(_cur_l, "_hswq_nvfp4_product_tc", False):
                                _under = _next_load_under(_cur_l)
                                if _under is not None and _is_foreign_int8_protect_load(
                                    _under
                                ):
                                    _ops_peel_load._load_quantized_module = _under
                                    _peeled_load_n += 1
                                    _cur_l = _under
                                    continue
                                break
                            if not _is_foreign_int8_protect_load(_cur_l):
                                break
                            _nxt_l = _next_load_under(_cur_l)
                            if _nxt_l is None or _nxt_l is _cur_l:
                                break
                            _ops_peel_load._load_quantized_module = _nxt_l
                            _peeled_load_n += 1
                            _cur_l = _nxt_l
                        if _peeled_load_n:
                            cleared.append(
                                f"int8_protect_load_peel={_peeled_load_n}"
                            )
                            print(
                                "HSWQ INT8/NVFP4: peeled ZI INT8-protect "
                                f"load overlay layers={_peeled_load_n}"
                            )
                    except Exception as e_load_peel:
                        print(
                            "HSWQ INT8/NVFP4: INT8-protect load peel failed: "
                            f"{e_load_peel}"
                        )

                    # --- Peel ZI/NVFP4 Linear.convert_weight / set_weight wraps ---
                    # uninstall_zimage_nvfp4_lora_bake only peels Dynamic.load /
                    # load_models_gpu. ZI attach_nvfp4_linear_lora_bake mutates
                    # MixedPrecision Linear in place; after ZI→SDXL the third SDXL
                    # bake still logs ConvRot int8_protect convert/set and LoRA
                    # strength dies. Restore stock (INT8) convert/set here.
                    def _peel_lora_bake_wrap_local(fn):
                        cur = fn
                        for _ in range(8):
                            if not callable(cur):
                                return cur
                            if int(
                                getattr(cur, "_hswq_nvfp4_lora_bake_ver", 0) or 0
                            ) <= 0:
                                return cur
                            stock = getattr(
                                cur, "_hswq_nvfp4_lora_bake_stock", None
                            )
                            if stock is not None and stock is not cur:
                                cur = stock
                                continue
                            closure = getattr(cur, "__closure__", None)
                            code = getattr(cur, "__code__", None)
                            if closure is None or code is None:
                                return cur
                            names = code.co_freevars
                            nxt = None
                            for i, nfree in enumerate(names):
                                if nfree in (
                                    "stock_convert_weight",
                                    "stock_set_weight",
                                ):
                                    nxt = closure[i].cell_contents
                                    break
                            if nxt is None or nxt is cur:
                                return cur
                            cur = nxt
                        return cur

                    peel_fn = _peel_lora_bake_wrap_local
                    for _pname, _pmod in list(__import__("sys").modules.items()):
                        if _pmod is None:
                            continue
                        _plow = str(_pname).replace("\\", "/").lower()
                        if (
                            "zi_nvfp4_forward" not in _plow
                            and not _plow.endswith("nvfp4_forward")
                            and ".nvfp4_forward" not in _plow
                        ):
                            continue
                        _helper = _safe_getattr(
                            _pmod, "_peel_lora_bake_wrap", None
                        )
                        if callable(_helper):
                            peel_fn = _helper
                            break

                    _peeled_lin_ids = set()

                    def _peel_linear_lora_bake(Lin, label: str) -> None:
                        if Lin is None or not isinstance(Lin, type):
                            return
                        lid = id(Lin)
                        if lid in _peeled_lin_ids:
                            return
                        for meth in ("convert_weight", "set_weight"):
                            fn = getattr(Lin, meth, None)
                            if not callable(fn):
                                continue
                            ver = int(
                                getattr(fn, "_hswq_nvfp4_lora_bake_ver", 0) or 0
                            )
                            if ver <= 0:
                                continue
                            try:
                                peeled = peel_fn(fn)
                            except Exception as e_peel_fn:
                                print(
                                    "HSWQ INT8/NVFP4: Linear LoRA bake peel "
                                    f"helper failed ({label}.{meth}): {e_peel_fn}"
                                )
                                continue
                            if peeled is fn or not callable(peeled):
                                continue
                            try:
                                setattr(Lin, meth, peeled)
                                _peeled_lin_ids.add(lid)
                                cleared.append(
                                    f"linear_lora_bake_peel@{label}.{meth}"
                                    f"=ver{ver}"
                                )
                                print(
                                    "HSWQ INT8/NVFP4: Peeled NVFP4/ZI Linear "
                                    f"LoRA bake wrap {label}.{meth} "
                                    f"(was ver={ver})"
                                )
                            except Exception as e_set:
                                print(
                                    "HSWQ INT8/NVFP4: Linear LoRA bake peel "
                                    f"setattr failed ({label}.{meth}): {e_set}"
                                )

                    try:
                        import comfy.ops as _comfy_ops

                        _peel_linear_lora_bake(
                            getattr(_comfy_ops, "Linear", None),
                            "comfy.ops.Linear",
                        )
                        for _an in dir(_comfy_ops):
                            try:
                                _obj = getattr(_comfy_ops, _an, None)
                            except Exception:
                                continue
                            if isinstance(_obj, type) and _safe_hasattr(
                                _obj, "convert_weight"
                            ):
                                _peel_linear_lora_bake(
                                    _obj, f"comfy.ops.{_an}"
                                )
                    except Exception as e_ops:
                        print(
                            "HSWQ INT8/NVFP4: comfy.ops Linear LoRA bake peel "
                            f"skipped: {e_ops}"
                        )

                    for _name, _mod in list(__import__("sys").modules.items()):
                        if _mod is None:
                            continue
                        _nlow = str(_name).replace("\\", "/").lower()
                        if not (
                            "comfy.ops" in _nlow
                            or "nvfp4" in _nlow
                            or "comfy_quant" in _nlow
                            or "zimage_nvfp4" in _nlow
                            or "hswq" in _nlow
                        ):
                            continue
                        try:
                            _Lin = _safe_getattr(_mod, "Linear", None)
                            if isinstance(_Lin, type):
                                _cvt = getattr(_Lin, "convert_weight", None)
                                if callable(_cvt) and int(
                                    getattr(
                                        _cvt, "_hswq_nvfp4_lora_bake_ver", 0
                                    )
                                    or 0
                                ) > 0:
                                    _peel_linear_lora_bake(
                                        _Lin, f"{_name}.Linear"
                                    )
                        except Exception:
                            pass

                    # Nested MixedPrecisionOps.Linear may not sit on a module.
                    try:
                        import gc as _gc_peel

                        for _obj in _gc_peel.get_objects():
                            if not isinstance(_obj, type):
                                continue
                            if getattr(_obj, "__name__", "") != "Linear":
                                continue
                            _modn = str(
                                getattr(_obj, "__module__", "") or ""
                            ).lower()
                            if not (
                                "comfy" in _modn
                                or "ops" in _modn
                                or "quant" in _modn
                            ):
                                continue
                            _cvt = getattr(_obj, "convert_weight", None)
                            if not (
                                callable(_cvt)
                                and int(
                                    getattr(
                                        _cvt, "_hswq_nvfp4_lora_bake_ver", 0
                                    )
                                    or 0
                                )
                                > 0
                            ):
                                continue
                            _peel_linear_lora_bake(
                                _obj, f"gc:{_modn}.Linear"
                            )
                    except Exception as e_gc_peel:
                        print(
                            "HSWQ INT8/NVFP4: gc Linear LoRA bake peel "
                            f"skipped: {e_gc_peel}"
                        )

                    # --- NVFP4 runtime pools + scale caches (SDXL + any twin) ---
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        try:
                            nlow = str(name).replace("\\", "/").lower()
                            name_hit = (
                                name.endswith("nvfp4_runtime")
                                or ".nvfp4_runtime" in name
                                or "nvfp4_runtime" in nlow
                            )
                            # Name-first: avoid probing unrelated modules whose
                            # __getattr__ raises (seedvr2 flashattention stubs).
                            if not name_hit and "nvfp4" not in nlow and "hswq" not in nlow:
                                continue
                            has_pool = (
                                _safe_hasattr(mod, "_ACT_Q_POOL")
                                or _safe_hasattr(mod, "_ROT_OUT_POOL")
                                or _safe_hasattr(mod, "_GRAPH_CACHE")
                                or _safe_hasattr(mod, "clear_nvfp4_runtime_pools")
                            )
                            if not name_hit and not has_pool:
                                continue
                            api_ok = False
                            fn = _safe_getattr(mod, "clear_nvfp4_runtime_pools", None)
                            if callable(fn):
                                try:
                                    fn()
                                    api_ok = True
                                    cleared.append(f"nvfp4_runtime_pools@{name}")
                                    print(
                                        "HSWQ INT8/NVFP4: Cleared HSWQ NVFP4 runtime pools / "
                                        f"CUDA graphs via {name}"
                                    )
                                except Exception as e2:
                                    print(
                                        f"HSWQ INT8/NVFP4: NVFP4 runtime pool clear failed "
                                        f"({name}): {e2}"
                                    )
                            n_pool = 0
                            for attr in (
                                "_ACT_Q_POOL",
                                "_ROT_OUT_POOL",
                                "_GRAPH_CACHE",
                                "_INV_NVFP4_AMAX_DENOM",
                                "_ONES_SCALE",
                            ):
                                n_pool += _clear_dict_attr(mod, attr)
                            if not api_ok:
                                cg = _safe_getattr(mod, "clear_nvfp4_cudagraphs", None)
                                if callable(cg):
                                    try:
                                        cg()
                                    except Exception:
                                        pass
                            if n_pool:
                                cleared.append(f"nvfp4_runtime_inplace={n_pool}@{name}")
                                if not api_ok:
                                    print(
                                        "HSWQ INT8/NVFP4: Cleared NVFP4 runtime dicts "
                                        f"in-place (n={n_pool}) via {name}"
                                    )
                        except Exception as e_pool:
                            print(
                                f"HSWQ INT8/NVFP4: NVFP4 runtime scan skip "
                                f"({name}): {e_pool}"
                            )

                    # --- Hadamard globals: SDXL nvfp4 + ZI zi_nvfp4 + INT8 ---
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        try:
                            nlow = str(name).replace("\\", "/").lower()
                            is_nv_h = (
                                name.endswith("nvfp4_hadamard")
                                or ".nvfp4_hadamard" in name
                                or "zi_nvfp4_hadamard" in nlow
                                or nlow.endswith("nvfp4_hadamard")
                            )
                            is_int8_h = (
                                "native_convert_int8" in nlow
                                or nlow.endswith("native_convert_int8")
                            )
                            if not is_nv_h and not is_int8_h:
                                if not (
                                    "hswq" in nlow
                                    or "nvfp4" in nlow
                                    or "zimage" in nlow
                                ):
                                    continue
                                has_h = (
                                    _safe_hasattr(mod, "_HADAMARD_CACHE")
                                    or _safe_hasattr(mod, "_H4_CACHE")
                                    or _safe_hasattr(
                                        mod, "clear_hadamard_global_caches"
                                    )
                                )
                                if not has_h:
                                    continue
                            fn = _safe_getattr(
                                mod, "clear_hadamard_global_caches", None
                            )
                            if callable(fn):
                                n_h = int(fn() or 0)
                                cleared.append(f"hadamard_api={n_h}@{name}")
                                print(
                                    "HSWQ INT8/NVFP4: Cleared Hadamard global caches "
                                    f"(n={n_h}) via {name}"
                                )
                            n_h2 = 0
                            for attr in ("_HADAMARD_CACHE", "_H4_CACHE"):
                                n_h2 += _clear_dict_attr(mod, attr)
                            if n_h2:
                                tag = "int8" if is_int8_h and not is_nv_h else "nvfp4"
                                cleared.append(
                                    f"hadamard_inplace_{tag}={n_h2}@{name}"
                                )
                                if not callable(fn):
                                    print(
                                        "HSWQ INT8/NVFP4: Cleared Hadamard dicts "
                                        f"in-place (n={n_h2}, {tag}) via {name}"
                                    )
                        except Exception as e_h:
                            print(
                                f"HSWQ INT8/NVFP4: Hadamard global clear "
                                f"failed ({name}): {e_h}"
                            )

                    # --- Loader parity / stats clear (best-effort; never gates local gc) ---
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        nlow = str(name).replace("\\", "/").lower()
                        if not (
                            "nvfp4_comfy_parity" in nlow
                            or "nvfp4_forward" in nlow
                            or "zi_nvfp4_forward" in nlow
                            or "comfy_quant_nvfp4" in nlow
                        ):
                            continue
                        for api_name in (
                            "clear_nvfp4_parity_hadamard_caches",
                            "reset_nvfp4_forward_stats",
                            "reset_nvfp4_lora_log_counters",
                        ):
                            fn = _safe_getattr(mod, api_name, None)
                            if not callable(fn):
                                continue
                            try:
                                ret = fn()
                                cleared.append(f"{api_name}@{name}={ret!r}")
                                print(
                                    "HSWQ INT8/NVFP4: "
                                    f"{api_name} via {name} -> {ret!r}"
                                )
                            except Exception as e2:
                                print(
                                    f"HSWQ INT8/NVFP4: {api_name} failed "
                                    f"({name}): {e2}"
                                )

                    # --- ALWAYS local gc: every HSWQ module residual (INT8+NVFP4+Detailer models) ---
                    local_dropped = 0
                    try:
                        for obj in gc.get_objects():
                            try:
                                if not isinstance(obj, torch.nn.Module):
                                    continue
                                for attr in _hswq_drop_attrs:
                                    try:
                                        if not hasattr(obj, attr):
                                            continue
                                        _empty_cuda_tensor(getattr(obj, attr, None))
                                        if _drop_attr(obj, attr):
                                            local_dropped += 1
                                    except Exception:
                                        pass
                                # Every ``_hswq_*`` residual (bool arms, sets, tensors, …)
                                try:
                                    for k, v in list(vars(obj).items()):
                                        if not (
                                            isinstance(k, str) and k.startswith("_hswq_")
                                        ):
                                            continue
                                        if k in _hswq_drop_attrs:
                                            continue
                                        if torch.is_tensor(v):
                                            _empty_cuda_tensor(v)
                                        if _drop_attr(obj, k):
                                            local_dropped += 1
                                except Exception:
                                    pass
                            except Exception:
                                continue
                        if local_dropped:
                            cleared.append(f"hswq_module_attrs_gc={local_dropped}")
                            print(
                                "HSWQ INT8/NVFP4: Local gc dropped HSWQ module attrs "
                                f"(n={local_dropped}; INT8+NVFP4 H/bake/arm + all _hswq_*)"
                            )
                    except Exception as e3:
                        print(
                            f"HSWQ INT8/NVFP4: HSWQ module attr gc clear skipped: {e3}"
                        )

                    if cleared:
                        print(
                            "HSWQ INT8/NVFP4: Reset HSWQ/comfy_kitchen caches "
                            + ", ".join(cleared)
                        )
                    else:
                        print("HSWQ INT8/NVFP4: HSWQ/comfy_kitchen caches already empty")

                def _force_unregister_comfy_pins() -> int:
                    """Unregister every cudaHostRegister tracked by ComfyUI PINNED_MEMORY."""
                    nonlocal pins_unregistered
                    freed = 0
                    try:
                        import comfy.model_management as mm
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: cannot import model_management for pin nuke: {e}")
                        return 0
                    pinned = getattr(mm, "PINNED_MEMORY", None)
                    if not isinstance(pinned, dict):
                        print("HSWQ INT8/NVFP4: PINNED_MEMORY dict missing")
                        return 0
                    before = int(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0)
                    print(
                        f"HSWQ INT8/NVFP4: Force-unregister PINNED_MEMORY "
                        f"before={before / (1024 * 1024):.1f} MB entries={len(pinned)}"
                    )
                    for ptr, size in list(pinned.items()):
                        try:
                            if torch.cuda.cudart().cudaHostUnregister(int(ptr)) == 0:
                                pins_unregistered += 1
                                freed += int(size or 0)
                            else:
                                try:
                                    mm.discard_cuda_async_error()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            pinned.pop(ptr, None)
                        except Exception:
                            pass
                    try:
                        mm.TOTAL_PINNED_MEMORY = 0
                    except Exception:
                        pass
                    print(
                        f"HSWQ INT8/NVFP4: Force-unregister done "
                        f"unregistered={pins_unregistered} "
                        f"approx={freed / (1024 * 1024):.1f} MB"
                    )
                    return freed

                def _is_real_nn(obj) -> bool:
                    try:
                        return isinstance(obj, torch.nn.Module)
                    except Exception:
                        return False

                def _unwrap_nn(obj):
                    cur = obj
                    for _ in range(8):
                        if cur is None:
                            return None
                        if _is_real_nn(cur):
                            return cur
                        nxt = getattr(cur, "model", None)
                        if nxt is None or nxt is cur:
                            nxt = getattr(cur, "diffusion_model", None)
                        if nxt is None or nxt is cur:
                            return cur if _is_real_nn(cur) else None
                        cur = nxt
                    return cur if _is_real_nn(cur) else None

                def _is_hswq_int8_nn(module) -> bool:
                    """True for HSWQ INT8 and/or NVFP4 (incl. ZI ConvRot) UNet modules.

                    Pure NVFP4 packs have ``format=nvfp4`` comfy_quant markers and
                    ``_hswq_nvfp4_convrot`` arms — they are not ``int8_tensorwise``.
                    Detecting only INT8 left ZI ConvRot models half-purged: Method 3
                    nuked CUDA tensors while live Modules kept dead
                    ``_hswq_nvfp4_parity_H`` → 2nd gen noise.
                    """
                    if module is None or not _is_real_nn(module):
                        return False
                    baked = getattr(module, "_hswq_int8_baked_keys", None)
                    if baked:
                        return True
                    if getattr(module, "_hswq_int8_baked_uuid", None) is not None:
                        return True
                    # Z Image Dynamic LoRA NVFP4 bake bookkeeping (separate from INT8).
                    if getattr(module, "_hswq_zi_nvfp4_baked_keys", None):
                        return True
                    if getattr(module, "_hswq_zi_nvfp4_baked_uuid", None) is not None:
                        return True
                    try:
                        for m in module.modules():
                            if (
                                getattr(m, "_hswq_nvfp4_convrot", False)
                                or getattr(m, "_hswq_nvfp4", False)
                                or getattr(m, "_hswq_int8_convrot", False)
                                # INT8 Conv2d ConvRot (QuantConv2d) — not Linear protect
                                or getattr(m, "_hswq_convrot", False)
                                or getattr(m, "_hswq_nvfp4_parity_H", None) is not None
                                or getattr(m, "_hswq_nvfp4_H", None) is not None
                                or getattr(m, "_hswq_nvfp4_w_plain", None) is not None
                                or getattr(m, "_hswq_zi_nvfp4_baked_keys", None)
                                or getattr(m, "_hswq_zi_nvfp4_baked_uuid", None) is not None
                            ):
                                return True
                            # Any residual ``_hswq_*`` (INT8 / NVFP4 / bake / TC caches)
                            try:
                                for k in vars(m):
                                    if isinstance(k, str) and k.startswith("_hswq_"):
                                        return True
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        for name, buf in module.named_buffers():
                            if not (name.endswith("comfy_quant") or name.endswith(".comfy_quant")):
                                continue
                            try:
                                raw = buf.detach().cpu()
                                if raw.dtype == torch.uint8 and raw.numel() > 0:
                                    import json
                                    conf = json.loads(bytes(raw.tolist()).decode("utf-8", errors="ignore"))
                                    if isinstance(conf, dict):
                                        fmt = conf.get("format")
                                        if fmt in ("int8_tensorwise", "nvfp4"):
                                            return True
                                        if "format" in conf:
                                            # Known non-HSWQ format → keep scanning
                                            continue
                            except Exception:
                                pass
                            # Unparseable comfy_quant on an HSWQ pack → treat as hit
                            return True
                    except Exception:
                        pass
                    return False

                def _loaded_holds_hswq_int8(loaded_model) -> bool:
                    if loaded_model is None:
                        return False
                    for attr in ("model", "real_model"):
                        try:
                            v = getattr(loaded_model, attr, None)
                            if callable(v):
                                try:
                                    v = v()
                                except Exception:
                                    v = None
                            nn = _unwrap_nn(v)
                            if _is_hswq_int8_nn(nn):
                                return True
                            inner = getattr(v, "model", None) if v is not None else None
                            if _is_hswq_int8_nn(_unwrap_nn(inner)):
                                return True
                        except Exception:
                            pass
                    return False

                def _kill_tensor_storage(t) -> int:
                    # Same rule as _force_empty_cuda_storage: unpin/CPU-safe only;
                    # empty(0) only for CUDA. CPU wipe broke ZI TE after Ollama purge.
                    if t is None:
                        return 0
                    freed = 0
                    try:
                        data = getattr(t, "data", t)
                        if data is None:
                            return 0
                        nbytes = int(getattr(data, "nbytes", 0) or 0)
                        is_cuda = False
                        try:
                            is_cuda = bool(getattr(data, "is_cuda", False))
                            if not is_cuda:
                                dev = getattr(data, "device", None)
                                is_cuda = getattr(dev, "type", None) == "cuda"
                        except Exception:
                            pass
                        try:
                            if bool(getattr(data, "is_pinned", lambda: False)()):
                                try:
                                    import comfy.model_management as mm
                                    mm.unpin_memory(data)
                                except Exception:
                                    try:
                                        torch.cuda.cudart().cudaHostUnregister(data.data_ptr())
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        if not is_cuda:
                            return 0
                        dtype = getattr(data, "dtype", torch.float32)
                        empty = torch.empty(0, dtype=dtype, device="cpu")
                        if hasattr(t, "data"):
                            t.data = empty
                        freed = nbytes
                    except Exception:
                        pass
                    return freed

                def _kill_module_vram(module, label: str) -> int:
                    freed = 0
                    print(f"HSWQ INT8/NVFP4: Killing module VRAM ({label}) type={type(module).__name__}")
                    try:
                        if hasattr(module, "to") and callable(module.to):
                            try:
                                module.to("cpu")
                            except Exception as e:
                                print(f"HSWQ INT8/NVFP4: .to('cpu') warning ({label}): {e}")
                    except Exception:
                        pass
                    try:
                        for _n, p in list(module.named_parameters()):
                            freed += _kill_tensor_storage(p)
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: param kill warning ({label}): {e}")
                    try:
                        for _n, b in list(module.named_buffers()):
                            freed += _kill_tensor_storage(b)
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: buffer kill warning ({label}): {e}")
                    # Drop ALL HSWQ module residuals (cache + bake + arm), every submodule.
                    # Loader presence does not matter — next load re-arms ConvRot/TC.
                    # INT8 (Linear + Conv2d) + NVFP4 / ZI + TC forward caches
                    _hswq_kill_attrs = (
                        "_hswq_nvfp4_parity_H",
                        "_hswq_nvfp4_H",
                        "_hswq_nvfp4_w_plain",
                        "_hswq_nvfp4_alpha",
                        "_hswq_nvfp4_no_cudagraph",
                        "_hswq_int8_baked_keys",
                        "_hswq_int8_baked_uuid",
                        "_hswq_zi_nvfp4_baked_keys",
                        "_hswq_zi_nvfp4_baked_uuid",
                        "_hswq_nvfp4_convrot",
                        "_hswq_nvfp4_convrot_groupsize",
                        "_hswq_nvfp4",
                        "_hswq_int8_convrot",
                        "_hswq_int8_convrot_groupsize",
                        "_hswq_convrot",
                        "_hswq_convrot_groupsize",
                    )
                    try:
                        for m in module.modules():
                            for attr in _hswq_kill_attrs:
                                if not hasattr(m, attr):
                                    continue
                                try:
                                    val = getattr(m, attr, None)
                                    if torch.is_tensor(val):
                                        freed += _kill_tensor_storage(val)
                                except Exception:
                                    pass
                                try:
                                    delattr(m, attr)
                                except Exception:
                                    try:
                                        setattr(m, attr, None)
                                    except Exception:
                                        pass
                            # Every residual ``_hswq_*`` (bool arms, sets, tensors, …)
                            try:
                                for k, v in list(vars(m).items()):
                                    if not (isinstance(k, str) and k.startswith("_hswq_")):
                                        continue
                                    if k in _hswq_kill_attrs:
                                        continue
                                    if torch.is_tensor(v):
                                        freed += _kill_tensor_storage(v)
                                    try:
                                        delattr(m, k)
                                    except Exception:
                                        try:
                                            setattr(m, k, None)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: HSWQ attr kill warning ({label}): {e}")
                    print(f"HSWQ INT8/NVFP4: Killed ~{freed / (1024 * 1024):.1f} MB CUDA storage ({label})")
                    return freed

                def _unload_patcher(obj) -> int:
                    nonlocal patchers_unloaded
                    freed = 0
                    try:
                        if hasattr(obj, "partially_unload_ram") and callable(obj.partially_unload_ram):
                            try:
                                freed += int(obj.partially_unload_ram(1e30) or 0)
                            except Exception:
                                pass
                        if hasattr(obj, "unregister_inactive_pins") and callable(obj.unregister_inactive_pins):
                            try:
                                freed += int(obj.unregister_inactive_pins(1e30) or 0)
                            except Exception:
                                pass
                        if hasattr(obj, "partially_unload") and callable(obj.partially_unload):
                            try:
                                obj.partially_unload(None, 1e30)
                            except Exception:
                                try:
                                    obj.partially_unload(torch.device("cpu"), 1e30)
                                except Exception:
                                    pass
                        if hasattr(obj, "unpatch_model") and callable(obj.unpatch_model):
                            try:
                                obj.unpatch_model(torch.device("cpu"), unpatch_weights=True)
                            except Exception:
                                pass
                        if hasattr(obj, "model_unload") and callable(obj.model_unload):
                            try:
                                obj.model_unload()
                            except Exception:
                                pass
                        patchers_unloaded += 1
                    except Exception:
                        pass
                    return freed

                _mem_diag("before")

                # 0) Batched Detailer pin pool
                print("HSWQ INT8/NVFP4: Method 0 - Draining HSWQ Batched Detailer PinCache...")
                bytes_killed += _drain_hswq_pin_cache()
                print("HSWQ INT8/NVFP4: Method 0s - Detailer SEGS / PromptExecutor cache...")
                bytes_killed += _purge_detailer_segs_and_executor_cache()

                # 1) ComfyUI loaded models (INT8 first, then unload everything)
                print("HSWQ INT8/NVFP4: Method 1 - current_loaded_models...")
                models_checked_mm = 0
                models_found_mm = 0
                try:
                    import comfy.model_management as mm
                    if hasattr(mm, "current_loaded_models"):
                        current_loaded_models = mm.current_loaded_models
                        print(f"HSWQ INT8/NVFP4: current_loaded_models count={len(current_loaded_models)}")
                        for i in range(len(current_loaded_models) - 1, -1, -1):
                            loaded_model = current_loaded_models[i]
                            models_checked_mm += 1
                            try:
                                is_int8 = _loaded_holds_hswq_int8(loaded_model)
                                if is_int8:
                                    models_found_mm += 1
                                    print(
                                        f"HSWQ INT8/NVFP4: Found HSWQ INT8/NVFP4 at current_loaded_models[{i}] "
                                        f"type={type(loaded_model).__name__}"
                                    )
                                    try:
                                        loaded_model.currently_used = False
                                    except Exception:
                                        pass
                                    nn = None
                                    try:
                                        nn = _unwrap_nn(getattr(loaded_model, "model", None))
                                    except Exception:
                                        nn = None
                                    if nn is not None:
                                        bytes_killed += _kill_module_vram(nn, f"current_loaded_models[{i}]")
                                    hswq_cleared += 1
                                # Always tear down pin/hostbuf on every loaded model
                                try:
                                    inner = getattr(loaded_model, "model", None)
                                    if inner is not None:
                                        bytes_killed += _unload_patcher(inner)
                                except Exception:
                                    pass
                                try:
                                    if hasattr(loaded_model, "model_unload") and callable(loaded_model.model_unload):
                                        loaded_model.model_unload()
                                except Exception as e:
                                    print(f"HSWQ INT8/NVFP4: model_unload warning: {e}")
                                current_loaded_models.pop(i)
                                print(f"HSWQ INT8/NVFP4: Removed current_loaded_models[{i}] (int8={is_int8})")
                            except Exception as e:
                                print(f"HSWQ INT8/NVFP4: Error at current_loaded_models[{i}]: {e}")
                    try:
                        if hasattr(mm, "unload_all_models") and callable(mm.unload_all_models):
                            mm.unload_all_models()
                            print("HSWQ INT8/NVFP4: unload_all_models() issued")
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: unload_all_models warning: {e}")
                    try:
                        if torch.cuda.is_available() and hasattr(mm, "free_memory"):
                            for di in range(torch.cuda.device_count()):
                                mm.free_memory(1e30, torch.device(f"cuda:{di}"))
                            print("HSWQ INT8/NVFP4: free_memory(1e30) issued for all CUDA devices")
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: free_memory warning: {e}")
                    if hasattr(mm, "cleanup_models_gc") and callable(mm.cleanup_models_gc):
                        try:
                            mm.cleanup_models_gc()
                        except Exception as e:
                            print(f"HSWQ INT8/NVFP4: cleanup_models_gc warning: {e}")
                except Exception as e:
                    print(f"HSWQ INT8/NVFP4: Error in Method 1: {e}")
                    import traceback
                    print(f"HSWQ INT8/NVFP4: Traceback: {traceback.format_exc()}")
                print(
                    f"HSWQ INT8/NVFP4: Method 1 complete - checked {models_checked_mm}, found {models_found_mm}"
                )
                _mem_diag("after_method1")

                # 2) Force HostUnregister every ComfyUI-tracked pin (NOT sys.modules dir/getattr —
                #    that triggers kornia LazyLoader basicsr install prompts)
                print("HSWQ INT8/NVFP4: Method 2 - Force HostUnregister PINNED_MEMORY...")
                bytes_killed += _force_unregister_comfy_pins()
                _mem_diag("after_method2")

                # 3) gc nuclear: INT8 modules + ModelPatchers + pinned/CUDA tensors
                print("HSWQ INT8/NVFP4: Method 3 - gc nuclear (no sys.modules getattr)...")
                objects_checked = 0
                models_found_in_gc = 0
                try:
                    import comfy.model_management as mm
                except Exception:
                    mm = None
                try:
                    for obj in gc.get_objects():
                        objects_checked += 1
                        if objects_checked > 500000:
                            print("HSWQ INT8/NVFP4: gc scan limit 500000")
                            break
                        try:
                            tname = type(obj).__name__
                            if tname in (
                                "ModelPatcher",
                                "ModelPatcherDynamic",
                                "LoadedModel",
                            ):
                                bytes_killed += _unload_patcher(obj)
                                continue
                            if _is_real_nn(obj) and _is_hswq_int8_nn(obj):
                                models_found_in_gc += 1
                                hswq_cleared += 1
                                print(f"HSWQ INT8/NVFP4: Found HSWQ INT8/NVFP4 in gc type={tname}")
                                bytes_killed += _kill_module_vram(obj, f"gc:{tname}")
                                continue
                            if torch.is_tensor(obj):
                                nbytes = int(getattr(obj, "nbytes", 0) or 0)
                                if nbytes < 1024 * 1024:
                                    continue
                                try:
                                    if bool(obj.is_pinned()):
                                        try:
                                            if mm is not None:
                                                mm.unpin_memory(obj)
                                        except Exception:
                                            try:
                                                torch.cuda.cudart().cudaHostUnregister(obj.data_ptr())
                                            except Exception:
                                                pass
                                        pins_unregistered += 1
                                        bytes_killed += nbytes
                                except Exception:
                                    pass
                                try:
                                    if bool(obj.is_cuda):
                                        cuda_tensors_killed += 1
                                        bytes_killed += _kill_tensor_storage(obj)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception as e:
                    print(f"HSWQ INT8/NVFP4: Error in Method 3: {e}")
                    import traceback
                    print(f"HSWQ INT8/NVFP4: Traceback: {traceback.format_exc()}")
                print(
                    f"HSWQ INT8/NVFP4: Method 3 complete - checked {objects_checked}, "
                    f"int8={models_found_in_gc}, patchers={patchers_unloaded}, "
                    f"cuda_tensors={cuda_tensors_killed}"
                )

                # Second PinCache drain + second PINNED_MEMORY sweep
                print("HSWQ INT8/NVFP4: Method 0b - Second PinCache drain...")
                bytes_killed += _drain_hswq_pin_cache()
                print("HSWQ INT8/NVFP4: Method 0s2 - Second Detailer SEGS / executor sweep...")
                bytes_killed += _purge_detailer_segs_and_executor_cache()
                print("HSWQ INT8/NVFP4: Method 2b - Second PINNED_MEMORY sweep...")
                bytes_killed += _force_unregister_comfy_pins()

                # Nuclear CUDA tensor kill may have destroyed kitchen workspaces
                # while leaving dead refs in module-level dicts — clear them.
                print("HSWQ INT8/NVFP4: Method 2c - Reset comfy_kitchen CUDA caches...")
                _reset_comfy_kitchen_cuda_caches()

                # Reset INT8 LoRA counters (dict-only, no dir())
                print("HSWQ INT8/NVFP4: Resetting comfy_quant_int8 counters...")
                try:
                    for mod_name, mod in _sys_modules():
                        if mod is None or "comfy_quant_int8" not in str(mod_name):
                            continue
                        d = getattr(mod, "__dict__", None)
                        if not isinstance(d, dict):
                            continue
                        reset_fn = d.get("reset_int8_lora_log_counters")
                        if callable(reset_fn):
                            print(f"HSWQ INT8/NVFP4: Calling reset_int8_lora_log_counters via {mod_name}")
                            reset_fn()
                            break
                except Exception as e:
                    print(f"HSWQ INT8/NVFP4: counter reset skipped: {e}")

                print("HSWQ INT8/NVFP4: Running garbage collection...")
                gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    print("HSWQ INT8/NVFP4: Clearing CUDA cache...")
                    for device_idx in range(torch.cuda.device_count()):
                        with torch.cuda.device(device_idx):
                            torch.cuda.empty_cache()
                            try:
                                torch.cuda.ipc_collect()
                            except Exception:
                                pass
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    print("HSWQ INT8/NVFP4: CUDA cache cleared for all devices")
                else:
                    print("HSWQ INT8/NVFP4: CUDA not available, skipped CUDA cache clear")

                try:
                    import comfy.model_management as mm
                    if hasattr(mm, "soft_empty_cache") and callable(mm.soft_empty_cache):
                        try:
                            mm.soft_empty_cache(True)
                        except TypeError:
                            mm.soft_empty_cache()
                except Exception:
                    pass

                _mem_diag("after")
                print(
                    f"HSWQ INT8/NVFP4: Done — cleared {hswq_cleared} HSWQ INT8/NVFP4 ref(s), "
                    f"pins_unregistered={pins_unregistered}, "
                    f"patchers={patchers_unloaded}, "
                    f"cuda_tensors={cuda_tensors_killed}, "
                    f"approx {bytes_killed / (1024 * 1024):.1f} MB tracked"
                )

            except Exception as e:
                print(f"HSWQ INT8/NVFP4: Error purging models: {e}")
                import traceback
                print(f"HSWQ INT8/NVFP4: Traceback: {traceback.format_exc()}")
```

### 3.3 Ollama purge (only when Ollama toggle ON)

**Source:** `nodes/purge_vram.py` at tip `f5c30e9`, **lines 3568-4000** (inclusive).

```python
        # Purge Ollama VRAM for comfyui-ollama + comfyui-ollama-describer.
        # Both packs talk to the external Ollama server (no in-process torch models).
        # describer defaults keep_model_alive=-1 (indefinite load) — must empty /api/ps.
        if purge_ollama:
            try:
                print("Ollama: Starting FULL purge (zero residual; includes comfyui-ollama-describer)...")
                import json
                import shutil
                import subprocess
                import time
                import urllib.error
                import urllib.request

                def _is_comfyui_ollama_module(mod_name: str) -> bool:
                    n = str(mod_name).replace("\\", "/").lower()
                    return (
                        "compfyuiollama" in n
                        or "comfyui-ollama" in n
                        or "comfyui_ollama" in n
                        or "ollama-describer" in n
                        or "ollama_describer" in n
                        or (n.endswith("deprecated_nodes") and "ollama" in n)
                    )

                def _normalize_ollama_url(host: str) -> str:
                    host = str(host).strip().rstrip("/")
                    if not host:
                        return ""
                    if not host.startswith("http"):
                        host = "http://" + host
                    return host.rstrip("/")

                def _ollama_urls() -> list:
                    urls = []
                    env_host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL")
                    if env_host:
                        u = _normalize_ollama_url(env_host)
                        if u:
                            urls.append(u)
                    # Harvest URLs from loaded ollama custom-node modules / node state
                    for mod_name, mod in list(sys.modules.items()):
                        if mod is None or not _is_comfyui_ollama_module(mod_name):
                            continue
                        print(f"Ollama: scanning module {mod_name}")
                        for attr in ("DEFAULT_URL", "OLLAMA_URL", "url", "base_url", "api_host"):
                            v = getattr(mod, attr, None)
                            if isinstance(v, str) and ("http://" in v or "https://" in v or "11434" in v):
                                u = _normalize_ollama_url(v)
                                if u:
                                    urls.append(u)
                        # comfyui-ollama-describer: class INPUT_TYPES defaults for api_host
                        for cls_name in (
                            "OllamaImageCaptioner",
                            "OllamaImageDescriber",
                            "OllamaTextDescriber",
                            "OllamaConnectivityV2",
                            "OllamaGenerateV2",
                        ):
                            cls = getattr(mod, cls_name, None)
                            if cls is None or not hasattr(cls, "INPUT_TYPES"):
                                continue
                            try:
                                spec = cls.INPUT_TYPES()
                                required = (spec or {}).get("required") or {}
                                for key in ("api_host", "url"):
                                    entry = required.get(key)
                                    if (
                                        isinstance(entry, tuple)
                                        and len(entry) >= 2
                                        and isinstance(entry[1], dict)
                                    ):
                                        default = entry[1].get("default")
                                        if isinstance(default, str) and default.strip():
                                            u = _normalize_ollama_url(default)
                                            if u:
                                                urls.append(u)
                            except Exception:
                                pass
                    try:
                        for obj in gc.get_objects():
                            try:
                                if not hasattr(obj, "__class__"):
                                    continue
                                cname = type(obj).__name__
                                # Client from ollama package (used by both packs)
                                if cname == "Client" and hasattr(obj, "host"):
                                    v = getattr(obj, "host", None)
                                    if isinstance(v, str) and v.strip():
                                        u = _normalize_ollama_url(v)
                                        if u:
                                            urls.append(u)
                                if "Ollama" not in cname and "ollama" not in cname.lower():
                                    if not (cname == "OllamaUtil"):
                                        continue
                                # describer uses api_host; comfyui-ollama uses url
                                for attr in ("url", "_url", "host", "base_url", "api_host"):
                                    v = getattr(obj, attr, None)
                                    if isinstance(v, str) and v.strip():
                                        u = _normalize_ollama_url(v)
                                        if u:
                                            urls.append(u)
                                # Nested client.host on OllamaUtil
                                client = getattr(obj, "client", None)
                                if client is not None:
                                    for attr in ("host", "base_url", "url"):
                                        v = getattr(client, attr, None)
                                        if isinstance(v, str) and v.strip():
                                            u = _normalize_ollama_url(v)
                                            if u:
                                                urls.append(u)
                                # OllamaConnectivityV2 returns dict via execute; instances may stash meta
                                for attr in ("saved_meta", "meta", "connectivity"):
                                    bag = getattr(obj, attr, None)
                                    if isinstance(bag, dict):
                                        u = bag.get("url") or bag.get("api_host")
                                        if not u and isinstance(bag.get("connectivity"), dict):
                                            u = bag["connectivity"].get("url") or bag["connectivity"].get(
                                                "api_host"
                                            )
                                        if isinstance(u, str) and u.strip():
                                            nu = _normalize_ollama_url(u)
                                            if nu:
                                                urls.append(nu)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Ollama: URL harvest via gc skipped: {e}")
                    urls.append("http://127.0.0.1:11434")
                    urls.append("http://localhost:11434")
                    seen = set()
                    out = []
                    for u in urls:
                        u = str(u).rstrip("/")
                        if u and u not in seen:
                            seen.add(u)
                            out.append(u)
                    return out

                def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 30.0):
                    data = None
                    headers = {"Accept": "application/json"}
                    if body is not None:
                        data = json.dumps(body).encode("utf-8")
                        headers["Content-Type"] = "application/json"
                    req = urllib.request.Request(url, data=data, headers=headers, method=method)
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        raw = resp.read()
                        if not raw:
                            return None
                        return json.loads(raw.decode("utf-8", errors="ignore"))

                def _model_names_from_ps(payload) -> list:
                    names = []
                    if payload is None:
                        return names
                    if isinstance(payload, dict):
                        models = payload.get("models")
                    elif isinstance(payload, list):
                        models = payload
                    else:
                        models = getattr(payload, "models", None)
                    for m in models or []:
                        name = None
                        if isinstance(m, dict):
                            name = m.get("model") or m.get("name")
                        else:
                            name = getattr(m, "model", None) or getattr(m, "name", None)
                        if name:
                            names.append(str(name))
                    # unique, stable
                    seen = set()
                    out = []
                    for n in names:
                        if n not in seen:
                            seen.add(n)
                            out.append(n)
                    return out

                def _ps_names(base_url: str) -> list:
                    ps_url = base_url.rstrip("/") + "/api/ps"
                    try:
                        return _model_names_from_ps(_http_json("GET", ps_url, None, timeout=10.0))
                    except Exception as e:
                        print(f"Ollama: GET {ps_url} failed: {e}")
                        return []

                def _unload_one_http(base_url: str, name: str) -> None:
                    root = base_url.rstrip("/")
                    # Official unload: empty prompt/messages + keep_alive 0 (int and "0"), stream false
                    bodies_gen = (
                        {"model": name, "prompt": "", "keep_alive": 0, "stream": False},
                        {"model": name, "keep_alive": 0, "stream": False},
                        {"model": name, "prompt": "", "keep_alive": "0", "stream": False},
                    )
                    for body in bodies_gen:
                        try:
                            _http_json("POST", root + "/api/generate", body, timeout=120.0)
                            print(f"Ollama: /api/generate keep_alive=0 -> {name}")
                            break
                        except Exception as e:
                            print(f"Ollama: /api/generate unload try failed for {name}: {e}")
                    bodies_chat = (
                        {"model": name, "messages": [], "keep_alive": 0, "stream": False},
                        {"model": name, "messages": [], "keep_alive": "0", "stream": False},
                    )
                    for body in bodies_chat:
                        try:
                            _http_json("POST", root + "/api/chat", body, timeout=120.0)
                            print(f"Ollama: /api/chat keep_alive=0 -> {name}")
                            break
                        except Exception as e:
                            print(f"Ollama: /api/chat unload try failed for {name}: {e}")

                def _unload_one_client(client, name: str) -> None:
                    for kwargs in (
                        {"model": name, "prompt": "", "keep_alive": 0},
                        {"model": name, "prompt": "", "keep_alive": "0"},
                    ):
                        try:
                            client.generate(**kwargs)
                            print(f"Ollama: Client.generate keep_alive=0 -> {name}")
                            break
                        except Exception as e:
                            print(f"Ollama: Client.generate failed for {name}: {e}")
                    if hasattr(client, "chat") and callable(client.chat):
                        for kwargs in (
                            {"model": name, "messages": [], "keep_alive": 0},
                            {"model": name, "messages": [], "keep_alive": "0"},
                        ):
                            try:
                                client.chat(**kwargs)
                                print(f"Ollama: Client.chat keep_alive=0 -> {name}")
                                break
                            except Exception as e:
                                print(f"Ollama: Client.chat failed for {name}: {e}")

                def _ollama_stop_cli(name: str) -> None:
                    try:
                        r = subprocess.run(
                            ["ollama", "stop", name],
                            capture_output=True,
                            text=True,
                            timeout=60,
                            shell=False,
                        )
                        print(
                            f"Ollama: CLI stop {name} rc={r.returncode} "
                            f"stdout={(r.stdout or '').strip()[:200]} "
                            f"stderr={(r.stderr or '').strip()[:200]}"
                        )
                    except FileNotFoundError:
                        print("Ollama: CLI `ollama` not on PATH (skipped)")
                    except Exception as e:
                        print(f"Ollama: CLI stop failed for {name}: {e}")

                def _purge_server_until_empty(base_url: str, max_rounds: int = 8) -> tuple:
                    """Return (unload_attempts, residual_names). residual must be [] for success."""
                    attempts = 0
                    client = None
                    try:
                        from ollama import Client

                        client = Client(host=base_url)
                    except Exception as e:
                        print(f"Ollama: Client optional ({base_url}): {e}")

                    residual = _ps_names(base_url)
                    print(f"Ollama: {base_url} initial loaded={residual if residual else '(none)'}")
                    for round_i in range(1, max_rounds + 1):
                        residual = _ps_names(base_url)
                        if not residual and client is not None and hasattr(client, "ps"):
                            try:
                                residual = _model_names_from_ps(client.ps())
                            except Exception:
                                pass
                        if not residual:
                            print(f"Ollama: {base_url} EMPTY after round {round_i - 1} (verified /api/ps)")
                            return attempts, []
                        print(f"Ollama: {base_url} round {round_i}/{max_rounds} still loaded={residual}")
                        for name in list(residual):
                            attempts += 1
                            _unload_one_http(base_url, name)
                            if client is not None:
                                _unload_one_client(client, name)
                            _ollama_stop_cli(name)
                        time.sleep(0.35)
                    residual = _ps_names(base_url)
                    if residual:
                        print(f"Ollama: WARNING residual still loaded on {base_url}: {residual}")
                    return attempts, residual

                total_unloaded = 0
                residual_all = []
                for base in _ollama_urls():
                    print(f"Ollama: Purging server at {base}...")
                    n, residual = _purge_server_until_empty(base)
                    total_unloaded += n
                    for r in residual:
                        if r not in residual_all:
                            residual_all.append(r)

                # --- Comfy-side comfyui-ollama state: leave nothing ---
                sessions_cleared = 0
                context_cleared = 0
                files_wiped = 0
                instances_cleared = 0

                for mod_name, mod in list(sys.modules.items()):
                    if mod is None or not _is_comfyui_ollama_module(mod_name):
                        continue
                    try:
                        bag = getattr(mod, "CHAT_SESSIONS", None)
                        if isinstance(bag, dict):
                            for _sid, sess in list(bag.items()):
                                try:
                                    msgs = getattr(sess, "messages", None)
                                    if isinstance(msgs, list):
                                        msgs.clear()
                                    if hasattr(sess, "model"):
                                        setattr(sess, "model", "")
                                except Exception:
                                    pass
                            sessions_cleared += len(bag)
                            bag.clear()
                            print(f"Ollama: Cleared CHAT_SESSIONS via {mod_name} entries={sessions_cleared}")
                    except Exception as e:
                        print(f"Ollama: CHAT_SESSIONS clear failed ({mod_name}): {e}")

                    for attr in (
                        "OllamaGenerateAdvance",
                        "OllamaGenerate",
                        "OllamaVision",
                        "OllamaGenerateV2",
                        "OllamaChat",
                        "OllamaSaveContext",
                        "OllamaLoadContext",
                    ):
                        try:
                            cls = getattr(mod, attr, None)
                            if cls is None:
                                continue
                            if hasattr(cls, "saved_context"):
                                setattr(cls, "saved_context", None)
                                context_cleared += 1
                        except Exception as e:
                            print(f"Ollama: class {attr} clear failed ({mod_name}): {e}")

                    # Wipe on-disk saved_context artifacts under the custom node (keep .keep only)
                    try:
                        mod_file = getattr(mod, "__file__", None)
                        if mod_file:
                            base_dir = os.path.dirname(os.path.realpath(mod_file))
                            ctx_dir = os.path.join(base_dir, "saved_context")
                            if os.path.isdir(ctx_dir):
                                for fn in os.listdir(ctx_dir):
                                    if fn == ".keep":
                                        continue
                                    path = os.path.join(ctx_dir, fn)
                                    try:
                                        if os.path.isfile(path) or os.path.islink(path):
                                            os.remove(path)
                                            files_wiped += 1
                                        elif os.path.isdir(path):
                                            shutil.rmtree(path)
                                            files_wiped += 1
                                    except Exception as e:
                                        print(f"Ollama: wipe file failed {path}: {e}")
                                print(f"Ollama: wiped saved_context dir files via {mod_name} count={files_wiped}")
                    except Exception as e:
                        print(f"Ollama: saved_context dir wipe failed ({mod_name}): {e}")

                # Instance-level saved_context / message buffers on live Ollama node objects
                try:
                    for obj in gc.get_objects():
                        try:
                            if not hasattr(obj, "__class__"):
                                continue
                            cname = type(obj).__name__
                            if "Ollama" not in cname:
                                continue
                            changed = False
                            if hasattr(obj, "saved_context"):
                                setattr(obj, "saved_context", None)
                                changed = True
                            for attr in ("messages", "context", "history", "chat_history"):
                                if hasattr(obj, attr):
                                    val = getattr(obj, attr)
                                    if isinstance(val, list):
                                        val.clear()
                                        changed = True
                                    elif val is not None and attr != "messages":
                                        try:
                                            setattr(obj, attr, None)
                                            changed = True
                                        except Exception:
                                            pass
                            if changed:
                                instances_cleared += 1
                        except Exception:
                            pass
                    if instances_cleared:
                        print(f"Ollama: Cleared instance state on {instances_cleared} Ollama objects")
                except Exception as e:
                    print(f"Ollama: gc instance sweep failed: {e}")

                try:
                    gc.collect()
                except Exception:
                    pass

                # Final verify every known URL
                for base in _ollama_urls():
                    left = _ps_names(base)
                    if left:
                        print(f"Ollama: FINAL VERIFY FAIL {base} still loaded={left}")
                        for x in left:
                            if x not in residual_all:
                                residual_all.append(x)
                    else:
                        print(f"Ollama: FINAL VERIFY OK {base} loaded=(none)")

                print(
                    f"Ollama: Done — unload_attempts={total_unloaded}, "
                    f"sessions_cleared={sessions_cleared}, "
                    f"context_attrs_cleared={context_cleared}, "
                    f"files_wiped={files_wiped}, "
                    f"instances_cleared={instances_cleared}, "
                    f"residual={residual_all if residual_all else '(none)'}"
                )
            except Exception as e:
                print(f"Ollama: Error purging: {e}")
                import traceback
                print(f"Ollama: Traceback: {traceback.format_exc()}")
```

### 3.4 Memory Manager -- `free_memory(1e30)` (not `0`)

**Source:** `memory_manager.py` at tip `f5c30e9` (same in `nodes/memory_manager.py`), virtual-memory reset blocks.

```python
            if reset_virtual_memory:
                try:
                    import comfy.model_management
                    if hasattr(comfy.model_management, 'free_memory'):
                        # Request a huge free so Comfy actually unloads / empties CUDA.
                        # free_memory(0, ...) is a no-op and left Krea2 NVFP4 ~9GB resident.
                        if torch.cuda.is_available():
                            try:
                                for di in range(torch.cuda.device_count()):
                                    comfy.model_management.free_memory(
                                        1e30, torch.device(f"cuda:{di}")
                                    )
                            except Exception as e:
                                print(f"Virtual memory reset (CUDA) failed: {e}")
                        print("Virtual memory reset")
                except Exception as e:
                    print(f"Virtual memory reset failed: {e}")
            
            if restore_original_functions:
                try:
                    import comfy.model_management
                    print("Original functions restored")
                except Exception as e:
                    print(f"Function restoration failed: {e}")
            

            
            print("Comprehensive memory management completed")
            
        except Exception as e:
            print(f"Memory management error: {e}")
        
        return (anything,)


class SafeMemoryManager:
    """
    Recommended memory management node that prioritizes safe cleanup.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any, {}),
            "clean_gpu": ("BOOLEAN", {"default": True}),
            "force_gc": ("BOOLEAN", {"default": True}),
            "reset_virtual_memory": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "safe_manage_memory"
    CATEGORY = "Memory"

    def safe_manage_memory(self, anything, clean_gpu, force_gc, reset_virtual_memory):
        try:
            if clean_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("Safe GPU memory cleared")
            
            if force_gc:
                gc.collect()
                print("Safe garbage collection completed")
            
            if reset_virtual_memory:
                try:
                    import comfy.model_management
                    if hasattr(comfy.model_management, 'free_memory'):
                        # free_memory(0, ...) is a no-op; request a huge free on all devices.
                        if torch.cuda.is_available():
                            try:
                                for di in range(torch.cuda.device_count()):
                                    comfy.model_management.free_memory(
                                        1e30, torch.device(f"cuda:{di}")
                                    )
                            except Exception as e:
                                print(f"Safe virtual memory reset (CUDA) failed: {e}")
                        print("Safe virtual memory reset")
                except Exception as e:
                    print(f"Safe virtual memory reset failed: {e}")
```

---

## 4. Meaning of the code

### 4.1 Soft path (section 3.1)

| Piece | Meaning |
|---|---|
| `_force_empty_cuda_storage` | Replace storage with `empty(0)` **only if CUDA**; CPU tensors return 0 untouched |
| `_force_kill_nn_cuda` | `module.to("cpu")` then force-empty CUDA params/buffers |
| Loop + pop `current_loaded_models` | Soft unload, then kill leftover CUDA on the NN |
| `unload_all_models()` | Comfy hard unload after per-model pass |
| `free_memory(1e30, cuda:N)` | Real free request; `free_memory(0)` does nothing |
| Log `Force-killed ~X.XX GB CUDA storage...` | Soft residue was actually released |

This path runs with **HSWQ OFF**. It must not invoke HSWQ Methods 0-2c.

### 4.2 HSWQ toggle and entry (section 3.2)

- `HSWQ` / legacy `HSWQ INT8` -> `purge_hswq_int8`.
- All HSWQ nuclear work is gated by that flag only (**no auto-arm** from residual markers).

### 4.3 Helpers before Methods 0-3

| Helper | Meaning |
|---|---|
| `_drain_hswq_pin_cache` | Empties HSWQ Batched Detailer PinCache |
| `_purge_detailer_segs_and_executor_cache` | Clears Impact / SEGS / PromptExecutor caches in place |
| `_reset_comfy_kitchen_cuda_caches` | After Method 3: kitchen + **ZI peel** + NVFP4 pools + Hadamard + parity/forward resets + every `_hswq_*` on modules |
| `_force_unregister_comfy_pins` | Force `cudaHostUnregister` for ComfyUI `PINNED_MEMORY` |
| `_is_hswq_int8_nn` | Whole HSWQ surface detection (INT8 + NVFP4 + ZI + any `_hswq_*`) |
| `_kill_tensor_storage` | Unpin if needed; `empty(0)` **CUDA only** (same rule as soft path) |
| `_kill_module_vram` | CPU move + CUDA storage kill + delete known and residual `_hswq_*` on every submodule |

### 4.4 Method order (HSWQ ON)

1. **Method 0 / 0s** -- Detailer pins and SEGS first.
2. **Method 1** -- Unload HSWQ-holding patchers from `current_loaded_models`.
3. **Method 2** -- Unregister PINNED_MEMORY.
4. **Method 3** -- Nuclear CUDA storage kill (creates poison if pools/hooks remain).
5. **Method 0b / 0s2 / 2b** -- Second pass.
6. **Method 2c** -- Kitchen + **ZI stack peel** + NVFP4 pools + Hadamard + arm attr clear (**mandatory after Method 3**).
7. INT8 LoRA counter reset, `gc` + `empty_cache`.

### 4.5 Method 2c surfaces

| Surface | Cleared how |
|---|---|
| Kitchen | `_cublas_workspaces`, `_empty_cuda_tensors` |
| ZI stack peel | `_clear_zimage_parity_contamination_for_sdxl`, `restore_nvfp4_tc_product_stack`, `uninstall_zimage_nvfp4_lora_bake` |
| NVFP4 runtime | `clear_nvfp4_runtime_pools` + inplace pool dicts / scale placeholders |
| Hadamard | SDXL / ZI / INT8 caches + HSWQ-owned `_HADAMARD_CACHE` / `_H4_CACHE` |
| Parity / forward | `clear_nvfp4_parity_hadamard_caches`, forward/lora counter resets |
| Module attrs | Known list + **every** remaining `_hswq_*` |
| Detailer | PinCache + SEGS + PromptExecutor (twice around Method 3) |

### 4.6 Ollama (section 3.3)

| Piece | Meaning |
|---|---|
| URL harvest | env, module defaults, `Client.host`, describer `api_host` |
| `/api/ps` loop | unload via generate/chat `keep_alive=0`, Client, `ollama stop` |
| Comfy-side wipe | `CHAT_SESSIONS`, class/instance `saved_context`, on-disk `saved_context` (keep `.keep`) |
| FINAL VERIFY | re-check `/api/ps` on every known URL |

Ollama talks to an **external** server; it does not own in-process torch weights. Soft/HSWQ CUDA-only wipe still matters if those paths run in the same purge call.

### 4.7 Memory Manager (section 3.4)

| Before | After |
|---|---|
| `free_memory(0, 'cuda:0')` | no-op; Krea2 residue stayed |
| `free_memory(1e30, torch.device(f"cuda:{di}"))` for all devices | Comfy actually unloads / empties |

Same change in `MemoryManager` and `SafeMemoryManager`.

### 4.8 Design rules

1. **Distorch purge owns residuals** -- do not rely on Loader alone for post-Method-3 cleanup.
2. **Whole HSWQ surface** -- INT8 + NVFP4 + ZI hooks + Detailer + kitchen + Hadamard.
3. **Always clear after nuclear kill** -- Method 2c is mandatory when HSWQ purge ran Method 3.
4. **HSWQ ON only for nuclear path** -- soft residue is a separate always-on soft fix.
5. **CUDA-only `empty(0)`** -- never destroy CPU reload sources (CLIP / ZI TE).
6. **`free_memory(1e30)`** -- never `free_memory(0)` for "clear GPU".
7. Keep `nodes/*` and root twins synchronized.
8. Product spelling is **Distorch**. Workflow class id stays `DisTorchPurgeVRAMV2`.

### 4.9 How to verify in logs

Soft / Memory Manager (HSWQ OFF):

```text
Unloaded N model(s)
Force-killed ~X.XX GB CUDA storage from loaded models (MultiGPU/NVFP4 soft-unload residue)
unload_all_models() issued
free_memory(1e30) issued for all CUDA devices
```

HSWQ ON:

```text
HSWQ INT8/NVFP4: Starting purge process...
HSWQ INT8/NVFP4: Method 0 - Draining HSWQ Batched Detailer PinCache...
HSWQ INT8/NVFP4: Method 3 - gc nuclear ...
HSWQ INT8/NVFP4: Method 2c - Reset comfy_kitchen CUDA caches...
HSWQ INT8/NVFP4: HSWQ stack peel restore_nvfp4_tc_product_stack@... -> ...
HSWQ INT8/NVFP4: HSWQ stack peel uninstall_zimage_nvfp4_lora_bake@... -> ...
HSWQ INT8/NVFP4: Cleared HSWQ NVFP4 runtime pools / CUDA graphs ...
HSWQ INT8/NVFP4: Cleared Hadamard ...
HSWQ INT8/NVFP4: Done -- cleared ... HSWQ INT8/NVFP4 ref(s), ...
```

Ollama ON:

```text
Ollama: Starting FULL purge ...
Ollama: ... EMPTY after round ... (verified /api/ps)
Ollama: FINAL VERIFY OK ... loaded=(none)
```

Cross-check recipes:

1. Update **ComfyUI-DistorchMemoryManager** to tip `f5c30e9` (or later) and fully restart ComfyUI.
2. Krea2 NVFP4 -> soft purge / Memory Manager with **HSWQ OFF** -> VRAM should drop (no leftover ~9 GB); no `HSWQ INT8/NVFP4:` lines.
3. SDXL NVFP4 -> purge HSWQ ON -> Z Image NVFP4 -> purge HSWQ ON -> SDXL NVFP4 again -> third SDXL gen is not noise.
4. Ollama purge then CLIP / TE encode still works (no Embedding.weight 2-D error).

---

**End of Distorch HSWQ purge countermeasure guide (baseline `a5f25d5` -> tip `f5c30e9`).**