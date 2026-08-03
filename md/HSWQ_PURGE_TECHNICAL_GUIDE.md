# Technical Documentation: DistTorch Purge VRAM V2 — HSWQ Toggle


This document is the **complete English technical guide** for the **`HSWQ`** toggle on **DisTorch Purge VRAM V2** in [ComfyUI-DistorchMemoryManager](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager).

It covers:

1. **What** the HSWQ purge is for, and **why** it was required
2. **Which files** were added or modified
3. **Full source text** of the added/modified HSWQ purge code (as of the commits listed below)
4. **Meaning** of that code — pipeline, helpers, failure modes, and invariants

Canonical implementation lives in the **repository root** `purge_vram.py` (imported by `__init__.py`). The optional fallback `nodes/purge_vram.py` is **not** the live path when root import succeeds.

**Related commits on `main` (audit):**

| Hash | Change |
|------|--------|
| `1618a65` | Force-import PinCache; SEGS / PromptExecutor purge |
| `4865ecf` | Do **not** call `PromptExecutor.reset()` mid-prompt — in-place `.cache` / `.subcaches` clear only |
| `339f374` | UI toggle renamed to **`HSWQ`** (legacy kwargs `HSWQ INT8` still accepted) |
| `b7c7068` | Method **2c**: clear `comfy_kitchen` CUDA workspace / empty-tensor caches after nuclear kill |

Log lines still use the prefix `HSWQ INT8:` for grep continuity; the node checkbox label is **`HSWQ`**.

---

## 1. What this is for / why it was needed

### 1.1 Problem

HSWQ INT8 workflows (especially with **HSWQ Batched Detailer**) leave GPU / host memory that normal ComfyUI “unload” and DistTorch’s generic purge paths do **not** reclaim:

| Residue | Typical source | Symptom if left behind |
|---------|----------------|------------------------|
| INT8 UNet / DiT weights (`int8_tensorwise` / `comfy_quant`) | Loader + bake | Dedicated VRAM stays high after “purge” |
| **PinCache** hostbuf pool | `ComfyUI-nunchaku-unofficial-loader` `nodes/hswq_pin_cache.py` | Task Manager dedicated VRAM / pinned host memory after Detailer |
| Large **SEGS** / IMAGE tensors | Impact Detailer + `PromptExecutor` caches | Same — caches keep strong refs |
| ComfyUI **`PINNED_MEMORY`** / `cudaHostRegister` | `comfy.model_management.pin_memory` | Pins survive soft-unload |
| Orphan large CUDA tensors | `gc.get_objects()` survivors | VRAM not returned to the driver |
| **comfy_kitchen** cuBLAS INT8 workspaces | Kitchen CUDA backend module dicts | After nuclear tensor kill: next gen fails with `cublas_gemm_int8` / `Invoked with types: PyCapsule...` because dead workspace refs remain |

Without a dedicated path, operators saw: purge → reload → bake OK → **second** generation crash on kitchen INT8 GEMM; or purge that looked “done” while Windows Task Manager still showed large dedicated VRAM.

### 1.2 Goal of the `HSWQ` toggle

When **`HSWQ`** is enabled on Purge VRAM V2, DistTorch runs a **nuclear HSWQ-oriented pipeline** that:

1. Drains the loader **PinCache** (force-import sibling custom node if needed)
2. Clears Detailer **SEGS** / large IMAGE entries in **PromptExecutor** caches **in place** (never mid-prompt `reset()`)
3. Unloads `current_loaded_models`, killing INT8 modules’ CUDA storage
4. Force-unregisters every ComfyUI-tracked pin
5. Scans `gc` for INT8 modules, patchers, pinned/CUDA tensors ≥ 1 MiB
6. Repeats PinCache / SEGS / pin sweeps
7. **Resets comfy_kitchen CUDA caches** (Method 2c)
8. Resets INT8 LoRA log counters, then `gc` + `empty_cache` / `soft_empty_cache`

The toggle does **not** replace a full ComfyUI restart for every class of leak; it targets the HSWQ INT8 + Detailer residue class that was measured in production.

### 1.3 Hard invariant (do not regress)

**Never** call `PromptExecutor.reset()` while a prompt is running.

Mid-prompt `reset()` destroyed `RAMPressureCache` internal state and caused:

`AttributeError: 'RAMPressureCache' object has no attribute 'cache_key_set'`

Correct approach: walk executor objects and **clear `.cache` / `.subcaches` dictionaries in place**. That is what `_purge_detailer_segs_and_executor_cache` does (commit `4865ecf`).

### 1.4 Hard invariant (Method 2c)

Nuclear CUDA tensor kill can destroy kitchen cuBLAS workspace **storages** while leaving **dead entries** in:

- `comfy_kitchen.backends.cuda._cublas_workspaces`
- `comfy_kitchen.backends.cuda._empty_cuda_tensors`

Reload / re-bake does not always recreate those entries → next `cublas_gemm_int8` sees invalid capsules. Method **2c** clears those dicts after the nuclear pass (commit `b7c7068`).

---

## 2. Files added / modified

### 2.1 DistTorch (this repository)

| Path | Role |
|------|------|
| `purge_vram.py` (repo **root**) | **Canonical** `DisTorchPurgeVRAMV2` — HSWQ toggle + full pipeline |
| `__init__.py` | Imports `from .purge_vram import DisTorchPurgeVRAMV2` first; falls back to `nodes.purge_vram` only if root import fails |
| `nodes/purge_vram.py` | Legacy / fallback copy — **do not** treat as source of truth when root file is present |
| `md/HSWQ_PURGE_TECHNICAL_GUIDE.md` | This document |

### 2.2 Loader (sibling custom node — purge calls into it)

| Path | Role |
|------|------|
| `ComfyUI-nunchaku-unofficial-loader/nodes/hswq_pin_cache.py` | Detailer-scoped PinCache; exposes drain/purge hooks DistTorch force-imports |
| Soft-unload / `PINNED_MEMORY` pop on loader side | Complements DistTorch Method 0 / 2 so pins are not only DistTorch-visible |

DistTorch discovers the loader by scanning `custom_nodes` siblings and **force-importing** `hswq_pin_cache` if the module is not already in `sys.modules` (so purge still works when Detailer has not run yet in this process).

### 2.3 What was *not* changed for this feature

- Generic DistTorch purge toggles (non-HSWQ) remain separate branches in the same `purge()` method.
- ComfyUI core is not patched for HSWQ purge; all logic is custom-node-side.

---

## 3. Full text of added / modified code

The following extracts are taken **verbatim** from the current root `purge_vram.py` (line numbers as of the guide build). They are the complete HSWQ-related surface: UI wiring + the entire `if purge_hswq_int8:` block including nested helpers and Methods 0–2c.


### Extract: INPUT_TYPES toggle and kwargs wiring (`purge_vram.py` lines 29-50)

```python
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
                "purge_seedvr2_models": ("BOOLEAN", {"default": False, "tooltip": "Clear SeedVR2 DiT (base) and VAE models from cache"}),
                "purge_qwen3vl_models": ("BOOLEAN", {"default": False, "tooltip": "Clear Qwen3-VL models from GPU memory"}),
                "purge_nunchaku_models": ("BOOLEAN", {"default": False, "tooltip": "Clear Nunchaku models (FLUX/Z-Image/Qwen-Image) from GPU memory"}),
                "HSWQ": ("BOOLEAN", {"default": False, "tooltip": "Clear HSWQ INT8 models from GPU memory"}),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "purge_vram"
    CATEGORY = "DisTorch/Memory"

    def purge_vram(self, anything, purge_cache, purge_models, purge_seedvr2_models, purge_qwen3vl_models, purge_nunchaku_models, **kwargs):
        # Toggle label is "HSWQ"; accept legacy "HSWQ INT8" for old workflows.
        purge_hswq_int8 = bool(kwargs.get("HSWQ", kwargs.get("HSWQ INT8", False)))
```

### Extract: Full HSWQ purge block (if purge_hswq_int8) (`purge_vram.py` lines 1964-2690)

```python
        # Purge HSWQ INT8 (comfy_quant int8_tensorwise) + Batched Detailer pin pool
        # + orphaned ComfyUI cudaHostRegister / CUDA tensors (Task Manager dedicated GPU mem)
        if purge_hswq_int8:
            try:
                print("HSWQ INT8: Starting purge process...")
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
                            f"HSWQ INT8: [{tag}] TOTAL_PINNED_MEMORY="
                            f"{total_pin / (1024 * 1024):.1f} MB entries={pin_entries}"
                        )
                    except Exception as e:
                        print(f"HSWQ INT8: [{tag}] pin diag failed: {e}")
                    if torch.cuda.is_available():
                        try:
                            for di in range(torch.cuda.device_count()):
                                alloc = torch.cuda.memory_allocated(di)
                                reserved = torch.cuda.memory_reserved(di)
                                free_b, total_b = torch.cuda.mem_get_info(di)
                                used_sys = max(0, total_b - free_b)
                                print(
                                    f"HSWQ INT8: [{tag}] cuda:{di} "
                                    f"allocated={alloc / (1024 ** 3):.2f}GB "
                                    f"reserved={reserved / (1024 ** 3):.2f}GB "
                                    f"sys_used={used_sys / (1024 ** 3):.2f}GB"
                                )
                        except Exception as e:
                            print(f"HSWQ INT8: [{tag}] cuda diag failed: {e}")

                def _drain_hswq_pin_cache() -> int:
                    drained = 0

                    def _call_purge(mod, mod_name: str) -> int:
                        fn = getattr(mod, "purge_pin_cache", None)
                        if callable(fn):
                            got = int(fn() or 0)
                            print(
                                f"HSWQ INT8: PinCache purged via {mod_name}: "
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
                                f"HSWQ INT8: PinCache _drain_pool via {mod_name}: "
                                f"{total / (1024 * 1024):.1f} MB"
                            )
                            return total
                        if pool is not None:
                            pool.clear()
                            setattr(mod, "_PIN_CACHE_TOTAL", 0)
                            print(f"HSWQ INT8: PinCache pool cleared via {mod_name}")
                            return total
                        return 0

                    for mod_name, mod in _sys_modules():
                        if mod is None or "hswq_pin_cache" not in str(mod_name):
                            continue
                        try:
                            return _call_purge(mod, str(mod_name))
                        except Exception as e:
                            print(f"HSWQ INT8: PinCache purge via {mod_name} failed: {e}")

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
                            print(f"HSWQ INT8: Force-import PinCache from {pin_py}")
                            spec = importlib.util.spec_from_file_location(
                                "hswq_pin_cache_force_purge", pin_py
                            )
                            if spec is None or spec.loader is None:
                                continue
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            sys.modules["hswq_pin_cache_force_purge"] = mod
                            return _call_purge(mod, pin_py)
                    except Exception as e:
                        print(f"HSWQ INT8: PinCache force-import failed: {e}")

                    print("HSWQ INT8: PinCache module not loaded (nothing to drain)")
                    return 0

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
                                            f"HSWQ INT8: in-place cache clear "
                                            f"failed: {e}"
                                        )
                            except Exception as e:
                                print(
                                    f"HSWQ INT8: PromptExecutor cache clear "
                                    f"failed: {e}"
                                )
                    except Exception as e:
                        print(f"HSWQ INT8: PromptExecutor scan failed: {e}")
                    print(
                        f"HSWQ INT8: PromptExecutor in-place cache clear "
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
                            f"HSWQ INT8: Impact preview/SEG bridge caches cleared "
                            f"entries={impact_cleared}"
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
                        print(f"HSWQ INT8: SEGS tensor sweep failed: {e}")
                    print(
                        f"HSWQ INT8: Detailer SEGS/cache sweep "
                        f"tensors_touched={tensor_killed} "
                        f"approx={freed_hint / (1024 * 1024):.1f} MB"
                    )
                    return freed_hint

                def _reset_comfy_kitchen_cuda_caches() -> None:
                    """Drop comfy_kitchen global CUDA buffers after nuclear tensor kill.

                    Method 0s / Method 3 walk gc.get_objects() and replace storage on
                    large CUDA tensors. That includes comfy_kitchen's cuBLAS workspace
                    (4 MiB or 32 MiB uint8) which stays cached in
                    ``_cublas_workspaces``. Model reload does not recreate it —
                    get_cublas_workspace() returns the dead tensor →
                    cublas_gemm_int8 sees PyCapsule instead of ndarray (2nd gen).
                    """
                    try:
                        import comfy_kitchen.backends.cuda as ck_cuda
                    except Exception as e:
                        print(f"HSWQ INT8: comfy_kitchen cuda import skipped: {e}")
                        return
                    cleared = []
                    for attr in (
                        "_cublas_workspaces",
                        "_empty_cuda_tensors",
                    ):
                        bag = getattr(ck_cuda, attr, None)
                        if isinstance(bag, dict) and bag:
                            n = len(bag)
                            bag.clear()
                            cleared.append(f"{attr}={n}")
                    if cleared:
                        print(
                            "HSWQ INT8: Reset comfy_kitchen CUDA caches "
                            + ", ".join(cleared)
                        )
                    else:
                        print("HSWQ INT8: comfy_kitchen CUDA caches already empty")

                def _force_unregister_comfy_pins() -> int:
                    """Unregister every cudaHostRegister tracked by ComfyUI PINNED_MEMORY."""
                    nonlocal pins_unregistered
                    freed = 0
                    try:
                        import comfy.model_management as mm
                    except Exception as e:
                        print(f"HSWQ INT8: cannot import model_management for pin nuke: {e}")
                        return 0
                    pinned = getattr(mm, "PINNED_MEMORY", None)
                    if not isinstance(pinned, dict):
                        print("HSWQ INT8: PINNED_MEMORY dict missing")
                        return 0
                    before = int(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0)
                    print(
                        f"HSWQ INT8: Force-unregister PINNED_MEMORY "
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
                        f"HSWQ INT8: Force-unregister done "
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
                    """Strict: only real HSWQ INT8 UNet modules — no torch/_dynamo junk."""
                    if module is None or not _is_real_nn(module):
                        return False
                    baked = getattr(module, "_hswq_int8_baked_keys", None)
                    if baked:
                        return True
                    if getattr(module, "_hswq_int8_baked_uuid", None) is not None:
                        return True
                    try:
                        for name, buf in module.named_buffers():
                            if not (name.endswith("comfy_quant") or name.endswith(".comfy_quant")):
                                continue
                            try:
                                raw = buf.detach().cpu()
                                if raw.dtype == torch.uint8 and raw.numel() > 0:
                                    import json
                                    conf = json.loads(bytes(raw.tolist()).decode("utf-8", errors="ignore"))
                                    if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                                        return True
                                    if isinstance(conf, dict) and "format" in conf:
                                        return conf.get("format") == "int8_tensorwise"
                            except Exception:
                                pass
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
                        dtype = getattr(data, "dtype", torch.float32)
                        empty = torch.empty(0, dtype=dtype, device="cpu")
                        if hasattr(t, "data"):
                            t.data = empty
                        if is_cuda:
                            freed = nbytes
                    except Exception:
                        pass
                    return freed

                def _kill_module_vram(module, label: str) -> int:
                    freed = 0
                    print(f"HSWQ INT8: Killing module VRAM ({label}) type={type(module).__name__}")
                    try:
                        if hasattr(module, "to") and callable(module.to):
                            try:
                                module.to("cpu")
                            except Exception as e:
                                print(f"HSWQ INT8: .to('cpu') warning ({label}): {e}")
                    except Exception:
                        pass
                    try:
                        for _n, p in list(module.named_parameters()):
                            freed += _kill_tensor_storage(p)
                    except Exception as e:
                        print(f"HSWQ INT8: param kill warning ({label}): {e}")
                    try:
                        for _n, b in list(module.named_buffers()):
                            freed += _kill_tensor_storage(b)
                    except Exception as e:
                        print(f"HSWQ INT8: buffer kill warning ({label}): {e}")
                    try:
                        if hasattr(module, "_hswq_int8_baked_keys"):
                            module._hswq_int8_baked_keys = None
                        if hasattr(module, "_hswq_int8_baked_uuid"):
                            module._hswq_int8_baked_uuid = None
                    except Exception:
                        pass
                    print(f"HSWQ INT8: Killed ~{freed / (1024 * 1024):.1f} MB CUDA storage ({label})")
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
                print("HSWQ INT8: Method 0 - Draining HSWQ Batched Detailer PinCache...")
                bytes_killed += _drain_hswq_pin_cache()
                print("HSWQ INT8: Method 0s - Detailer SEGS / PromptExecutor cache...")
                bytes_killed += _purge_detailer_segs_and_executor_cache()

                # 1) ComfyUI loaded models (INT8 first, then unload everything)
                print("HSWQ INT8: Method 1 - current_loaded_models...")
                models_checked_mm = 0
                models_found_mm = 0
                try:
                    import comfy.model_management as mm
                    if hasattr(mm, "current_loaded_models"):
                        current_loaded_models = mm.current_loaded_models
                        print(f"HSWQ INT8: current_loaded_models count={len(current_loaded_models)}")
                        for i in range(len(current_loaded_models) - 1, -1, -1):
                            loaded_model = current_loaded_models[i]
                            models_checked_mm += 1
                            try:
                                is_int8 = _loaded_holds_hswq_int8(loaded_model)
                                if is_int8:
                                    models_found_mm += 1
                                    print(
                                        f"HSWQ INT8: Found INT8 at current_loaded_models[{i}] "
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
                                    print(f"HSWQ INT8: model_unload warning: {e}")
                                current_loaded_models.pop(i)
                                print(f"HSWQ INT8: Removed current_loaded_models[{i}] (int8={is_int8})")
                            except Exception as e:
                                print(f"HSWQ INT8: Error at current_loaded_models[{i}]: {e}")
                    try:
                        if hasattr(mm, "unload_all_models") and callable(mm.unload_all_models):
                            mm.unload_all_models()
                            print("HSWQ INT8: unload_all_models() issued")
                    except Exception as e:
                        print(f"HSWQ INT8: unload_all_models warning: {e}")
                    try:
                        if torch.cuda.is_available() and hasattr(mm, "free_memory"):
                            for di in range(torch.cuda.device_count()):
                                mm.free_memory(1e30, torch.device(f"cuda:{di}"))
                            print("HSWQ INT8: free_memory(1e30) issued for all CUDA devices")
                    except Exception as e:
                        print(f"HSWQ INT8: free_memory warning: {e}")
                    if hasattr(mm, "cleanup_models_gc") and callable(mm.cleanup_models_gc):
                        try:
                            mm.cleanup_models_gc()
                        except Exception as e:
                            print(f"HSWQ INT8: cleanup_models_gc warning: {e}")
                except Exception as e:
                    print(f"HSWQ INT8: Error in Method 1: {e}")
                    import traceback
                    print(f"HSWQ INT8: Traceback: {traceback.format_exc()}")
                print(
                    f"HSWQ INT8: Method 1 complete - checked {models_checked_mm}, found {models_found_mm}"
                )
                _mem_diag("after_method1")

                # 2) Force HostUnregister every ComfyUI-tracked pin (NOT sys.modules dir/getattr —
                #    that triggers kornia LazyLoader basicsr install prompts)
                print("HSWQ INT8: Method 2 - Force HostUnregister PINNED_MEMORY...")
                bytes_killed += _force_unregister_comfy_pins()
                _mem_diag("after_method2")

                # 3) gc nuclear: INT8 modules + ModelPatchers + pinned/CUDA tensors
                print("HSWQ INT8: Method 3 - gc nuclear (no sys.modules getattr)...")
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
                            print("HSWQ INT8: gc scan limit 500000")
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
                                print(f"HSWQ INT8: Found INT8 in gc type={tname}")
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
                    print(f"HSWQ INT8: Error in Method 3: {e}")
                    import traceback
                    print(f"HSWQ INT8: Traceback: {traceback.format_exc()}")
                print(
                    f"HSWQ INT8: Method 3 complete - checked {objects_checked}, "
                    f"int8={models_found_in_gc}, patchers={patchers_unloaded}, "
                    f"cuda_tensors={cuda_tensors_killed}"
                )

                # Second PinCache drain + second PINNED_MEMORY sweep
                print("HSWQ INT8: Method 0b - Second PinCache drain...")
                bytes_killed += _drain_hswq_pin_cache()
                print("HSWQ INT8: Method 0s2 - Second Detailer SEGS / executor sweep...")
                bytes_killed += _purge_detailer_segs_and_executor_cache()
                print("HSWQ INT8: Method 2b - Second PINNED_MEMORY sweep...")
                bytes_killed += _force_unregister_comfy_pins()

                # Nuclear CUDA tensor kill may have destroyed kitchen workspaces
                # while leaving dead refs in module-level dicts — clear them.
                print("HSWQ INT8: Method 2c - Reset comfy_kitchen CUDA caches...")
                _reset_comfy_kitchen_cuda_caches()

                # Reset INT8 LoRA counters (dict-only, no dir())
                print("HSWQ INT8: Resetting comfy_quant_int8 counters...")
                try:
                    for mod_name, mod in _sys_modules():
                        if mod is None or "comfy_quant_int8" not in str(mod_name):
                            continue
                        d = getattr(mod, "__dict__", None)
                        if not isinstance(d, dict):
                            continue
                        reset_fn = d.get("reset_int8_lora_log_counters")
                        if callable(reset_fn):
                            print(f"HSWQ INT8: Calling reset_int8_lora_log_counters via {mod_name}")
                            reset_fn()
                            break
                except Exception as e:
                    print(f"HSWQ INT8: counter reset skipped: {e}")

                print("HSWQ INT8: Running garbage collection...")
                gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    print("HSWQ INT8: Clearing CUDA cache...")
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
                    print("HSWQ INT8: CUDA cache cleared for all devices")
                else:
                    print("HSWQ INT8: CUDA not available, skipped CUDA cache clear")

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
                    f"HSWQ INT8: Done — cleared {hswq_cleared} INT8 ref(s), "
                    f"pins_unregistered={pins_unregistered}, "
                    f"patchers={patchers_unloaded}, "
                    f"cuda_tensors={cuda_tensors_killed}, "
                    f"approx {bytes_killed / (1024 * 1024):.1f} MB tracked"
                )

            except Exception as e:
                print(f"HSWQ INT8: Error purging models: {e}")
                import traceback
                print(f"HSWQ INT8: Traceback: {traceback.format_exc()}")
```



---

## 4. Meaning of the code

### 4.1 Toggle wiring (`INPUT_TYPES` / kwargs)

- UI key: **`"HSWQ"`** (boolean, default `False`).
- Runtime: `purge_hswq_int8 = bool(kwargs.get("HSWQ", kwargs.get("HSWQ INT8", False)))` so old workflows that still send `"HSWQ INT8"` keep working after the rename (`339f374`).

### 4.2 Pipeline order (when `purge_hswq_int8` is true)

```text
_mem_diag("before")
  Method 0   — drain HSWQ Batched Detailer PinCache (_drain_hswq_pin_cache)
  Method 0s  — SEGS / PromptExecutor in-place cache clear
  Method 1   — current_loaded_models: kill INT8 VRAM, unload all, free_memory, cleanup_models_gc
  Method 2   — Force HostUnregister every PINNED_MEMORY entry
  Method 3   — gc nuclear: ModelPatcher* / INT8 nn / pinned+CUDA tensors ≥1MiB
  Method 0b  — second PinCache drain
  Method 0s2 — second SEGS / executor sweep
  Method 2b  — second PINNED_MEMORY sweep
  Method 2c  — reset comfy_kitchen CUDA workspace / empty-tensor caches
  — reset comfy_quant_int8 LoRA log counters (dict walk, no dir() on LazyLoaders)
  — gc.collect ×2, empty_cache / ipc_collect / synchronize, soft_empty_cache
_mem_diag("after")
```

Counters logged at Done: `hswq_cleared`, `pins_unregistered`, `patchers_unloaded`, `cuda_tensors_killed`, approximate `bytes_killed`.

### 4.3 Helper meanings

| Helper | Meaning |
|--------|---------|
| `_sys_modules()` | Safe iteration over `sys.modules` without triggering LazyLoader install prompts |
| `_mem_diag(tag)` | Prints torch / NVML-style memory snapshot for before/after debugging |
| `_drain_hswq_pin_cache()` | Locates loader PinCache module (force-import sibling path if needed) and drains the hostbuf pool |
| `_purge_detailer_segs_and_executor_cache()` | Finds PromptExecutor / cache objects; clears SEGS/IMAGE-heavy entries **in place** — **no** `.reset()` |
| `_reset_comfy_kitchen_cuda_caches()` | Method 2c — clear kitchen module-level CUDA workspace dicts |
| `_force_unregister_comfy_pins()` | Walk `mm.PINNED_MEMORY`, unpin / `cudaHostUnregister`, pop entries, fix `TOTAL_PINNED_MEMORY` |
| `_is_hswq_int8_nn` / `_loaded_holds_hswq_int8` | Detect INT8 / comfy_quant modules on LoadedModel |
| `_kill_tensor_storage` / `_kill_module_vram` | Move module to CPU, replace param/buffer `.data` with empty CPU tensors, clear bake metadata attrs |
| `_unload_patcher` | `partially_unload_ram` / `unregister_inactive_pins` / `partially_unload` / `unpatch_model` / `model_unload` |

### 4.4 Why Method 1 unloads *everything*, not only INT8

INT8 detection is used to prioritize `_kill_module_vram` and counting, but **every** `current_loaded_models` entry is torn down (patcher unload + `model_unload` + pop). HSWQ residue often sits next to non-INT8 patchers that still hold pins/hostbufs. Leaving them would defeat Task Manager reclaim.

### 4.5 Why Method 3 avoids `sys.modules` getattr storms

Walking modules with `dir()` / reckless `getattr` can trigger **kornia / basicsr LazyLoader** install prompts mid-purge. Method 3 uses `gc.get_objects()` and type-name checks instead. Counter reset similarly uses `__dict__.get` only on modules whose name contains `comfy_quant_int8`.

### 4.6 Failure modes already fixed (do not reintroduce)

| Failure | Wrong fix | Correct fix |
|---------|-----------|-------------|
| Mid-prompt `RAMPressureCache` / `cache_key_set` | Call `PromptExecutor.reset()` | In-place cache dict clear (`4865ecf`) |
| 2nd gen `cublas_gemm_int8` PyCapsule error after purge+reload | Blame purge of SEGS alone / skip nuclear | Method **2c** kitchen dict clear after nuclear (`b7c7068`) |
| PinCache not drained if Detailer never ran | Assume module already imported | Force-import sibling `hswq_pin_cache.py` (`1618a65`) |

### 4.7 Operator notes

1. Enable **`HSWQ`** on Purge VRAM V2 when clearing after HSWQ INT8 + Batched Detailer (or when Task Manager still shows dedicated VRAM after a normal purge).
2. Prefer **reload the model after** HSWQ purge (purge clears live weights/workspaces; reload recreates them).
3. If kitchen errors return after a DistTorch update, confirm Method **2c** log line: `Method 2c - Reset comfy_kitchen CUDA caches...`.
4. Edit **root** `purge_vram.py` only; keep `nodes/purge_vram.py` in sync only if you intentionally maintain the fallback.

---

## 5. Self-check before claiming “HSWQ purge is done”

```
□ Root purge_vram.py is what __init__.py imports
□ Toggle label is HSWQ; legacy kwargs HSWQ INT8 still accepted
□ No PromptExecutor.reset() in the HSWQ block
□ Method 2c present after second pin sweep
□ PinCache force-import path still present
□ English-only prose in this guide (repo English-only rule)
```

---

*(End of HSWQ Purge technical guide. Code extracts above match the live root `purge_vram.py` at guide build time.)*
