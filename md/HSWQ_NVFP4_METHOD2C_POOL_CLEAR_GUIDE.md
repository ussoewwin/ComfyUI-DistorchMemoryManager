# Distorch HSWQ Purge — NVFP4 Runtime Pool Clear (Method 2c)

**Scope:** ComfyUI-DistorchMemoryManager (ComfyUI-VRAM-Manager) — Distorch Purge VRAM V2  
**Symptom target:** Second generation in the same ComfyUI process after HSWQ purge  
**Audience:** Users and integrators of this custom node (public technical note)  
**Date:** 2026-07-21  

This document explains the Distorch-side fix for ConvRot **NVFP4** second-run failure after Distorch VRAM purge. INT8 kitchen/workspace clear already existed; this change **adds** NVFP4 pool/graph clear on the same Method 2c path and makes the running import prefer `nodes/purge_vram.py`.

Repository: [ussoewwin/ComfyUI-DistorchMemoryManager](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager)

---

## 1. Error content

### 1.1 Observed sequence

1. First prompt with SDXL HSWQ **NVFP4** (ConvRot / kitchen `quantize_nvfp4` path): **succeeds**.
2. Distorch **Purge VRAM V2** runs with the **HSWQ** toggle enabled (log prefix historically `HSWQ INT8:`).
3. Second prompt in the **same** ComfyUI process: forward floods with kitchen / TC pooled path failures.

### 1.2 Representative exception (second run)

```text
pooled TC path failed: quantize_nvfp4() ... Invoked with types: PyCapsule ...
```

Typical shape:

- Exception class: `TypeError` (or wrapper logging `pooled TC path failed`)
- Kernel / binding: `comfy_kitchen` CUDA extension `quantize_nvfp4`
- Unexpected argument type: **`PyCapsule`** (DLPack / dead tensor storage handle) where a live CUDA tensor buffer was expected
- Timing: **after** Distorch HSWQ purge methods (especially nuclear CUDA tensor kill + Method 2c kitchen reset that previously only cleared kitchen dicts)

### 1.3 What the first-run log looked like (healthy)

On a healthy first run you typically see:

- NVFP4 stack applied (`full stack applied` / LoRA bake OK)
- No `pooled TC path failed`
- No `PyCapsule` in `quantize_nvfp4` argument types

### 1.4 What Distorch logged before this fix (relevant excerpt)

Method 2c ran, but only kitchen-side bags were reported, for example:

```text
HSWQ INT8: Method 2c - Reset comfy_kitchen CUDA caches...
HSWQ INT8: Reset comfy_kitchen CUDA caches _cublas_workspaces=..., _empty_cuda_tensors=...
```

**Missing** (before the fix): any line such as:

```text
Cleared HSWQ NVFP4 runtime pools / CUDA graphs
... nvfp4_runtime_pools ...
```

That absence is the Distorch-side smoking gun: kitchen caches were reset; **NVFP4 module-level pools were not**.

### 1.5 Related INT8 historical error (same Method 2c family)

INT8 second-gen after purge previously failed with a similar pattern on kitchen cuBLAS workspace:

```text
cublas_gemm_int8 ... PyCapsule ...
```

Method 2c already cleared `_cublas_workspaces` / `_empty_cuda_tensors` for that INT8 case. The NVFP4 failure is the **same structural class** on a **different** pool owner (`nvfp4_runtime`).

---

## 2. Root cause

### 2.1 Distorch HSWQ purge destroys CUDA storage aggressively

When the HSWQ purge toggle is on, Distorch walks several methods, including:

| Method | Role |
|--------|------|
| Method 0 / 0s | PinCache / Detailer SEGS / PromptExecutor cache |
| Method 1 | `unload_all_models` / `free_memory` |
| Method 2 / 2b | Force-unregister ComfyUI `PINNED_MEMORY` |
| Method 3 | Nuclear `gc` scan — replace / kill large CUDA tensor storage |
| Method 2c | Reset **module-level** CUDA caches that still hold dead refs |

Method 0s / Method 3 do not only free “models”. They walk live Python objects (`gc.get_objects()` / SEGS sweeps) and **replace or empty CUDA storage** on large tensors still referenced from caches.

### 2.2 HSWQ NVFP4 keeps module-level pools outside those dicts

In ComfyUI-nunchaku-unofficial-loader, NVFP4 activation / rotation / CUDA-graph reuse lives in:

`nodes/nvfp4/nvfp4_runtime.py`

Conceptually:

| Symbol | Role |
|--------|------|
| `_ACT_Q_POOL` | Reused `qx` / scale buffers for `quantize_nvfp4_act_pooled` |
| `_ROT_OUT_POOL` | Reused rotate matmul outputs |
| `_GRAPH_CACHE` | CUDA graphs for NVFP4 GEMM paths |
| `clear_nvfp4_runtime_pools()` | Clears all of the above |

Those pools are **module globals**. Distorch Method 3 may:

1. Find the **tensors inside** the pools via `gc`, and
2. Destroy / replace their CUDA storage,

while the **dict entries remain**, still pointing at Python tensor objects whose backing storage is now invalid.

### 2.3 Why reload alone does not heal it

After purge:

- UNet / MODEL may be unloaded and later reloaded.
- Reloading **weights** does **not** recreate `_ACT_Q_POOL` / `_ROT_OUT_POOL` / `_GRAPH_CACHE`.
- The next forward still calls `quantize_nvfp4_act_pooled` → kitchen `_C.quantize_nvfp4(...)`.
- Kitchen wraps buffers for DLPack; dead storage surfaces as **`PyCapsule`** in the binding → `TypeError`.

### 2.4 Why INT8 clear was already present but NVFP4 still broke

Method 2c already cleared **comfy_kitchen** module dicts:

- `_cublas_workspaces`
- `_empty_cuda_tensors`

That is sufficient for the INT8 `cublas_gemm_int8` PyCapsule class.

NVFP4 pools live in **another package** (`nvfp4_runtime`), not in `comfy_kitchen.backends.cuda`. Clearing kitchen alone left NVFP4 pools poisoned.

### 2.5 Secondary import trap (why an install can look “unfixed”)

This repository ships two purge implementations:

| Path | Role |
|------|------|
| `nodes/purge_vram.py` | Canonical implementation (Method 2c + NVFP4 clear) |
| root `purge_vram.py` | Legacy copy; may lag behind `nodes/` |

If `__init__.py` imported **root first**, ComfyUI kept executing the old Method 2c (kitchen only). Preferring `.nodes.purge_vram` is part of making the Distorch-side NVFP4 clear actually run after users update the custom node.

---

## 3. Countermeasure overview

### 3.1 Distorch-side strategy (this guide)

1. **Keep** existing INT8 / kitchen Method 2c behavior.
2. **Add** a call to `clear_nvfp4_runtime_pools()` from any loaded `nvfp4_runtime` module found in `sys.modules` (the import name varies across ComfyUI custom-node layouts).
3. **Prefer** importing `DisTorchPurgeVRAMV2` from `.nodes.purge_vram` so an outdated root `purge_vram.py` cannot silently win.
4. **Retarget log prefix** from `HSWQ INT8:` to `HSWQ INT8/NVFP4:` so purge logs match the dual INT8+NVFP4 Method 2c scope.

### 3.2 What success looks like in Distorch logs

After ComfyUI restart with the fixed `nodes/purge_vram.py` loaded:

```text
HSWQ INT8/NVFP4: Method 2c - Reset comfy_kitchen CUDA caches...
HSWQ INT8/NVFP4: Cleared HSWQ NVFP4 runtime pools / CUDA graphs
HSWQ INT8/NVFP4: Reset comfy_kitchen CUDA caches ..., nvfp4_runtime_pools
```

(Exact kitchen bag names may vary; the important new token is `nvfp4_runtime_pools` and/or the “Cleared HSWQ NVFP4 runtime pools” line.)

Then: first prompt → purge → second prompt must show **zero** `pooled TC path failed` / `PyCapsule` on `quantize_nvfp4`.

### 3.3 What this Distorch fix does *not* claim

- It does not replace loader-side early-return / upgrade pool clears (those are a separate safety net in unofficial-loader).
- It does not delete Nunchaku SVDQ paths.
- It does not change INT8 PinCache / SEGS / Method 3 algorithms beyond log labeling and the Method 2c NVFP4 addition.

### 3.4 Where the fix lives in this repository

| File | Role for this fix |
|------|-------------------|
| `nodes/purge_vram.py` | Canonical Method 2c body (kitchen + NVFP4 pool clear) |
| `__init__.py` | Imports `DisTorchPurgeVRAMV2` from `.nodes.purge_vram` first |
| `purge_vram.py` (repository root) | Legacy fallback only; may not include NVFP4 clear |

Users obtain the fix by updating **ComfyUI-DistorchMemoryManager** from this repository (Manager / git pull / release package), then fully restarting ComfyUI so the new `nodes/purge_vram.py` is loaded.

---

## 4. Modified file names

### 4.1 Distorch (required for this fix)

| File | Change |
|------|--------|
| `nodes/purge_vram.py` | Extended `_reset_comfy_kitchen_cuda_caches()` to clear HSWQ NVFP4 runtime pools; Method 2c call site unchanged in role; log prefix `HSWQ INT8/NVFP4:` |
| `__init__.py` | Prefer `from .nodes.purge_vram import DisTorchPurgeVRAMV2`; root `purge_vram` is fallback only |

### 4.2 Paths inside this repository

- `nodes/purge_vram.py`
- `__init__.py`

On GitHub:

- https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/nodes/purge_vram.py
- https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/__init__.py

### 4.3 External API consumed (not a Distorch file; called by Distorch)

Loader function Distorch invokes when the module is loaded:

- Package: `ComfyUI-nunchaku-unofficial-loader`
- Module: `nodes.nvfp4.nvfp4_runtime`
- Function: `clear_nvfp4_runtime_pools()`

Minimal definition (loader side, for reference):

```python
def clear_nvfp4_runtime_pools() -> None:
    _ACT_Q_POOL.clear()
    _ROT_OUT_POOL.clear()
    clear_nvfp4_cudagraphs()
```

---

## 5. Full text of the added / modified Distorch code

### 5.1 `__init__.py` — Purge VRAM V2 import preference

```python
# Import Purge VRAM V2 node.
# Prefer nodes.purge_vram (Method 2c clears HSWQ NVFP4 runtime pools after Distorch
# nuclear kill). Root purge_vram.py is legacy fallback only.
try:
    from .nodes.purge_vram import DisTorchPurgeVRAMV2
    print("[ComfyUI-VRAM-Manager] Successfully imported DisTorchPurgeVRAMV2 from .nodes.purge_vram")
except ImportError as e:
    try:
        from .purge_vram import DisTorchPurgeVRAMV2
        print("[ComfyUI-VRAM-Manager] Successfully imported DisTorchPurgeVRAMV2 from .purge_vram")
    except ImportError as e2:
        print(f"[ComfyUI-VRAM-Manager] WARNING: Failed to import DisTorchPurgeVRAMV2: {e2}")
        DisTorchPurgeVRAMV2 = None
```

### 5.2 `nodes/purge_vram.py` — `_reset_comfy_kitchen_cuda_caches` (Method 2c body)

This is the complete nested function as shipped for INT8 kitchen + NVFP4 pool clear (log prefix included):

```python
                def _reset_comfy_kitchen_cuda_caches() -> None:
                    """Drop comfy_kitchen + HSWQ NVFP4 pooled CUDA buffers after nuclear kill.

                    Method 0s / Method 3 walk gc.get_objects() and replace storage on
                    large CUDA tensors. That includes:

                    - comfy_kitchen cuBLAS workspace (``_cublas_workspaces``) →
                      dead tensor → ``cublas_gemm_int8`` PyCapsule (INT8 2nd gen)
                    - HSWQ NVFP4 ``_ACT_Q_POOL`` / ``_ROT_OUT_POOL`` / CUDA-graph
                      cache → dead qx/sx buffers → ``quantize_nvfp4`` PyCapsule
                      (ConvRot NVFP4 2nd gen after Distorch purge)

                    Model reload alone does not recreate those module-level pools.
                    """
                    try:
                        import comfy_kitchen.backends.cuda as ck_cuda
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: comfy_kitchen cuda import skipped: {e}")
                        ck_cuda = None
                    cleared = []
                    if ck_cuda is not None:
                        for attr in (
                            "_cublas_workspaces",
                            "_empty_cuda_tensors",
                        ):
                            bag = getattr(ck_cuda, attr, None)
                            if isinstance(bag, dict) and bag:
                                n = len(bag)
                                bag.clear()
                                cleared.append(f"{attr}={n}")
                    # HSWQ NVFP4 act / GEMM pools — scan sys.modules (import path varies).
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        if not (
                            name.endswith("nvfp4_runtime")
                            or ".nvfp4_runtime" in name
                        ):
                            continue
                        fn = getattr(mod, "clear_nvfp4_runtime_pools", None)
                        if not callable(fn):
                            continue
                        try:
                            fn()
                            cleared.append("nvfp4_runtime_pools")
                            print(
                                "HSWQ INT8/NVFP4: Cleared HSWQ NVFP4 runtime pools / CUDA graphs"
                            )
                            break
                        except Exception as e2:
                            print(
                                f"HSWQ INT8/NVFP4: NVFP4 runtime pool clear failed ({name}): {e2}"
                            )
                    if cleared:
                        print(
                            "HSWQ INT8/NVFP4: Reset comfy_kitchen CUDA caches "
                            + ", ".join(cleared)
                        )
                    else:
                        print("HSWQ INT8/NVFP4: comfy_kitchen CUDA caches already empty")
```

### 5.3 `nodes/purge_vram.py` — Method 2c call site (HSWQ purge sequence)

```python
                # Nuclear CUDA tensor kill may have destroyed kitchen workspaces
                # while leaving dead refs in module-level dicts — clear them.
                print("HSWQ INT8/NVFP4: Method 2c - Reset comfy_kitchen CUDA caches...")
                _reset_comfy_kitchen_cuda_caches()
```

### 5.4 Log prefix change (same file)

All HSWQ purge `print` lines that previously used the prefix `HSWQ INT8:` now use:

```text
HSWQ INT8/NVFP4:
```

Rationale: Method 2c (and the HSWQ purge toggle) cover **both** INT8 kitchen workspace clear and NVFP4 runtime pool clear. A log that says only `INT8` is misleading when NVFP4 pools are also cleared on the same path.

---

## 6. Meaning of the code

### 6.1 Why Method 2c exists at all

Methods 0s / 3 are **storage killers**. They are correct for freeing VRAM held by SEGS / executor caches / stray CUDA tensors. They are **incorrect** if module-level dicts keep names pointing at those tensors after storage death.

Method 2c is the **dictionary hygiene** step:

- Clear kitchen workspaces that INT8 GEMM reuses.
- Clear NVFP4 pools/graphs that NVFP4 act quant / GEMM reuses.

Without Method 2c, “purge succeeded” (allocated VRAM drops) can still leave **poisoned caches** for the next forward.

### 6.2 Kitchen clear block (`_cublas_workspaces`, `_empty_cuda_tensors`)

```python
for attr in ("_cublas_workspaces", "_empty_cuda_tensors"):
    bag = getattr(ck_cuda, attr, None)
    if isinstance(bag, dict) and bag:
        bag.clear()
```

**Meaning:**

- These dicts cache CUDA tensors for comfy_kitchen performance.
- After nuclear kill, entries can be dead.
- `.clear()` drops the references so the next INT8 kitchen call allocates fresh buffers.
- This is the historical INT8 second-gen fix; **kept unchanged in role**.

### 6.3 NVFP4 `sys.modules` scan

```python
for name, mod in list(__import__("sys").modules.items()):
    if name.endswith("nvfp4_runtime") or ".nvfp4_runtime" in name:
        fn = getattr(mod, "clear_nvfp4_runtime_pools", None)
        if callable(fn):
            fn()
            ...
            break
```

**Meaning:**

- Distorch must not hard-depend on one frozen import string (`nodes.nvfp4.nvfp4_runtime` vs other package-qualified names ComfyUI may register).
- Scanning `sys.modules` finds whatever name ComfyUI actually used when the unofficial-loader was imported.
- Calling `clear_nvfp4_runtime_pools()` empties `_ACT_Q_POOL`, `_ROT_OUT_POOL`, and CUDA graphs.
- `break` after first success: one clear is enough; avoid double-clear noise.

**If the loader was never imported this session:** no matching module → no `nvfp4_runtime_pools` in the cleared list. That is correct (nothing to clear). After an NVFP4 run, the module should be present.

### 6.4 Why Distorch calls the loader API instead of duplicating pools

Pools are owned by unofficial-loader. Distorch should not reimplement pool keys. The contract is:

1. Loader exports `clear_nvfp4_runtime_pools()`.
2. Distorch Method 2c calls it after nuclear kill.
3. Next NVFP4 forward rebuilds pools on demand with live storage.

### 6.5 `__init__.py` import preference — meaning

```python
from .nodes.purge_vram import DisTorchPurgeVRAMV2  # first
# root purge_vram only if nodes import fails
```

**Meaning:**

- Operational truth is `nodes/purge_vram.py`.
- Root `purge_vram.py` can drift (older Method 2c without NVFP4 clear, line-ending churn, partial updates).
- Preferring nodes makes “Method 2c was updated in `nodes/`” equal “ComfyUI actually runs that Method 2c”.
- Startup log must show:

```text
Successfully imported DisTorchPurgeVRAMV2 from .nodes.purge_vram
```

If it shows `.purge_vram` (root), the install is on the legacy import path — NVFP4 clear may be missing until `nodes/purge_vram.py` imports successfully.

### 6.6 Log prefix `HSWQ INT8/NVFP4:` — meaning

- Toggle UI may still be labeled `HSWQ` (legacy workflows accept `HSWQ INT8`).
- Console prefix documents that this purge path handles **INT8 kitchen + NVFP4 pools**.
- Operators grepping logs for second-gen bugs should search `HSWQ INT8/NVFP4` and `nvfp4_runtime_pools`.

### 6.7 Causal chain (compact)

```text
Run 1 NVFP4
  → fills _ACT_Q_POOL / graphs with live CUDA tensors
Distorch HSWQ purge Method 0s/3
  → destroys CUDA storage of those tensors (VRAM freed)
Method 2c kitchen-only (old)
  → kitchen dicts cleared; NVFP4 pools STILL hold dead tensors
Run 2 NVFP4
  → quantize_nvfp4 sees PyCapsule / TypeError
Method 2c + clear_nvfp4_runtime_pools (new)
  → pools empty; Run 2 allocates fresh buffers → OK
```

### 6.8 Verification checklist

1. Full ComfyUI restart after updating **ComfyUI-DistorchMemoryManager** to a build that includes this Method 2c change.
2. Startup: import from `.nodes.purge_vram`.
3. Run NVFP4 once.
4. Run Distorch HSWQ purge; confirm Method 2c logs `nvfp4_runtime_pools` / “Cleared HSWQ NVFP4…”.
5. Second NVFP4 prompt: no `pooled TC path failed`, no `PyCapsule` on `quantize_nvfp4`.

---

## Appendix A — Relation to unofficial-loader (out of Distorch scope)

Loader-side pool clear on early-return / stack upgrade is a **separate** safety net for paths that never run Distorch Method 2c. For the reported failure (**purge then second prompt**), Distorch Method 2c NVFP4 clear is the primary fix.

## Appendix B — Document map

| Section | This document |
|---------|---------------|
| (1) Error content | §1 |
| (2) Root cause | §2 |
| (3) Countermeasure overview | §3 |
| (4) Modified file names | §4 |
| (5) Full modified code | §5 |
| (6) Meaning | §6 |

---

**End of Distorch NVFP4 Method 2c guide.**
