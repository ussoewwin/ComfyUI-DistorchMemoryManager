# Distorch HSWQ Purge — Krea2 ConvRot NVFP4 Architecture and Complete Implementation Specification

**Module:** `DisTorchPurgeVRAMV2` (General Purge VRAM V2)  
**Target Architecture:** Krea2 ConvRot NVFP4 / INT8 Low-Rank LoRA Residual Branch  
**Target Repository:** `ussoewwin/ComfyUI-HSWQ-Loader-and-Tools`  
**Base Commit:** `85e46d2de81ee8f03e69871e99b52f1dc1400ce1`  
**Fixed Commit:** `a46e669b56ebf7fefc1b7f5a3b75583302e41c21`  
**Date:** 2026-08-24  

---

## 1. Architectural Significance and Technical Background

### 1.1 Krea2 ConvRot NVFP4 Execution Pipeline

In the latest update of `ComfyUI-HSWQ-Loader-and-Tools`, Krea2 DiT models introduced a dedicated **ConvRot NVFP4 (Tensor Core)** pipeline alongside a hybrid **Low-Rank LoRA Residual Baking** mechanism. The architecture operates across several interdependent layers:

1. **Rank-Decomposed LoRA Residual Accumulation (`_hswq_krea2_lora_res` / `_hswq_krea2_lora_res_gpu`)**:
   - In standard weight baking, applying LoRA weights to 4-bit quantized matrices requires dequantizing, adding the delta, and requantizing to NVFP4. Because small style LoRA weights introduce micro-deltas ($\sim 0.1\% - 0.8\%$ of weight $a_{max}$), direct 4-bit requantization ($\sim 4\% - 8\%$ step size per code) rounds away fine details and degrades high-frequency style fidelity.
   - To preserve full mathematical accuracy without dequantization loss, Krea2 preserves the base packed 4-bit tensor in VRAM and stores unquantized LoRA deltas as rank-decomposed down/up projections directly on `torch.nn.Module` instances:
     $$\Delta W = \text{scale} \cdot (\text{mat\_up} \times \text{mat\_dn})$$
   - These low-rank tensors are stored inside a nested list of tuples:
     `_hswq_krea2_lora_res = [(mat_dn_0, mat_up_0, scale_0), (mat_dn_1, mat_up_1, scale_1), ...]`
   - In addition, pre-allocated GPU staging buffers are cached in `_hswq_krea2_lora_res_gpu`.
2. **Krea2 Ops Wrapping and Dispatch Hierarchy**:
   - `comfy.ops.mixed_precision_ops` is wrapped with `_hswq_krea2_stack` to dynamically inject Krea2 NVFP4 tensor-core Linear operations.
   - `comfy.utils.convert_old_quants` is intercepted with `_hswq_krea2_oldquants` to handle legacy quantized checkpoints.
   - `comfy.model_detection.detect_unet_config` is patched via `_hswq_krea2_txtlayers_fix` to override `txtlayers=8` misdetection caused by packed projector dimension checks, ensuring 12-layer CLIP conditioning operates properly.
   - `ModelPatcherDynamic.load` and `comfy.model_management.load_models_gpu` are intercepted to attach dynamic Krea2 bake hooks.
3. **Module-Level Pooling and CUDA Graph Allocations**:
   - `_ACT_Q_POOL` (activation quantization buffer pool), `_ROT_OUT_POOL` (rotation matrix output buffer pool), `_GRAPH_CACHE` (CUDA Graph instances), and `_HADAMARD_CACHE` remain resident in global module memory.

### 1.2 Failure Modes in Previous Purge Implementations

Prior to this update (`v2.4.3`), triggering an HSWQ purge left several critical failure surfaces:

- **Silent VRAM Leaks and Residual Cross-Contamination**:
  - Legacy `_empty_cuda_tensor` and `_kill_tensor_storage` functions inspected only top-level `torch.Tensor` instances (`if torch.is_tensor(t)`).
  - Because `_hswq_krea2_lora_res` is a `list` of `tuple` containers, the internal `mat_dn` and `mat_up` GPU tensors were skipped during memory sweeps. These tensors remained resident in VRAM, causing unrecoverable memory retention and contaminating subsequent model loads with stale LoRA residuals.
- **Unpeeled Global Wrappers and Model Corruption**:
  - Unloading a Krea2 model without unwrapping `_hswq_krea2_stack`, `_hswq_krea2_oldquants`, and `_hswq_krea2_txtlayers_fix` caused subsequent loads of SDXL, Flux, or Z Image models to execute through Krea2-specific ops and incorrect UNet config detection logic.
- **Stale CUDA Graphs and Buffer Desynchronization**:
  - Persistent CUDA Graph instances referencing destroyed or reallocated tensor addresses resulted in illegal memory accesses (`CUDA error: invalid argument` / `PyCapsule` invocation errors) on subsequent generations.

---

## 2. Modified and Created Files

- **Modified Files**:
  1. `nodes/purge_vram.py`
  2. `purge_vram.py`  
  *(Both files maintain 100% identical source code with zero diff across the fallback tree)*

- **Documentation**:
  1. `md/HSWQ_KREA2_NVFP4_PURGE_COMPLETE_GUIDE.md` (This document)

---

## 3. Complete Source Code Diff (from commit `85e46d2de81ee8f03e69871e99b52f1dc1400ce1`)

The following diff represents the exact, complete, and unabridged code modifications committed to the repository:

```diff
diff --git a/nodes/purge_vram.py b/nodes/purge_vram.py
index 23d36b6..d1ba4ca 100644
--- a/nodes/purge_vram.py
+++ b/nodes/purge_vram.py
@@ -2438,16 +2438,27 @@ class DisTorchPurgeVRAMV2:
                         return 0
 
                     def _empty_cuda_tensor(t) -> None:
-                        if t is None or not torch.is_tensor(t):
-                            return
-                        try:
-                            data = getattr(t, "data", t)
-                            if not bool(getattr(data, "is_cuda", False)):
-                                return
-                            empty = torch.empty(0, dtype=data.dtype, device=data.device)
-                            t.data = empty
-                        except Exception:
-                            pass
+                        if t is None:
+                            return
+                        if torch.is_tensor(t):
+                            try:
+                                data = getattr(t, "data", t)
+                                if not bool(getattr(data, "is_cuda", False)):
+                                    return
+                                empty = torch.empty(0, dtype=data.dtype, device=data.device)
+                                t.data = empty
+                            except Exception:
+                                pass
+                            return
+                        # Nested containers (e.g. _hswq_krea2_lora_res = [(mat_dn, mat_up, scale), ...])
+                        if isinstance(t, (list, tuple)):
+                            for item in t:
+                                if item is not None:
+                                    _empty_cuda_tensor(item)
+                        elif isinstance(t, dict):
+                            for v in t.values():
+                                if v is not None:
+                                    _empty_cuda_tensor(v)
 
                     # Known residual names (INT8 + NVFP4 + bake + forward caches).
                     # Stray walk below also drops every other ``_hswq_*`` on Modules.
@@ -2474,10 +2485,19 @@ class DisTorchPurgeVRAMV2:
                         # INT8 Conv2d ConvRot (comfy_quant_int8 QuantConv2d)
                         "_hswq_convrot",
                         "_hswq_convrot_groupsize",
-                        # Krea2 ConvRot NVFP4 (bake bookkeeping / pack stamps)
+                        # Krea2 ConvRot NVFP4 (bake bookkeeping / pack stamps / LoRA residuals)
                         "_hswq_krea2_nvfp4_pack",
                         "_hswq_krea2_nvfp4_baked_keys",
                         "_hswq_krea2_nvfp4_baked_uuid",
+                        "_hswq_krea2_lora_res",
+                        "_hswq_krea2_lora_res_gpu",
+                        "_hswq_krea2_tc",
+                        "_hswq_krea2_stack",
+                        "_hswq_krea2_full_load",
+                        "_hswq_krea2_oldquants",
+                        "_hswq_krea2_prev_oldquants",
+                        "_hswq_krea2_txtlayers_fix",
+                        "_hswq_krea2_prev_dynamic_load",
                     )
 
                     # --- comfy_kitchen ---
@@ -2620,11 +2640,13 @@ class DisTorchPurgeVRAMV2:
                             f"{e_load_peel}"
                         )
 
-                    # --- Peel Krea2 ConvRot NVFP4 stack (mixed_precision_ops +
-                    #     convert_old_quants). uninstall_krea2_nvfp4_lora_bake only
-                    #     peels Dynamic.load / load_models_gpu, not these ops wraps. ---
+                    # --- Peel Krea2 ConvRot NVFP4 stack (mixed_precision_ops +
+                    #     convert_old_quants + detect_unet_config). uninstall_krea2_nvfp4_lora_bake only
+                    #     peels Dynamic.load / load_models_gpu, not these ops wraps. ---
                     try:
                         import comfy.ops as _ops_peel_krea2_mp
                         import comfy.utils as _utils_peel_krea2
+                        import comfy.model_detection as _md_peel_krea2
 
                         def _peel_krea2_mp_once():
                             cur = getattr(_ops_peel_krea2_mp, "mixed_precision_ops", None)
@@ -2657,8 +2679,25 @@ class DisTorchPurgeVRAMV2:
                                 cur = nxt
                             return peeled
 
+                        def _peel_krea2_txtlayers_once():
+                            cur = getattr(_md_peel_krea2, "detect_unet_config", None)
+                            seen = set()
+                            peeled = 0
+                            while cur is not None and callable(cur) and id(cur) not in seen:
+                                seen.add(id(cur))
+                                if not getattr(cur, "_hswq_krea2_txtlayers_fix", False):
+                                    break
+                                nxt = _closure_load_cell(cur, "_prev_detect_txt")
+                                if nxt is None or nxt is cur:
+                                    break
+                                _md_peel_krea2.detect_unet_config = nxt
+                                peeled += 1
+                                cur = nxt
+                            return peeled
+
                         _mp_peeled_k2 = _peel_krea2_mp_once()
                         _oq_peeled_k2 = _peel_krea2_oldquants_once()
+                        _txt_peeled_k2 = _peel_krea2_txtlayers_once()
                         if _mp_peeled_k2:
                             cleared.append(f"krea2_mp_stack_peel={_mp_peeled_k2}")
                             print(
@@ -2670,6 +2709,12 @@ class DisTorchPurgeVRAMV2:
                                 "HSWQ INT8/NVFP4: peeled Krea2 convert_old_quants "
                                 f"layers={_oq_peeled_k2}"
                             )
+                        if _txt_peeled_k2:
+                            cleared.append(f"krea2_txtlayers_peel={_txt_peeled_k2}")
+                            print(
+                                "HSWQ INT8/NVFP4: peeled Krea2 detect_unet_config txtlayers "
+                                f"fix layers={_txt_peeled_k2}"
+                            )
                     except Exception as e_krea2_peel:
                         print(
                             "HSWQ INT8/NVFP4: Krea2 stack peel failed: "
@@ -3006,10 +3051,11 @@ class DisTorchPurgeVRAMV2:
                             continue
                         nlow = str(name).replace("\\", "/").lower()
                         if not (
-                            "nvfp4_comfy_parity" in nlow
-                            or "nvfp4_forward" in nlow
-                            or "zi_nvfp4_forward" in nlow
-                            or "comfy_quant_nvfp4" in nlow
+                            "nvfp4" in nlow
+                            or "zi_nvfp4" in nlow
+                            or "comfy_quant" in nlow
+                            or "hswq" in nlow
+                            or "patches" in nlow
                         ):
                             continue
                         for api_name in (
@@ -3016,5 +3062,9 @@ class DisTorchPurgeVRAMV2:
                             "clear_nvfp4_parity_hadamard_caches",
                             "reset_nvfp4_forward_stats",
                             "reset_nvfp4_lora_log_counters",
+                            "reset_krea2_nvfp4_lora_bake_log_counters",
+                            "reset_int8_lora_log_counters",
+                            "clear_nvfp4_runtime_pools",
+                            "clear_nvfp4_cudagraphs",
                         ):
                             fn = _safe_getattr(mod, api_name, None)
                             if not callable(fn):
@@ -3141,7 +3191,7 @@ class DisTorchPurgeVRAMV2:
                     return cur if _is_real_nn(cur) else None
 
                 def _is_hswq_int8_nn(module) -> bool:
-                    """True for HSWQ INT8 and/or NVFP4 (incl. ZI ConvRot) UNet modules.
+                    """True for HSWQ INT8 and/or NVFP4 (incl. ZI / Krea2 ConvRot) UNet modules.
 
                     Pure NVFP4 packs have ``format=nvfp4`` comfy_quant markers and
                     ``_hswq_nvfp4_convrot`` arms — they are not ``int8_tensorwise``.
@@ -3161,6 +3211,10 @@ class DisTorchPurgeVRAMV2:
                         return True
                     if getattr(module, "_hswq_krea2_nvfp4_pack", False):
                         return True
+                    if getattr(module, "_hswq_krea2_lora_res", None) is not None:
+                        return True
+                    if getattr(module, "_hswq_krea2_lora_res_gpu", None) is not None:
+                        return True
                     try:
                         for m in module.modules():
                             if (
@@ -3174,6 +3228,8 @@ class DisTorchPurgeVRAMV2:
                                 or getattr(m, "_hswq_krea2_nvfp4_baked_keys", None)
                                 or getattr(m, "_hswq_krea2_nvfp4_baked_uuid", None) is not None
                                 or getattr(m, "_hswq_krea2_nvfp4_pack", False)
+                                or getattr(m, "_hswq_krea2_lora_res", None) is not None
+                                or getattr(m, "_hswq_krea2_lora_res_gpu", None) is not None
                             ):
                                 return True
                             # Any residual ``_hswq_*`` (INT8 / NVFP4 / bake / TC caches)
@@ -3235,6 +3291,18 @@ class DisTorchPurgeVRAMV2:
                     # empty(0) only for CUDA. CPU wipe broke ZI TE after Ollama purge.
                     if t is None:
                         return 0
+                    if isinstance(t, (list, tuple)):
+                        freed_nested = 0
+                        for item in t:
+                            if item is not None:
+                                freed_nested += _kill_tensor_storage(item)
+                        return freed_nested
+                    if isinstance(t, dict):
+                        freed_nested = 0
+                        for v in t.values():
+                            if v is not None:
+                                freed_nested += _kill_tensor_storage(v)
+                        return freed_nested
                     freed = 0
                     try:
                         data = getattr(t, "data", t)
@@ -3314,10 +3382,19 @@ class DisTorchPurgeVRAMV2:
                         "_hswq_int8_convrot_groupsize",
                         "_hswq_convrot",
                         "_hswq_convrot_groupsize",
-                        # Krea2 ConvRot NVFP4 (bake bookkeeping / pack stamps)
+                        # Krea2 ConvRot NVFP4 (bake bookkeeping / pack stamps / LoRA residuals)
                         "_hswq_krea2_nvfp4_pack",
                         "_hswq_krea2_nvfp4_baked_keys",
                         "_hswq_krea2_nvfp4_baked_uuid",
+                        "_hswq_krea2_lora_res",
+                        "_hswq_krea2_lora_res_gpu",
+                        "_hswq_krea2_tc",
+                        "_hswq_krea2_stack",
+                        "_hswq_krea2_full_load",
+                        "_hswq_krea2_oldquants",
+                        "_hswq_krea2_prev_oldquants",
+                        "_hswq_krea2_txtlayers_fix",
+                        "_hswq_krea2_prev_dynamic_load",
                     )
                     try:
                         for m in module.modules():
@@ -3325,7 +3402,7 @@ class DisTorchPurgeVRAMV2:
                                     continue
                                 try:
                                     val = getattr(m, attr, None)
-                                    if torch.is_tensor(val):
+                                    if val is not None:
                                         freed += _kill_tensor_storage(val)
                                 except Exception:
                                     pass
```

---

## 4. Technical Analysis and Implementation Rationale

### 4.1 Recursive Container Traversal for Tensor Memory Eradication

```python
                    def _empty_cuda_tensor(t) -> None:
                        if t is None:
                            return
                        if torch.is_tensor(t):
                            try:
                                data = getattr(t, "data", t)
                                if not bool(getattr(data, "is_cuda", False)):
                                    return
                                empty = torch.empty(0, dtype=data.dtype, device=data.device)
                                t.data = empty
                            except Exception:
                                pass
                            return
                        # Nested containers (e.g. _hswq_krea2_lora_res = [(mat_dn, mat_up, scale), ...])
                        if isinstance(t, (list, tuple)):
                            for item in t:
                                if item is not None:
                                    _empty_cuda_tensor(item)
                        elif isinstance(t, dict):
                            for v in t.values():
                                if v is not None:
                                    _empty_cuda_tensor(v)
```
- **Mechanism**: In PyTorch, assigning `None` or executing `delattr` on an object drops the Python reference, but if a cycle or external namespace holds a reference, CUDA allocations remain resident in GPU memory. `_empty_cuda_tensor` mutates the underlying `Tensor.data` pointer in-place using `torch.empty(0, dtype=data.dtype, device=data.device)`, forcing immediate deallocation by the PyTorch CUDA caching allocator.
- **Recursive Container Handling**: By supporting `list`, `tuple`, and `dict` structures, this logic traverses arbitrary nestings, ensuring every low-rank tensor tuple `(mat_dn, mat_up, scale)` within `_hswq_krea2_lora_res` is zeroed.

### 4.2 Comprehensive Krea2 Residual Attribute Registration

```python
                        "_hswq_krea2_nvfp4_pack",
                        "_hswq_krea2_nvfp4_baked_keys",
                        "_hswq_krea2_nvfp4_baked_uuid",
                        "_hswq_krea2_lora_res",
                        "_hswq_krea2_lora_res_gpu",
                        "_hswq_krea2_tc",
                        "_hswq_krea2_stack",
                        "_hswq_krea2_full_load",
                        "_hswq_krea2_oldquants",
                        "_hswq_krea2_prev_oldquants",
                        "_hswq_krea2_txtlayers_fix",
                        "_hswq_krea2_prev_dynamic_load",
```
- **Mechanism**: Modules scanned during GC traversal have all execution marks, UUID identifiers, tensor-core mode markers, and residual tensors explicitly dropped and deleted. This prevents stale state reuse across model reloads.

### 4.3 Closure-Level Unwrapping of `detect_unet_config` (`_hswq_krea2_txtlayers_fix`)

```python
                        def _peel_krea2_txtlayers_once():
                            cur = getattr(_md_peel_krea2, "detect_unet_config", None)
                            seen = set()
                            peeled = 0
                            while cur is not None and callable(cur) and id(cur) not in seen:
                                seen.add(id(cur))
                                if not getattr(cur, "_hswq_krea2_txtlayers_fix", False):
                                    break
                                nxt = _closure_load_cell(cur, "_prev_detect_txt")
                                if nxt is None or nxt is cur:
                                    break
                                _md_peel_krea2.detect_unet_config = nxt
                                peeled += 1
                                cur = nxt
                            return peeled
```
- **Mechanism**: The Krea2 NVFP4 loader wraps `comfy.model_detection.detect_unet_config` in a closure that captures the original detector as `_prev_detect_txt`. `_peel_krea2_txtlayers_once` uses `_closure_load_cell` to inspect the function's `__closure__` cells, extracts `_prev_detect_txt`, and restores the original function pointer on `comfy.model_detection`. This completely neutralizes Krea2-specific projector overrides for subsequent workflows.

### 4.4 Broad Dynamic Submodule Scanning and API Reset Loops

```python
                        for api_name in (
                            "clear_nvfp4_parity_hadamard_caches",
                            "reset_nvfp4_forward_stats",
                            "reset_nvfp4_lora_log_counters",
                            "reset_krea2_nvfp4_lora_bake_log_counters",
                            "reset_int8_lora_log_counters",
                            "clear_nvfp4_runtime_pools",
                            "clear_nvfp4_cudagraphs",
                        ):
```
- **Mechanism**: Dynamically iterates over all loaded Python modules matching `nvfp4`, `zi_nvfp4`, `comfy_quant`, `hswq`, or `patches` within `sys.modules`. It executes all cleanup entry points to flush Hadamard matrix lookup tables, reset statistics and log spam counters, clear activation/rotation buffer pools, and invalidate CUDA Graph handles.

### 4.5 Recursive Storage Deallocation in `_kill_tensor_storage` & Module VRAM Purge

```python
                def _kill_tensor_storage(t) -> int:
                    if t is None:
                        return 0
                    if isinstance(t, (list, tuple)):
                        freed_nested = 0
                        for item in t:
                            if item is not None:
                                freed_nested += _kill_tensor_storage(item)
                        return freed_nested
                    if isinstance(t, dict):
                        freed_nested = 0
                        for v in t.values():
                            if v is not None:
                                freed_nested += _kill_tensor_storage(v)
                        return freed_nested
                    freed = 0
                    try:
                        data = getattr(t, "data", t)
                        if getattr(data, "is_cuda", False):
                            sz = int(data.nelement()) * int(data.element_size())
                            freed += sz
                            t.data = torch.empty(0, dtype=data.dtype, device=data.device)
                    except Exception:
                        pass
                    return freed
```
- **Mechanism**: In `_kill_module_vram`, the guard `if val is not None:` ensures container objects such as lists and tuples are passed into `_kill_tensor_storage`, which recursively zeroes all nested CUDA storage before dropping the module attribute.
