# Release Notes v2.4.6: HSWQ Pin/Unpin Memory Safety, CUDA Abort Crash Prevention & TCON NVFP4 TC Re-arm Fix

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v2.4.6.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

**Module:** `DisTorchPurgeVRAMV2` (General Purge VRAM V2)  
**Target Repository:** `ussoewwin/ComfyUI-DistorchMemoryManager`  
**Base Release:** `v2.4.5`  
**Release Tag:** `v2.4.6`  
**Date:** 2026-08-27  

---

## Table of contents

1. [Executive Summary & Technical Background](#1-executive-summary--technical-background)
2. [Root Cause Analysis](#2-root-cause-analysis)
   - [2.1 Unpin Warning Flood (`Tried to unpin tensor not pinned by ComfyUI`)](#21-unpin-warning-flood-tried-to-unpin-tensor-not-pinned-by-comfyui)
   - [2.2 CUDA Driver Abort (`Fatal Python error: Aborted`)](#22-cuda-driver-abort-fatal-python-error-aborted)
   - [2.3 TCON NVFP4 Second Generation Noise & TC Stack Re-arming](#23-tcon-nvfp4-second-generation-noise--tc-stack-re-arming)
3. [Architecture & Implementation Changes](#3-architecture--implementation-changes)
   - [3.1 Strict `PINNED_MEMORY` Guard on `mm.unpin_memory`](#31-strict-pinned_memory-guard-on-mmunpin_memory)
   - [3.2 Elimination of Raw `cudaHostUnregister` on Untracked Tensors](#32-elimination-of-raw-cudahostunregister-on-untracked-tensors)
   - [3.3 Queue Flags for Executor Cache Invalidation & Loader Re-arming](#33-queue-flags-for-executor-cache-invalidation--loader-re-arming)
   - [3.4 Cleanup of Obsolete PinCache & Detailer SEGS Sweeps](#34-cleanup-of-obsolete-pincache--detailer-segs-sweeps)
4. [Source Code Modifications & Verification](#4-source-code-modifications--verification)

---

## 1. Executive Summary & Technical Background

Version **v2.4.6** of `ComfyUI-DistorchMemoryManager` resolves memory management defects encountered during deep HSWQ INT8 / NVFP4 VRAM purge workflows:

1. **Suppression of False Unpin Warnings**: Eliminates repeated `[WARNING] Tried to unpin tensor not pinned by ComfyUI` log spam during Purge VRAM execution.
2. **CUDA Driver Abort Prevention**: Eliminates `Fatal Python error: Aborted` crashes caused by illegal `cudaHostUnregister` calls on PyTorch page-locked host memory.
3. **TCON NVFP4 TC Re-arming**: Enforces executor cache invalidation after HSWQ stack peeling so subsequent generations re-arm the Tensor Core (W4A4) pipeline cleanly without 2nd-generation noise.
4. **Detailer / PinCache Sweep Cleanup**: Deprecates obsolete `_drain_hswq_pin_cache` and `_purge_detailer_segs_and_executor_cache` routines following the removal of Pin Buffer Cache in HSWQ.

---

## 2. Root Cause Analysis

### 2.1 Unpin Warning Flood (`Tried to unpin tensor not pinned by ComfyUI`)

When `_kill_tensor_storage` or GC nuclear sweeps encountered tensors with `tensor.is_pinned() == True`:
- PyTorch native tensors allocated in pinned host RAM (via `torch.empty(..., pin_memory=True)`, DataLoader workers, or SEGS crop buffers) return `True` for `is_pinned()`.
- However, these tensors were never registered into ComfyUI's internal tracking dictionary (`comfy.model_management.PINNED_MEMORY`), which only tracks buffers registered via ComfyUI's `mm.pin_memory()` through `cudaHostRegister`.
- When `mm.unpin_memory(tensor)` is called on a pointer absent from `PINNED_MEMORY`, ComfyUI emits `logging.warning("Tried to unpin tensor not pinned by ComfyUI")`.

### 2.2 CUDA Driver Abort (`Fatal Python error: Aborted`)

- Pinned host memory in PyTorch falls into two distinct categories:
  1. **Host-allocated memory** (`cudaHostAlloc` / `cudaMallocHost`): Allocated directly as page-locked memory by PyTorch allocator. Calling `cudaHostUnregister` on this memory is illegal in CUDA and raises `cudaErrorInvalidValue`.
  2. **Host-registered memory** (`cudaHostRegister`): Standard pageable host memory registered into the CUDA MMU page table. Only this memory can be unregistered via `cudaHostUnregister`.
- When raw `cudaHostUnregister(data.data_ptr())` was invoked as a fallback for untracked pinned tensors, CUDA driver page-table tracking became corrupted.
- Consequently, during subsequent runtime pool clearing (`clear_nvfp4_runtime_pools`) or `torch.cuda.empty_cache()`, the CUDA runtime encountered corrupted internal driver state and terminated the Python process with `Fatal Python error: Aborted`.

### 2.3 TCON NVFP4 Second Generation Noise & TC Stack Re-arming

- When HSWQ purge peels ops wrappers, Hadamard caches, and runtime pools, the cached ModelPatcher outputs in ComfyUI's execution cache retain references to modified modules without re-running the loader node.
- On subsequent prompt execution, the unet was loaded without re-arming the NVFP4 Tensor Core (W4A4) stack, resulting in numerical mismatch and noisy image generation.

---

## 3. Architecture & Implementation Changes

### 3.1 Strict `PINNED_MEMORY` Guard on `mm.unpin_memory`

```python
if bool(getattr(data, "is_pinned", lambda: False)()):
    try:
        import comfy.model_management as mm
        _pm = getattr(mm, "PINNED_MEMORY", None) or {}
        _ptr = data.data_ptr()
        if _ptr in _pm:
            # Registered with ComfyUI: unpin via mm (keeps bookkeeping in sync).
            mm.unpin_memory(data)
    except Exception:
        pass
```

Only tensors explicitly tracked in `mm.PINNED_MEMORY` are passed to `mm.unpin_memory()`. Untracked tensors are skipped without emitting warnings.

### 3.2 Elimination of Raw `cudaHostUnregister` on Untracked Tensors

All direct invocations of `torch.cuda.cudart().cudaHostUnregister()` on unverified memory pointers in `_kill_tensor_storage` and `Method 3` have been completely removed. `cudaHostUnregister` is exclusively used in `_force_unregister_comfy_pins()` over confirmed `PINNED_MEMORY` keys.

### 3.3 Queue Flags for Executor Cache Invalidation & Loader Re-arming

```python
try:
    mm.unload_models = True
    mm.free_memory = 1e30
except Exception:
    pass
```

Forces PromptExecutor to drop cached loader-node outputs, ensuring that subsequent prompts re-execute the loader and properly re-arm the Tensor Core stack.

### 3.4 Cleanup of Obsolete PinCache & Detailer SEGS Sweeps

Obsolete PinCache drain routines (`_drain_hswq_pin_cache`) and SEGS sweep functions (`_purge_detailer_segs_and_executor_cache`) were removed from both `nodes/purge_vram.py` and `purge_vram.py` to maintain a streamlined, zero-overhead purge execution flow.

---

## 4. Source Code Modifications & Verification

- **Primary Module**: [`nodes/purge_vram.py`](file:///D:/USERFILES/GitHub/ComfyUI-DistorchMemoryManager/nodes/purge_vram.py)
- **Fallback Module**: [`purge_vram.py`](file:///D:/USERFILES/GitHub/ComfyUI-DistorchMemoryManager/purge_vram.py)
- **Changelog**: [`changelog/changelog.md`](file:///D:/USERFILES/GitHub/ComfyUI-DistorchMemoryManager/changelog/changelog.md), [`zhmd/changelog/changelog.md`](file:///D:/USERFILES/GitHub/ComfyUI-DistorchMemoryManager/zhmd/changelog/changelog.md)
- **Live Deployment**: Synced directly to `custom_nodes/ComfyUI-DistorchMemoryManager/`
