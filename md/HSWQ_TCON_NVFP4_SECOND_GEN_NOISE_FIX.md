# HSWQ tcon NVFP4 Second-Generation Noise & SEG Cache Issue - Fix Guide

**Date:** 2026-08-26
**Scope:** ComfyUI-DistorchMemoryManager + ComfyUI-HSWQ-Loader-and-Tools
**Symptom:** With tcon (Z Image TC/W4A4) NVFP4 models, the second generation after a purge turns into noise.

---

## 1. Second-generation noise problem (tcon NVFP4)

### Root cause

After the purge strips the HSWQ stack, the second generation fails to LoRA-bake the NVFP4 layers correctly, producing noise.

Log evidence:

| Generation | Bake result | State |
|------|-----------|-------|
| 1st | `nvfp4_baked=86 int8_baked=94` / `NVFP4_LORA_BAKE_OK` | OK |
| 2nd | `nvfp4_baked=0 other_qt_baked=83` / `NVFP4_LORA_BAKE_N/A` | NVFP4 misclassified as `other_qt`, baked without ConvRot -> noise |

### Causal chain

1. The purge peels the `ops._load_quantized_module` wrap (`_hswq_nvfp4_full_load` stamp disappears)
2. `apply_comfy_quant_nvfp4_patches()` early-returns based only on the `_PATCHES_APPLIED` flag and `stack_ver`
3. On reload, `arm_nvfp4_module` never runs, so modules never get the `_hswq_nvfp4_convrot` flag
4. The bake function cannot detect NVFP4 layers and mis-bakes them as `other_qt` (ConvRot rotation not applied)
5. Weights are corrupted -> noise

### Fixes (3)

#### `fdc60bc` - HSWQ-Loader-and-Tools (primary fix)

`nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py`

Added the `_load_wrap_ok` condition to both early-return paths:

```python
_load_wrap_ok = bool(
    getattr(ops._load_quantized_module, "_hswq_nvfp4_full_load", False)
)
if (
    _PATCHES_APPLIED
    and _load_wrap_ok          # <- added: do not early-return if the wrap was peeled
    and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
    and stack_ver >= _NVFP4_STACK_VER
):
    return True
```

When the purge peeled the wrap, both early returns are skipped, the full re-application runs, `_load_quantized_module` is re-wrapped, and modules are re-armed.

#### `d97bb5b` - HSWQ-Loader-and-Tools

`nodes/zimage_nvfp4/load_unet.py`

Added `_install_permanent_dynamic_load_guard()`: an outer guard on `ModelPatcherDynamic.load` that does NOT carry the `_hswq_zi_nvfp4_lora_bake` stamp, so the purge's deep-clean walks past it. Every `Dynamic.load` calls `_ensure_dynamic_load_bake_wrap()` and re-arms the bake hook if it was peeled (no-op when already armed).

#### `2936341` - DistorchMemoryManager

`purge_vram.py`

After the purge completes, sets `unload_models` + `free_memory` queue flags so ComfyUI's executor drops the loader nodes' output cache (MODEL objects) and the next generation re-runs the loader nodes, re-applying the TC (W4A4) stack patches inside `load_unet`. The purge's own full-reset behavior is unchanged.

---

## 2. SEG cache issue

### `2bc0075` - DistorchMemoryManager

**Root fix for the "Tried to unpin tensor not pinned by ComfyUI" warning**

Cause: `mm.unpin_memory()` can only handle tensors registered in ComfyUI's `PINNED_MEMORY` dict. Detailer/SEGS cache tensors are unregistered, so it logs the warning and returns `False` (it never raises). The old `try/except` fallback therefore never triggered - warnings were printed while the tensors stayed unpinned.

Fix: check `PINNED_MEMORY` registration before unpinning.

- Registered -> `mm.unpin_memory()` (kept in sync with ComfyUI's bookkeeping)
- Unregistered -> direct `cudaHostUnregister()` (released reliably, no warning)

Not a warning silencer: routes each tensor to the correct release path based on registration state.

### `f59716a` - DistorchMemoryManager

**Complete removal of the Detailer/SEGS cache path**

Obsolete Detailer/SEGS cache handling removed:

- `_drain_hswq_pin_cache()` deleted
- `_purge_detailer_segs_and_executor_cache()` deleted
- 4 call sites (Method 0 / 0s / 0b / 0s2) deleted
- 265 lines removed in total

The HSWQ full reset (model unload, PINNED_MEMORY unregister, kitchen caches, Hadamard, Linear bake peel, VRAM release) is unchanged.

---

## Commit list

| Repository | Commit | Content |
|-----------|---------|---------|
| HSWQ-Loader-and-Tools | `fdc60bc` | Full stack re-application when the purge peeled the `_load_quantized_module` wrap |
| HSWQ-Loader-and-Tools | `d97bb5b` | Permanent Dynamic.load guard (cannot be peeled) auto re-arms the bake hook |
| DistorchMemoryManager | `2936341` | Reset loader-node output cache after purge (TC stack rebuild) |
| DistorchMemoryManager | `2bc0075` | Root fix for the unpin warning via PINNED_MEMORY registration check |
| DistorchMemoryManager | `f59716a` | Remove the Detailer/SEGS cache sweep |

**All synced to dev / installed and pushed.**
