# SageAttention Patch Feature — Complete Guide

## Overview

`ComfyUI-DistorchMemoryManager` gained a node that patches ComfyUI's attention path to SageAttention. The standard attention implementation can be replaced with SageAttention for better memory efficiency and performance.

## Files added / modified

**File:** `__init__.py`

Only this file was changed for the initial patch addition (later work lives in `nodes/sa.py`).

## Added code and meaning

### 1. Imports

```python
from comfy.ldm.modules import attention as comfy_attention
import comfy.model_management as mm
from comfy.cli_args import args
from comfy.ldm.modules.attention import wrap_attn
```

**Meaning:**
- `comfy_attention`: access to ComfyUI attention helpers (inspect original attention)
- `mm`: `model_management` (Flash-Attention state / version when available)
- `args`: CLI flags (whether Flash-Attention was enabled from the command line)
- `wrap_attn`: decorator that matches ComfyUI's attention calling convention

### 2. SageAttention mode list (around line 2179)

```python
sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]
```

**Meaning:** selectable SageAttention implementations

- `disabled`: turn SageAttention off (restore baseline attention)
- `auto`: automatic SageAttention implementation
- `sageattn_qk_int8_pv_fp16_cuda`: CUDA (QK int8, PV FP16)
- `sageattn_qk_int8_pv_fp16_triton`: Triton (QK int8, PV FP16)
- `sageattn_qk_int8_pv_fp8_cuda`: CUDA (QK int8, PV FP8)
- `sageattn_qk_int8_pv_fp8_cuda++`: CUDA (QK int8, PV FP8, optimized)
- `sageattn3`: SageAttention 3 (Blackwell)
- `sageattn3_per_block_mean`: SageAttention 3 with per-block mean

### 3. `get_sage_func_dm` (lines 2181–2273)

Returns the attention function for the selected mode.

#### Version detection and logging (lines 2182–2206)

```python
def get_sage_func_dm(sage_attention, allow_compile=False):
    # Detect SageAttention version
    try:
        import sageattention
        sage_version = None
        try:
            sage_version = sageattention.__version__
        except AttributeError:
            try:
                import importlib.metadata
                sage_version = importlib.metadata.version("sageattention")
            except Exception:
                sage_version = None
        
        if sage_version and sage_version != "unknown":
            try:
                import torch
                cuda_version = torch.version.cuda or "unknown"
                torch_version = torch.version.__version__ or "unknown"
                logging.info(f"Patching comfy attention to use SageAttention {sage_version}+cu{cuda_version}torch{torch_version}")
            except:
                logging.info(f"Patching comfy attention to use SageAttention {sage_version}")
        else:
            logging.info("Patching comfy attention to use sageattn")
    except:
        logging.info("Patching comfy attention to use sageattn")
```

**Meaning:**
1. Detect SageAttention version (`__version__` or `importlib.metadata`)
2. Also read CUDA and PyTorch versions
3. Log a detailed line (e.g. `Patching comfy attention to use SageAttention 2.2.0+cu121torch2.3.0`)
4. Fall back to a generic message when version lookup fails

#### Per-mode function builders (lines 2208–2233)

```python
    from sageattention import sageattn
    if sage_attention == "auto":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    # ... other modes follow the same pattern
```

**Meaning:** Import the matching SageAttention entry point and wrap it in `sage_func` with the right kwargs.

#### `torch.compile` control (lines 2235–2236)

```python
    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)
```

**Meaning:** When `allow_compile=False`, disable `torch.compile` to avoid compile overhead.

#### ComfyUI-compatible wrapper (lines 2238–2272)

```python
    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        # ... tensor shape conversion ...
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        # ... output shape conversion ...
        return out
    return attention_sage
```

**Meaning:**
1. Wrap with `@wrap_attn` for ComfyUI's attention signature
2. Convert ComfyUI tensors (`q, k, v, heads`) into the layout SageAttention expects
3. Cast FP32 inputs to FP16 (SageAttention mainly runs in FP16)
4. Adjust mask ranks (batch / heads)
5. Restore original dtype and shape on the way out

### 4. `CallbacksMP` import (line 2276)

```python
from comfy.patcher_extension import CallbacksMP
```

**Meaning:** Import ComfyUI model-patch callbacks so work can run before and after model execution.

### 5. `PatchSageAttentionDM` class (lines 2277–2398)

Custom node that patches model attention to SageAttention.

#### Class / `INPUT_TYPES` (lines 2277–2292)

```python
class PatchSageAttentionDM():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."}),
        },
        "optional": {
            "allow_compile": ("BOOLEAN", {"default": False, "tooltip": "Allow the use of torch.compile for the sage attention function, requires latest sageattn 2.2.0 or higher."})
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching attention mode. This doesn't use the model patching system and thus can't be disabled without running the node again with 'disabled' option."
    CATEGORY = "Memory"
```

**Meaning:** ComfyUI node schema
- **Inputs:**
  - `model`: target MODEL
  - `sage_attention`: mode from `sageattn_modes`
  - `allow_compile`: optional `torch.compile` (default False)
- **Output:** patched MODEL
- **Category:** `Memory`

#### `patch` method (lines 2294–2398)

Runs when the node executes.

##### `patch_attention_enable` (lines 2297–2348)

```python
        @torch.compiler.disable()
        def patch_attention_enable(model):
            if sage_attention != "disabled":
                new_attention = get_sage_func_dm(sage_attention, allow_compile=allow_compile)
                def attention_override_sage(func, *args, **kwargs):
                    return new_attention.__wrapped__(*args, **kwargs)
                
                if "transformer_options" not in model.model_options:
                    model.model_options["transformer_options"] = {}
                model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
            else:
                # disabled path: clear override and log Flash-Attention state
                # ...
```

**Meaning:** Registered as `ON_PRE_RUN`; runs **before** each model execution.

1. **SageAttention enabled (lines 2299–2306):**
   - Call `get_sage_func_dm` (emits version log)
   - Install `attention_override_sage`
   - Set `model_options["transformer_options"]["optimized_attention_override"]` so ComfyUI uses SageAttention

2. **`disabled` (lines 2307–2348):**
   - Remove any existing override
   - Probe Flash-Attention via `model_management` (`FLASH_IS_AVAILABLE` → `flash_attention_enabled()` → function-name check)
   - If Flash-Attention is active, log version / FA-2 vs FA-3

##### `patch_attention_disable` (lines 2350–2393)

```python
        @torch.compiler.disable()
        def patch_attention_disable(model):
            if "transformer_options" in model.model_options:
                if "optimized_attention_override" in model.model_options["transformer_options"]:
                    del model.model_options["transformer_options"]["optimized_attention_override"]
            
            # Even when SA was on, log FA after reset
            flash_attention_enabled = False
            try:
                if hasattr(mm, 'FLASH_IS_AVAILABLE') and mm.FLASH_IS_AVAILABLE:
                    flash_attention_enabled = True
                elif hasattr(mm, 'flash_attention_enabled'):
                    flash_attention_enabled = mm.flash_attention_enabled()
                # ... Flash-Attention state checks and logging
            except:
                pass
```

**Meaning:** Registered as `ON_CLEANUP`; runs **after** each model execution.

1. **Clear override (lines 2352–2354):** remove attention override so the next run is not left patched accidentally
2. **Flash-Attention probe / log (lines 2356–2393):** same style as enable; logs baseline FA state even when SA was used for the run

##### Callback registration (lines 2395–2398)

```python
        model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_attention_enable)
        model_clone.add_callback(CallbacksMP.ON_CLEANUP, patch_attention_disable)
        
        return model_clone,
```

**Meaning:**
- `ON_PRE_RUN`: apply SageAttention before each run
- `ON_CLEANUP`: clean up and log after each run
- Callbacks keep logging and patching tied to every generation

### 6. Node registration (lines 2402–2416)

```python
# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "MemoryManager": MemoryManager,
    "SafeMemoryManager": SafeMemoryManager,
    "DisTorchPurgeVRAMV2": DisTorchPurgeVRAMV2,
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,
    "PatchSageAttentionDM": PatchSageAttentionDM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryManager": "Memory Manager",
    "SafeMemoryManager": "Safe Memory Manager",
    "DisTorchPurgeVRAMV2": "LayerUtility: Purge VRAM V2",
    "ModelPatchMemoryCleaner": "Model Patch Memory Cleaner",
    "PatchSageAttentionDM": "Patch Sage Attention DM",
}
```

**Meaning:** Register the node with ComfyUI
- `NODE_CLASS_MAPPINGS`: internal class name → class
- `NODE_DISPLAY_NAME_MAPPINGS`: UI display names

## Design notes

### 1. Dynamic logging

Callbacks log on every model run:
- Version line when SageAttention is enabled
- Baseline Flash-Attention state after cleanup
- Current attention path is visible each generation

### 2. Compatibility with upstream KJ

Matches `comfyui-kjnodes` `PathchSageAttentionKJ` behavior:
- Same SageAttention modes
- Same log format
- Same callback mechanism

### 3. Flash-Attention probing

Reads state from `model_management` when present:
- `FLASH_IS_AVAILABLE`
- `FLASH_ATTN_VERSION`
- `FLASH_ATTN_TYPE` (FA-2 / FA-3)
- `flash_attention_enabled()`

### 4. Cleanup

On `ON_CLEANUP`:
- Delete attention override
- Avoid leaking the patch into later runs
- Log baseline Flash-Attention state

### 5. Error handling

Important paths use try/except so:
- Version lookup failures still allow patching
- Flash-Attention probe failures still allow cleanup
- Fallbacks remain safe

## How to use

1. **Add the node in ComfyUI:** Memory → Patch Sage Attention DM
2. **Connect a model:** e.g. from CheckpointLoader
3. **Choose a SageAttention mode** from the dropdown
4. **Optional:** enable `allow_compile` (requires sageattn 2.2.0+)
5. **Run:** console logs appear each generation

## Example logs

### SageAttention enabled
```
Patching comfy attention to use SageAttention 2.2.0+cu121torch2.3.0
```

### SageAttention disabled (Flash-Attention available)
```
Restoring initial comfy attention
[ComfyUI] Using FA-3 (Flash-Attention 3.0.0) direct
```

### SageAttention disabled (Flash-Attention unavailable)
```
Restoring initial comfy attention
```

## Notes

1. **Patched every run:** callbacks re-apply SageAttention before each execution and clean up afterward
2. **Revert with `disabled`:** run the node again with `sage_attention = disabled`
3. **Logs every run:** console output grows because each generation logs
4. **`allow_compile`:** needs sageattn 2.2.0 or newer

## Summary

The SageAttention patch node in `ComfyUI-DistorchMemoryManager` provides KJ-compatible modes and logging via ComfyUI callbacks, applying attention overrides for each run and restoring a clean baseline afterward.
