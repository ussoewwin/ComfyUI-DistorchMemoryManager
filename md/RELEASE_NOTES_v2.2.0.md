# Release Notes v2.2.0: SageAttention Patch Feature

## Overview

Added Patch Sage Attention DM node for patching ComfyUI's attention mechanism to use SageAttention. This feature allows replacing ComfyUI's standard attention mechanism with SageAttention, providing improved memory efficiency and performance.

## Key Features

### Patch Sage Attention DM Node

- **Dynamic patching**: Uses ComfyUI's callback system (ON_PRE_RUN and ON_CLEANUP) for dynamic attention patching
- **Multiple SageAttention modes**: Supports auto, CUDA, Triton, and SageAttention 3 implementations
- **Version detection**: Automatically detects and logs SageAttention version with CUDA/PyTorch information
- **Flash-Attention state logging**: Logs Flash-Attention state when SageAttention is disabled
- **ComfyUI compatibility**: Compatible with ComfyUI's attention function format using wrap_attn decorator

### Supported SageAttention Modes

- `disabled`: Disable SageAttention and restore original attention mechanism
- `auto`: Automatic SageAttention implementation selection
- `sageattn_qk_int8_pv_fp16_cuda`: CUDA implementation (QK int8, PV FP16)
- `sageattn_qk_int8_pv_fp16_triton`: Triton implementation (QK int8, PV FP16)
- `sageattn_qk_int8_pv_fp8_cuda`: CUDA implementation (QK int8, PV FP8)
- `sageattn_qk_int8_pv_fp8_cuda++`: CUDA implementation (QK int8, PV FP8, optimized)
- `sageattn3`: SageAttention 3 implementation (Blackwell support)
- `sageattn3_per_block_mean`: SageAttention 3 implementation (per-block mean version)

### Version Detection and Logging

- **SageAttention version**: Detects version using __version__ attribute or importlib.metadata
- **CUDA/PyTorch versions**: Includes CUDA and PyTorch version information in logs
- **Flash-Attention state**: Checks and logs Flash-Attention state from model_management module
- **Dynamic logging**: Logs are output on every model execution via callbacks

## Technical Details

### Implementation Architecture

The implementation uses ComfyUI's callback system to dynamically patch attention:
- **ON_PRE_RUN callback**: Patches attention before each model execution
- **ON_CLEANUP callback**: Cleans up and logs Flash-Attention state after each execution

### Attention Function Wrapping

The implementation uses `wrap_attn` decorator to wrap SageAttention functions in ComfyUI's attention format:
- Converts tensor shapes from ComfyUI format (q, k, v, heads) to SageAttention format
- Handles FP32 to FP16 conversion (SageAttention primarily operates on FP16)
- Adjusts mask dimensions (adds batch and heads dimensions)
- Restores output to original data type and shape

### torch.compile Control

- **allow_compile option**: Optional boolean parameter to enable torch.compile
- **Default disabled**: torch.compile is disabled by default to avoid compilation overhead
- **Requirement**: torch.compile requires sageattn 2.2.0 or higher

### Error Handling

All critical operations are wrapped in try-except blocks:
- Version detection failures are handled gracefully
- Flash-Attention state checks are safe
- Fallback messages are provided when information cannot be retrieved

## Usage

1. **Add node in ComfyUI**: Add "Patch Sage Attention DM" node from "Memory" category
2. **Connect model**: Connect model from CheckpointLoader or similar
3. **Select SageAttention mode**: Choose desired mode from dropdown
4. **Configure options**: Enable `allow_compile` if desired (requires sageattn 2.2.0+)
5. **Execute**: Model execution will output logs to console

## Log Output Examples

### When SageAttention is Enabled
```
Patching comfy attention to use SageAttention 2.2.0+cu121torch2.3.0
```

### When SageAttention is Disabled (Flash-Attention Enabled)
```
Restoring initial comfy attention
[ComfyUI] Using FA-3 (Flash-Attention 3.0.0) direct
```

### When SageAttention is Disabled (Flash-Attention Disabled)
```
Restoring initial comfy attention
```

## Important Notes

1. **Patching occurs on every execution**: Due to callback usage, SageAttention is applied before each execution and cleaned up after
2. **Use 'disabled' to restore**: To disable SageAttention, run the node again with `sage_attention` set to `disabled`
3. **Logs output on every execution**: Console output increases as logs are output on every model execution
4. **allow_compile requirement**: torch.compile requires sageattn 2.2.0 or higher

## Files Modified

- `nodes/sa.py`: Added PatchSageAttentionDM class with SageAttention patching functionality

## Compatibility

This implementation replicates the functionality of `comfyui-kjnodes`' `PatchSageAttentionKJ` node:
- Supports the same SageAttention modes
- Uses the same log format
- Uses the same callback mechanism
