# Release Notes v2.3.0: Flash-Attention Auto-Load and Version Detection Features

## Overview

Added independent Flash-Attention auto-load and version detection features. These features are completely independent from ComfyUI's model_management module, eliminating the need to manually modify model_management.py after ComfyUI updates.

## Key Features

### Independent Version Detection

- **Flash-Attention version detection**: Completely independent version detection that doesn't rely on model_management module
- **FA-2/FA-3 type detection**: Automatically detects Flash-Attention 2 or 3 based on version number
- **SageAttention version detection**: Independent SageAttention version detection with CUDA/PyTorch information
- **Version information logging**: Version information is logged on every generation

### Flash-Attention Auto-Load

- **No CLI options required**: Flash-Attention is automatically loaded when SageAttention is set to `disabled` without requiring `--use-flash-attention` CLI option
- **Direct loading**: Uses `optimized_attention_override` to directly load Flash-Attention
- **Package import detection**: Detects Flash-Attention availability based on package import capability, not CLI options

### Dynamic Patching and Logging

- **ON_PRE_RUN callback**: Loads Flash-Attention when disabled, applies SageAttention when enabled, and outputs version logs
- **ON_CLEANUP callback**: Always outputs Flash-Attention logs even when SageAttention was active (ComfyUI resets to optimal kernel on cleanup)
- **Per-generation logging**: Version information is logged on every generation

## Technical Details

### Core Mechanism

The implementation uses ON_PRE_RUN and ON_CLEANUP callbacks for dynamic patching and logging:

1. **Generation start (ON_PRE_RUN callback)**:
   - If SageAttention is enabled: Get SageAttention function, set `optimized_attention_override` to apply patch, output SageAttention version log
   - If SageAttention is disabled: If Flash-Attention package is available, set `comfy_attention.attention_flash` to `optimized_attention_override` to directly load Flash-Attention, output Flash-Attention version log

2. **Generation end (ON_CLEANUP callback)**:
   - Delete `optimized_attention_override` to execute ComfyUI's kernel reset
   - ComfyUI automatically selects optimal kernel (Flash-Attention) as initial state
   - Output Flash-Attention version log at this point (even when SageAttention was active)

### Flash-Attention Version Detection Function

```python
def get_flash_attention_info():
    """
    Get Flash-Attention version and type information.
    Returns: (is_available, version, type)
    """
```

**Process flow**:
1. Check package import capability using `import flash_attn`
2. Attempt to get version string from `flash_attn.__version__`
3. Fallback to `importlib.metadata.version("flash-attn")` if __version__ doesn't exist
4. Split version string and get major version number
5. Determine FA-2 or FA-3: `major_version >= 3` → "FA-3", otherwise → "FA-2"
6. Return tuple: `(is_available, version, type)`

**Important point**: Determines based on package import capability only, regardless of `args.use_flash_attention` value. This allows accurate detection of actual Flash-Attention availability.

### SageAttention Version Detection Function

```python
def get_sage_attention_info():
    """
    Get SageAttention version information.
    Returns: (version, cuda_version, torch_version)
    """
```

**Process flow**:
1. Import sageattention package
2. Attempt to get version from `sageattention.__version__`
3. Fallback to `importlib.metadata.version("sageattention")` if __version__ doesn't exist
4. Get CUDA version from `torch.version.cuda`
5. Get PyTorch version from `torch.version.__version__`
6. Return tuple: `(version, cuda_version, torch_version)`

### Flash-Attention Auto-Load Mechanism

When SageAttention is set to `disabled`, Flash-Attention is loaded using `optimized_attention_override`:

```python
def attention_override_flash(func, *args, **kwargs):
    return comfy_attention.attention_flash(*args, **kwargs)
model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_flash
```

This bypasses ComfyUI's default attention selection logic and directly uses the `attention_flash` function.

## Important Points

### 1. Complete Independence

Completely independent from `model_management` module. This allows version information to always be retrieved regardless of ComfyUI updates.

### 2. No CLI Options Required

Flash-Attention can be loaded using only the node's `disabled` setting, without requiring `--use-flash-attention` CLI option. Determines based on package import capability only.

### 3. Per-Generation Logging

- **ON_PRE_RUN (before generation)**: Outputs SageAttention log if enabled, Flash-Attention log if disabled
- **ON_CLEANUP (after generation)**: Always checks and outputs Flash-Attention state (even when SageAttention is enabled, ComfyUI's kernel reset causes Flash-Attention to be selected as initial state)

### 4. Error Handling

All critical operations are wrapped in try-except blocks to ensure safe operation even when errors occur.

### 5. Kernel Reset Understanding

ComfyUI performs kernel reset on every generation, returning to initial state (optimal kernel). This initial state automatically selects Flash-Attention when available. Therefore, even when SageAttention is enabled, Flash-Attention logs are output during cleanup after generation ends.

## Benefits

- **No manual modifications needed**: Eliminates need to manually modify model_management.py after ComfyUI updates
- **Always available version info**: Version information can always be retrieved regardless of ComfyUI updates
- **Convenient Flash-Attention loading**: Flash-Attention can be loaded without CLI options
- **Better visibility**: Version information is logged on every generation, providing better visibility into attention mechanism state

## Files Modified

- `nodes/sa.py`: Added independent version detection functions and Flash-Attention auto-load functionality

## Summary

This implementation adds completely independent version detection and Flash-Attention auto-load features to the ComfyUI-DistorchMemoryManager node. By completely eliminating dependencies on model_management, version information for Flash-Attention and SageAttention can always be retrieved regardless of ComfyUI updates. Additionally, Flash-Attention (FA-2/FA-3) is automatically loaded when disabled, operating without CLI options. This eliminates the hassle of manually modifying model_management.py after each ComfyUI update.
