# ComfyUI-VRAM-Manager

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/icon.png" width="128">
</p>

**ComfyUI-VRAM-Manager** (formerly ComfyUI-DistorchMemoryManager) is an independent memory management custom node for ComfyUI. Provides Distorch memory management functionality for efficient GPU/CPU memory handling. Supports purging of SeedVR2, Qwen3-VL, and Nunchaku models (FLUX/Z-Image/Qwen-Image). Includes Model Patch Memory Cleaner for ModelPatchLoader workflows.

## Overview

This custom node was created to address OOM (Out Of Memory) issues in video generation workflows like Upscaling with WAN2.2. The key point is that these OOM errors are caused by **system RAM shortage, not VRAM shortage** (can occur even on 64GB RAM systems depending on resolution and video length).

This is a completely original implementation designed specifically for Distorch memory management. Simply place it in the `custom_nodes` folder for easy installation and removal.

## Features

### Five Node Types

#### Model Patch Memory Cleaner (New in v1.2.0)

* **Description**: Memory cleaner specifically for ModelPatcher loaded model patches
* **Features**: Clears model patches loaded via ModelPatchLoader to prevent OOM during upscaling
* **Input**: Any data type (ANY) passthrough
* **Output**: Any data type (ANY) passthrough
* **Options**:
  * `clear_model_patches`: Clear model patches loaded via ModelPatchLoader (default: True)
  * `clean_gpu`: Clear GPU memory (default: True)
  * `force_gc`: Force garbage collection (default: True)
* **Use Case**: Place this node after using ModelPatchLoader (e.g., Z-Image ControlNet, QwenImage BlockWise ControlNet, SigLIP MultiFeat Proj) and before upscaling operations to prevent OOM errors. This node is designed for patch model format loaded via ModelPatchLoader, which is an exceptional format different from standard ControlNet models.
* **Technical Details**: 
  * Detects ModelPatcher instances with `additional_models` or `attachments` containing model patches
  * Safely unloads model patches from VRAM
  * Performs cleanup_models_gc() to prevent memory leaks

#### Purge VRAM V2 Compatibility (v1.10, Enhanced in v1.2.0, v2.0.0, v2.2.0)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/png/pvram2.png" width="400">
</p>

* **Description**: Restored LayerStyle's **LayerUtility: Purge VRAM V2** inside the Distortch suite (original node) with enhanced model unloading capabilities, SeedVR2 support, and Qwen3-VL/Nunchaku model purging
* **Features**: Identical UI/behavior; keeps legacy workflows working without LayerStyle. Enhanced in v1.2.0 with more aggressive model unloading and improved error handling. Enhanced in v2.0.0 with Qwen3-VL and Nunchaku model purging support. Enhanced in v2.2.0 with Nunchaku SDXL model support. Now supports SeedVR2 DiT and VAE model purging, Qwen3-VL models, and Nunchaku models (FLUX/Z-Image/Qwen-Image/SDXL).
* **Input**: Any data type (ANY) passthrough
* **Options**:  
   * `purge_cache`: Run `gc.collect()`, flush CUDA caches, call `torch.cuda.ipc_collect()`  
   * `purge_models`: Enhanced model unloading (v1.2.0):
     * Calls `cleanup_models()` to remove dead models
     * Calls `cleanup_models_gc()` for garbage collection
     * Marks all models as not currently used
     * Aggressively unloads models via `model_unload()`
     * Calls `soft_empty_cache()` if available
   * `purge_seedvr2_models`: Clear SeedVR2 DiT and VAE models from cache
     * Clears all cached DiT models from SeedVR2's GlobalModelCache
     * Clears all cached VAE models from SeedVR2's GlobalModelCache
     * Clears runner templates
     * Properly releases model memory using SeedVR2's release_model_memory()
   * `purge_qwen3vl_models`: Clear Qwen3-VL models from GPU memory (v2.0.0)
     * Searches for Qwen3-VL models in sys.modules and gc.get_objects()
     * Handles device_map="auto" case for multi-device models
     * Clears model parameters, buffers, and internal state
   * `purge_nunchaku_models`: Clear Nunchaku models (FLUX/Z-Image/Qwen-Image/SDXL) from GPU memory (v2.0.0, Enhanced in v2.2.0)
     * Supports NunchakuFluxTransformer2dModel, NunchakuZImageTransformer2DModel, NunchakuQwenImageTransformer2DModel, and NunchakuSDXLUNet2DConditionModel (v2.2.0)
     * Disables CPU offload before clearing models
     * Searches in sys.modules, ComfyUI current_loaded_models, and gc.get_objects()
     * Clears cache and temporary data attributes (v2.2.0)
     * Handles NunchakuSDXL wrapper class with diffusion_model access (v2.2.0)
* **Enhancements in v1.2.0**:
  * More aggressive model unloading with proper error handling
  * None checks and callable() checks for all method calls
  * Improved error messages and logging
  * Safe handling of models with None real_model references
  * SeedVR2 model support for clearing DiT and VAE models
* **Enhancements in v2.0.0**:
  * Qwen3-VL model purging with device_map="auto" support
  * Nunchaku model purging (FLUX/Z-Image/Qwen-Image) with CPU offload handling
  * Enhanced CUDA cache clearing for all devices
  * Comprehensive debug logging for model detection and purging
  * Fixed any() function name collision with AnyType
  * Changed display name to ComfyUI-VRAM-Manager
* **Enhancements in v2.2.0**:
  * Nunchaku SDXL model purging support (NunchakuSDXLUNet2DConditionModel)
  * NunchakuSDXL wrapper class detection and handling
  * Cache and temporary data clearing for all Nunchaku detection methods
  * More aggressive garbage collection (3x gc.collect()) and CUDA cache clearing
  * Improved VRAM release for Nunchaku SDXL models (approximately 2.5GB)
  * Preserves model structure while clearing top-level parameters only
* **Reason**: The original LayerStyle node disappeared upstream, so we duplicated it here to keep older workflows alive. Enhanced in v1.2.0 to provide better memory management. SeedVR2 support added to handle SeedVR2's independent model caching system. Enhanced in v2.0.0 to support Qwen3-VL and Nunchaku models, which are not managed by ComfyUI's standard model_management. Enhanced in v2.2.0 to support Nunchaku SDXL models, which require special handling due to their wrapper class structure and need for cache clearing.

#### Safe Memory Manager (Recommended)

* **Description**: Safe memory management node
* **Features**: Completely prevents UI corruption with safe memory management
* **Input**: Any data type (ANY)
* **Output**: Any data type (ANY)
* **Options**:  
   * `clean_gpu`: Clear GPU memory  
   * `force_gc`: Force garbage collection  
   * `reset_virtual_memory`: Reset virtual memory

#### Memory Manager (Advanced)

* **Description**: Comprehensive memory management node (for advanced users)
* **Features**: Detailed memory management with UI corruption protection
* **Input**: Any data type (ANY)
* **Output**: Any data type (ANY)
* **Options**:  
   * `clean_gpu`: Clear GPU memory  
   * `clean_cpu`: Clear CPU memory (use with caution)  
   * `force_gc`: Force garbage collection  
   * `reset_virtual_memory`: Reset virtual memory  
   * `restore_original_functions`: Restore original functions

#### Patch Sage Attention DM (New in v2.3.0)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/png/sa.png" width="400">
</p>

* **Description**: Experimental node for patching ComfyUI's attention mechanism to use SageAttention
* **Features**: Replaces ComfyUI's standard attention with SageAttention for improved memory efficiency and performance
* **Input**: Model (MODEL)
* **Output**: Model (MODEL)
* **Options**:
  * `sage_attention`: SageAttention mode selection
    * `disabled`: Disable SageAttention (restore original attention)
    * `auto`: Automatic SageAttention implementation
    * `sageattn_qk_int8_pv_fp16_cuda`: CUDA implementation (QK int8, PV FP16)
    * `sageattn_qk_int8_pv_fp16_triton`: Triton implementation (QK int8, PV FP16)
    * `sageattn_qk_int8_pv_fp8_cuda`: CUDA implementation (QK int8, PV FP8)
    * `sageattn_qk_int8_pv_fp8_cuda++`: CUDA implementation (QK int8, PV FP8, optimized)
    * `sageattn3`: SageAttention 3 implementation (Blackwell support)
    * `sageattn3_per_block_mean`: SageAttention 3 implementation (per-block mean version)
  * `allow_compile`: Allow torch.compile for SageAttention function (requires sageattn 2.2.0 or higher, default: False)
* **Use Case**: Use this node to replace ComfyUI's attention mechanism with SageAttention for better memory efficiency and performance. The node patches attention on each model execution and automatically cleans up afterward.
* **Technical Details**:
  * Uses ComfyUI's callback system (ON_PRE_RUN, ON_CLEANUP) to patch attention dynamically
  * Automatically detects SageAttention version and logs detailed information
  * Handles Flash-Attention state detection and logging when disabled
  * Compatible with ComfyUI's attention function format via wrap_attn decorator
  * Supports multiple SageAttention implementations (CUDA, Triton, SageAttention 3)

## Installation

1. Clone or download to `ComfyUI/custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager.git
```

2. Install dependencies:

```bash
cd ComfyUI-DistorchMemoryManager
pip install -r requirements.txt
```

3. Restart ComfyUI
4. Nodes will appear in the "Memory" category in the node palette

## Usage

### Basic Usage

1. Add any memory management node to your workflow
2. Connect any data to the input
3. Configure options as needed
4. Connect output to the next node

### Recommended Workflow Placement

**For ModelPatchLoader workflows**:

```
[ModelPatchLoader] → [QwenImageDiffsynthControlnet] → [Model Patch Memory Cleaner] → [Upscaling Node]
```

**For general memory management**:

```
[Previous Node] → [Safe Memory Manager] → [Next Node]
```

or

```
[Previous Node] → [Memory Manager] → [Next Node]
```

### Recommended Settings

**For ModelPatchLoader workflows (patch model format)**:

* Use **Model Patch Memory Cleaner**
* `clear_model_patches: True`
* `clean_gpu: True`
* `force_gc: True`
* **Place after**: ModelPatchLoader usage, before upscaling operations
* **Note**: This is for patch model format loaded via ModelPatchLoader (e.g., Z-Image ControlNet, QwenImage BlockWise ControlNet, SigLIP MultiFeat Proj), which is an exceptional format different from standard ControlNet models.

**For video generation (WAN2.2, etc.)**:

* Use **Safe Memory Manager**
* `clean_gpu: True`
* `force_gc: True`
* `reset_virtual_memory: True`

**For maximum memory release**:

* Use **Memory Manager**
* `clean_cpu: True` (Warning: possible UI corruption)

## Troubleshooting

### Out of Memory Errors

**Solution**:

1. For ModelPatchLoader workflows: Use **Model Patch Memory Cleaner** after ControlNet usage
2. For general workflows: Use **Safe Memory Manager** or **Memory Manager**
3. Enable `clean_gpu` and `reset_virtual_memory`
4. Enable `force_gc` if needed

### OOM During Upscaling After ModelPatchLoader Usage

**Solution**:

1. Add **Model Patch Memory Cleaner** node after QwenImageDiffsynthControlnet (when using ModelPatchLoader)
2. Enable `clear_model_patches: True`
3. Enable `clean_gpu: True`
4. Enable `force_gc: True`
5. **Note**: This applies to patch model format loaded via ModelPatchLoader, not standard ControlNet models

### UI Corruption

**Solution**:

1. Use **Safe Memory Manager** (recommended) or **Model Patch Memory Cleaner**
2. Keep `clean_cpu` disabled (if using Memory Manager)
3. Enable only essential options

### OOM with Qwen3-VL Models

**Solution**:

1. Use **DisTorchPurgeVRAMV2** node
2. Enable `purge_qwen3vl_models: True` to clear Qwen3-VL models from GPU memory
3. Enable `purge_cache: True` and `purge_models: True` for comprehensive cleanup
4. The node handles device_map="auto" case for multi-device models automatically

### OOM with Nunchaku Models (FLUX/Z-Image/Qwen-Image/SDXL)

**Solution**:

1. Use **DisTorchPurgeVRAMV2** node
2. Enable `purge_nunchaku_models: True` to clear Nunchaku models from GPU memory
3. The node automatically disables CPU offload before clearing models
4. Enable `purge_cache: True` and `purge_models: True` for comprehensive cleanup
5. Works with NunchakuFluxTransformer2dModel, NunchakuZImageTransformer2DModel, NunchakuQwenImageTransformer2DModel, and NunchakuSDXLUNet2DConditionModel (v2.2.0)
6. For Nunchaku SDXL models, the node now clears cache and temporary data attributes, releasing approximately 2.5GB of VRAM (v2.2.0)

## Technical Details

### Implemented Features

* GPU memory clearing (`torch.cuda.empty_cache()`)
* GPU synchronization (`torch.cuda.synchronize()`)
* CPU memory clearing (`gc.collect()`)
* Virtual memory reset (`comfy.model_management.free_memory()`)
* Model patch detection and unloading (v1.2.0)
  * Detects ModelPatcher instances with `additional_models` or `attachments` containing patch model format
  * Safely unloads model patches via `model_unload()`
  * Removes from `current_loaded_models` list
  * Performs `cleanup_models_gc()` to prevent memory leaks
  * Handles exceptional patch model format loaded via ModelPatchLoader (different from standard ControlNet)
* Qwen3-VL model purging (v1.4.0)
  * Searches for Qwen3-VL models in sys.modules and gc.get_objects()
  * Handles device_map="auto" case for multi-device models
  * Clears model parameters, buffers, and internal state
  * Supports hf_device_map processing for distributed models
* Nunchaku model purging (v1.4.0, Enhanced in v2.2.0)
  * Supports NunchakuFluxTransformer2dModel, NunchakuZImageTransformer2DModel, NunchakuQwenImageTransformer2DModel, and NunchakuSDXLUNet2DConditionModel (v2.2.0)
  * Automatically disables CPU offload before clearing models
  * Searches in sys.modules, ComfyUI current_loaded_models, and gc.get_objects()
  * Handles nested model structures (ModelPatcher, ComfyFluxWrapper)
  * Clears offload_manager to release offloaded memory
  * NunchakuSDXL wrapper class detection and diffusion_model access (v2.2.0)
  * Cache and temporary data clearing (_cache, _state_dict_cache, _non_persistent_buffers_set) (v2.2.0)
  * More aggressive garbage collection and CUDA cache clearing for better VRAM release (v2.2.0)

### Safety Features

* Safe implementation to prevent UI corruption
* Error handling with exception processing
* Gradual memory clearing
* None checks and callable() checks for all method calls (v1.2.0)
* Robust error handling in cleanup_models() and is_dead() methods

## Additional Tips

* Expanding paging file size can also reduce OOM occurrences during upscaling
* Note: For OOM during video generation inference (where VRAM is critical), paging file expansion won't help
* For ModelPatchLoader workflows: Always use Model Patch Memory Cleaner before upscaling to prevent OOM. Note that patch model format loaded via ModelPatchLoader is an exceptional format different from standard ControlNet models.
* For Qwen3-VL workflows: Use DisTorchPurgeVRAMV2 with `purge_qwen3vl_models: True` after Qwen3-VL model usage to prevent OOM. The node automatically handles device_map="auto" case for models distributed across multiple devices.
* For Nunchaku workflows (FLUX/Z-Image/Qwen-Image/SDXL): Use DisTorchPurgeVRAMV2 with `purge_nunchaku_models: True` after Nunchaku model usage to prevent OOM. The node automatically disables CPU offload and clears models from all detection locations (sys.modules, ComfyUI model management, and gc.get_objects()). For Nunchaku SDXL models (v2.2.0), the node now includes cache clearing functionality that can release approximately 2.5GB of VRAM.
* For SageAttention workflows (v2.3.0): Use Patch Sage Attention DM node to replace ComfyUI's attention mechanism with SageAttention for improved memory efficiency and performance. The node supports multiple SageAttention implementations and automatically patches attention on each model execution. To disable SageAttention, run the node again with `sage_attention` set to `disabled`.

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

Bug reports and feature requests are welcome on the GitHub Issues page.

## Release History

* **v2.3.0** – Added Patch Sage Attention DM node for patching ComfyUI's attention mechanism to use SageAttention. Supports multiple SageAttention implementations (auto, CUDA, Triton, SageAttention 3) with dynamic patching via ComfyUI's callback system. Added independent version detection for Flash-Attention and SageAttention (completely independent from model_management module). Flash-Attention auto-load feature when disabled (no CLI options required). Automatically detects and logs SageAttention version with CUDA/PyTorch information, and Flash-Attention version with FA-2/FA-3 type detection. Version information is logged on every generation. Compatible with ComfyUI's attention function format. See [Release Notes v2.3.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.0) for details.
* **v2.2.0** – Added Patch Sage Attention DM node for patching ComfyUI's attention mechanism to use SageAttention. Supports multiple SageAttention implementations (auto, CUDA, Triton, SageAttention 3) with dynamic patching via ComfyUI's callback system. Automatically detects and logs SageAttention version with CUDA/PyTorch information. Compatible with ComfyUI's attention function format. See [Release Notes v2.2.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.2.0) for details.
* **v2.0.0** – Added Qwen3-VL and Nunchaku model purging support to DisTorchPurgeVRAMV2 node. Qwen3-VL models can now be purged from GPU memory with device_map="auto" support. Nunchaku models (FLUX/Z-Image/Qwen-Image) can be purged with CPU offload handling. Enhanced CUDA cache clearing to support all devices. Fixed any() function name collision with AnyType. Added comprehensive debug logging. Changed display name to ComfyUI-VRAM-Manager. See [Release Notes v2.0.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.0.0) for details.
* **v1.3.1** – Improved SeedVR2 cache detection and messaging. Removed duplicate messages. Clarified that cache_model=False (default) means models are never cached in GlobalModelCache. Added detailed debug information for cache state. See [Release Notes v1.3.1](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v1.3.1) for details.
* **v1.3.0** – Added SeedVR2 model purging support to DisTorchPurgeVRAMV2 node. Fixed 'NoneType' object is not callable errors in cleanup_models(). Fixed CPU device error in virtual memory reset. Improved path detection for SeedVR2 custom node to work across different user environments. See [Release Notes v1.3.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v1.3.0) for details.
* **v1.2.0** – Added Model Patch Memory Cleaner node for ModelPatchLoader model patches (patch model format). Prevents OOM during upscaling after ModelPatchLoader usage. Handles exceptional patch model format different from standard ControlNet models. Enhanced DisTorchPurgeVRAMV2 with more aggressive model unloading, improved error handling, and safe None checks. Added SeedVR2 support to purge DiT and VAE models from cache. Fixed CPU device error in virtual memory reset. Improved error handling in cleanup_models() and is_dead() methods in ComfyUI core. See [Release Notes v1.2.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v1.2.0) for details.
* v1.10.1 – Hotfix ensuring DisTorch Purge VRAM V2 node ships inside the package.
* v1.10 – Added the LayerUtility: Purge VRAM V2 compatibility node within DisTorch Memory Manager.
* v1.1.0 – Added ANY type I/O support, simplified node names, moved category to "Memory".
* v1.0.0 – Initial release with core memory management features.

## About

ComfyUI-VRAM-Manager (formerly ComfyUI-DistorchMemoryManager) - Independent memory management custom node for ComfyUI with Distorch support
