# ComfyUI-VRAM-Manager

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="zhmd/README.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/icon.png" width="128">
</p>

**ComfyUI-VRAM-Manager** (formerly ComfyUI-DistorchMemoryManager) is an independent memory management custom node for ComfyUI. Provides Distorch memory management functionality for efficient GPU/CPU memory handling. Supports purging of SeedVR2, Qwen3-VL, Nunchaku models (FLUX/Z-Image/Qwen-Image), HSWQ, and Ollama server VRAM. Includes Model Patch Memory Cleaner for ModelPatchLoader workflows. Auto-detects non-PyTorch VRAM usage via NVML to prevent OOM errors in multi-process environments.

## Overview

This custom node was created to address OOM (Out Of Memory) issues in video generation workflows like Upscaling with WAN2.2. The key point is that these OOM errors are caused by **system RAM shortage, not VRAM shortage** (can occur even on 64GB RAM systems depending on resolution and video length).

This is a completely original implementation designed specifically for Distorch memory management. Simply place it in the `custom_nodes` folder for easy installation and removal.

## Features

### General Manage VRAM (Startup Automatic Patch - New in v2.4.0)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/png/generalvram.png" width="600">
</p>

* **Description**: A fully automated, GPU-wide VRAM headroom optimizer that runs seamlessly at ComfyUI startup. It automatically detects system-wide non-PyTorch VRAM usage (browsers, Discord, OBS, desktop window managers) via NVML and dynamically configures ComfyUI's VRAM management on load.

##### The Core Limitation of Standard ComfyUI
* **PyTorch-Only Blind Spot**: By default, **ComfyUI can only recognize active VRAM allocations made within the PyTorch framework itself.**
* **The Problem**: Standard ComfyUI is completely blind to physical VRAM consumed by external, non-PyTorch applications (such as web browsers, Discord, OBS, or desktop window managers). Because it cannot detect this external overhead, ComfyUI often overestimates available VRAM, resulting in sudden Out-Of-Memory (OOM) crashes when attempting to load heavy models.
* **The Solution**: This startup patch uses NVML to query the absolute physical VRAM usage of the GPU, calculating the exact difference between system-wide consumption and PyTorch's active memory. It then overrides ComfyUI's headroom limit (`General Manage VRAM`) with this real-world value to guarantee multi-process memory safety.

##### Key Benefits & Features
* **Zero Node Setup**: Operates entirely in the background at startup. No node placement in workflows, manual connections, or toggle switches are required.
* **NVML-Powered Accuracy**: Utilizes the `pynvml` (NVIDIA Management Library) API to query real-time physical GPU memory status, ensuring perfect accuracy.
* **Auto-updated dependency**: On every ComfyUI load (and via ComfyUI-Manager `install.py`), `nvidia-ml-py` is upgraded with `pip install -U` so the NVML binding stays current without a manual upgrade.
* **Robust Multi-Process OOM Prevention**: Dynamically patches ComfyUI's internal VRAM headroom buffer on load by calculating the exact difference between system-wide physical GPU usage and PyTorch allocations.
* **Optimized for iGPU/dGPU Multi-GPU Setups**:
  * Perfect for setups that offload Windows desktop rendering and browser acceleration to the CPU's integrated graphics (e.g., Ryzen 9 7900 built-in Radeon iGPU) and reserve the RTX GPU exclusively for CUDA workloads.
  * The startup patch detects extremely low non-PyTorch overhead (e.g., `0.02 GB`), automatically shrinking the reserved chunk down from the default `0.68 GB` to `0.02 GB` to maximize ComfyUI's available memory.

---

### Four Node Types

#### Model Patch Memory Cleaner (New in v1.2.0)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/png/mpatch.png" width="400">
</p>

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

#### General Purge VRAM V2 (v1.10, Enhanced in v1.2.0, v2.0.0, v2.2.0, v2.4.1, v2.4.2, v2.4.3)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/png/pvram2.png" width="400">
</p>

* **Description**: Distortch suite node **General Purge VRAM V2** (formerly LayerStyle `LayerUtility: Purge VRAM V2`; class id `DisTorchPurgeVRAMV2`) with enhanced model unloading, SeedVR2 / Qwen3-VL / Nunchaku purging, (v2.4.1) an **`HSWQ`** toggle for full HSWQ VRAM purge, (v2.4.2) an **`Ollama`** toggle below **`HSWQ`** for zero-residual Ollama server VRAM purge, and (v2.4.3) HSWQ Method **2c** **NVFP4** runtime pool / CUDA graph clear so a second ConvRot NVFP4 generation after purge does not hit `quantize_nvfp4` / `PyCapsule` / `pooled TC path failed`
* **Features**: Same UI/behavior lineage as the LayerStyle original; keeps legacy workflows via class id `DisTorchPurgeVRAMV2`. Enhanced in v1.2.0 with more aggressive model unloading and improved error handling. Enhanced in v2.0.0 with Qwen3-VL and Nunchaku model purging support. Enhanced in v2.2.0 with Nunchaku SDXL model support. Enhanced in v2.4.1 with dedicated **`HSWQ`** purge pipeline (PinCache drain, PromptExecutor/SEGS in-place clear, HostUnregister, `comfy_kitchen` CUDA workspace reset). Enhanced in v2.4.2 with **`Ollama`** purge for **comfyui-ollama** and **comfyui-ollama-describer** (including describer's default `keep_model_alive=-1`). Enhanced in v2.4.3 with HSWQ Method **2c** `sys.modules` scan for `nvfp4_runtime` and `clear_nvfp4_runtime_pools()` (in addition to kitchen workspace reset); prefers `nodes/purge_vram.py`; log prefix `HSWQ INT8/NVFP4:`. Supports SeedVR2 DiT/VAE, Qwen3-VL, Nunchaku (FLUX/Z-Image/Qwen-Image/SDXL), HSWQ (including NVFP4), and Ollama server unload.
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
   * `HSWQ`: Purge HSWQ residual GPU (and related host) memory — whole HSWQ path, not INT8-only (v2.4.1; NVFP4 Method **2c** in v2.4.3)
     * Force-imports and drains HSWQ PinCache; clears PromptExecutor / SEGS caches in-place (does not call `reset()` mid-prompt)
     * Releases PINNED_MEMORY via HostUnregister where applicable
     * After core cleanup, resets `comfy_kitchen` CUDA workspace / empty-tensor caches (Method **2c**) so reload (including INT8 GEMM) still works after purge
     * (v2.4.3) After kitchen reset, scans `sys.modules` for `nvfp4_runtime` and calls `clear_nvfp4_runtime_pools()` to clear HSWQ **NVFP4** runtime pools / CUDA graphs
     * UI label is **`HSWQ`**; legacy workflow kwargs `"HSWQ INT8"` remain accepted; log prefix `HSWQ INT8/NVFP4:`
   * `Ollama`: Purge Ollama server VRAM loaded by **comfyui-ollama** and **comfyui-ollama-describer** (v2.4.2)
     * Toggle appears **directly below `HSWQ`** in the node UI (see screenshot above)
     * Targets describer's default `keep_model_alive=-1` (model stays loaded until explicitly unloaded)
     * Harvests `api_host` / `url` from both custom-node packs; loops `GET /api/ps` until empty
     * Sends `/api/generate` and `/api/chat` with `keep_alive=0`; runs `ollama stop`; Client API fallback when available
     * Clears in-process `CHAT_SESSIONS` / `saved_context`; deletes `saved_context/` files; final `/api/ps` verification
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
* **Enhancements in v2.4.1**:
  * Dedicated **`HSWQ`** toggle for full HSWQ VRAM purge
  * PinCache force-import/drain, PromptExecutor/SEGS in-place clear, HostUnregister path
  * Method **2c** `comfy_kitchen` CUDA workspace / empty-tensor cache reset after core purge
  * Fallback `nodes/purge_vram.py` kept in sync with the root node
* **Enhancements in v2.4.2**:
  * New **`Ollama`** toggle below **`HSWQ`** on DisTorchPurgeVRAMV2
  * Zero-residual Ollama purge for **comfyui-ollama** and **comfyui-ollama-describer**
  * `/api/ps` loop until empty, `keep_alive=0` on generate/chat, `ollama stop`, Client fallback
  * Clears `CHAT_SESSIONS`, `saved_context`, and on-disk `saved_context/` files
  * Fallback `nodes/purge_vram.py` kept in sync with the root node
* **Enhancements in v2.4.3**:
  * HSWQ Method **2c** clears HSWQ **NVFP4** runtime pools / CUDA graphs via `sys.modules` scan + `clear_nvfp4_runtime_pools()` (in addition to `comfy_kitchen` workspace reset)
  * Avoids second ConvRot NVFP4 generation failures after purge (`quantize_nvfp4` / `PyCapsule` / `pooled TC path failed`)
  * Prefers importing `DisTorchPurgeVRAMV2` from `nodes/purge_vram.py`; log prefix `HSWQ INT8/NVFP4:`
* **Reason**: The original LayerStyle node disappeared upstream, so we duplicated it here to keep older workflows alive. Enhanced in v1.2.0 to provide better memory management. SeedVR2 support added to handle SeedVR2's independent model caching system. Enhanced in v2.0.0 to support Qwen3-VL and Nunchaku models, which are not managed by ComfyUI's standard model_management. Enhanced in v2.2.0 to support Nunchaku SDXL models, which require special handling due to their wrapper class structure and need for cache clearing. Enhanced in v2.4.1 because HSWQ leftovers are not fully recovered by generic ComfyUI unload or DistTorch general purge. Enhanced in v2.4.2 because Ollama models loaded via comfyui-ollama / comfyui-ollama-describer (especially with `keep_model_alive=-1`) are not released by standard ComfyUI or HSWQ purge alone. Enhanced in v2.4.3 because HSWQ **NVFP4** runtime pools / CUDA graphs survive kitchen-only Method **2c** and break the next ConvRot NVFP4 generation after purge.

#### Memory Manager (Advanced)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/png/mmanager.png" width="400">
</p>

* **Description**: Comprehensive memory management node (for advanced users)
* **Features**: Detailed memory management with UI corruption protection and general VRAM management
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

* Use **Memory Manager**
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
2. For general workflows: Use **Memory Manager**
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

1. Use **Model Patch Memory Cleaner** or **Memory Manager**
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
* **For multi-process environments (v2.4.0)**: Non-PyTorch VRAM usage (browsers, Discord, OBS, etc.) is now **automatically detected via NVML at startup** and dynamically applied to ComfyUI's memory management (General Manage VRAM). This prevents OOM errors that occur when ComfyUI overestimates available VRAM, ensuring system-wide memory safety without requiring any manual node configuration.

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

Bug reports and feature requests are welcome on the GitHub Issues page.

## Release History

See [CHANGELOG.md](changelog/changelog.md) for detailed release history.

## About

ComfyUI-VRAM-Manager (formerly ComfyUI-DistorchMemoryManager) - Independent memory management custom node for ComfyUI with Distorch support
