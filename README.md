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

**ComfyUI-VRAM-Manager** (formerly ComfyUI-DistorchMemoryManager) is an independent memory management custom node for ComfyUI. Provides Distorch memory management functionality for efficient GPU/CPU memory handling. Supports purging of SeedVR2, Qwen3-VL, Nunchaku models (FLUX/Z-Image/Qwen-Image), HSWQ, and Ollama server VRAM. Features integrated Model Patch Memory Cleaner functionality inside General Purge VRAM V2 for ModelPatchLoader workflows. Auto-detects non-PyTorch VRAM usage via NVML to prevent OOM errors in multi-process environments.

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

### Three Node Types

#### General Purge VRAM V2

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-DistorchMemoryManager/main/png/pvram2.png" width="400">
</p>

* **Description**: Distortch suite node **General Purge VRAM V2** (formerly LayerStyle `LayerUtility: Purge VRAM V2`; class id `DisTorchPurgeVRAMV2`). A unified, aggressive memory purging node designed to eliminate GPU/CPU residual allocations and deep caches that standard ComfyUI garbage collection cannot reach. Fully supports standard ComfyUI models, ModelPatchLoader patches, SeedVR2, Qwen3-VL, Nunchaku (FLUX / Z-Image / Qwen-Image / SDXL), full HSWQ execution pipelines (including INT8 & NVFP4 runtime pools/graphs), and external Ollama server processes.
* **Features**:
  * Preserves complete UI and workflow compatibility with legacy LayerStyle pipelines via class id `DisTorchPurgeVRAMV2`.
  * Aggressive model unloading with robust error handling, callable verification, and safe handling of detached/None `real_model` references.
  * Deep model patch cleanup directly integrated under `clear_model_patches` with instant CUDA cache flush, synchronization, and GC.
  * Comprehensive memory evacuation across specialized third-party architectures (SeedVR2, Qwen3-VL, Nunchaku, HSWQ, Ollama).
* **Input**: Any data type (`ANY`) passthrough
* **Output**: Any data type (`ANY`) passthrough
* **Options**:
  * `purge_cache`: Run `gc.collect()`, flush CUDA caches across all active devices (`torch.cuda.empty_cache()`), and call `torch.cuda.ipc_collect()`.
  * `purge_models`: Deep model unloading pipeline:
    * Calls `cleanup_models()` to remove dead models
    * Calls `cleanup_models_gc()` for garbage collection
    * Marks all models as not currently used
    * Aggressively unloads models via `model_unload()`
    * Calls `soft_empty_cache()` if available
    * Includes robust error handling, `None` checks, and `callable()` validation for all method calls (safely handling models with `None` `real_model` references)
  * `clear_model_patches`: Clear model patches loaded via ModelPatchLoader (default: `True`; fully integrates the standalone Model Patch Memory Cleaner):
    * Clears model patches loaded via ModelPatchLoader (e.g., Z-Image ControlNet, QwenImage BlockWise ControlNet, SigLIP MultiFeat Proj) to prevent OOM during upscaling
    * Detects `ModelPatcher` instances with `additional_models` or `attachments` containing model patches
    * Safely unloads model patches from VRAM via `model_unload()` and removes them from `current_loaded_models`
    * Calls `cleanup_models_gc()`, followed immediately by `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`, and `gc.collect()` for instantaneous physical VRAM recovery even when `purge_models=False`
  * `purge_seedvr2_models`: Clear SeedVR2 DiT and VAE models from cache:
    * Clears all cached DiT models from SeedVR2's `GlobalModelCache`
    * Clears all cached VAE models from SeedVR2's `GlobalModelCache`
    * Clears runner templates
    * Properly releases model memory using SeedVR2's `release_model_memory()`
  * `purge_qwen3vl_models`: Clear Qwen3-VL models from GPU memory:
    * Searches for Qwen3-VL models in `sys.modules` and `gc.get_objects()`
    * Handles `device_map="auto"` case for multi-device models
    * Clears model parameters, buffers, and internal execution state
  * `purge_nunchaku_models`: Clear Nunchaku models (FLUX / Z-Image / Qwen-Image / SDXL) from GPU memory:
    * Supports `NunchakuFluxTransformer2dModel`, `NunchakuZImageTransformer2DModel`, `NunchakuQwenImageTransformer2DModel`, and `NunchakuSDXLUNet2DConditionModel`
    * Disables CPU offload before clearing models to prevent stalls
    * Searches in `sys.modules`, ComfyUI `current_loaded_models`, and `gc.get_objects()`
    * Clears cache and temporary data attributes across all detection methods
    * Handles `NunchakuSDXL` wrapper class with direct `diffusion_model` access; preserves model structure while clearing top-level parameters (~2.5GB VRAM recovered)
    * Performs aggressive garbage collection (3x `gc.collect()`) and CUDA cache clearing
  * `HSWQ`: Purge HSWQ residual GPU (and related host) memory — whole HSWQ path, not INT8-only:
    * Force-imports and drains HSWQ `PinCache`; clears `PromptExecutor` / `SEGS` caches in-place (does not call `reset()` mid-prompt)
    * Releases `PINNED_MEMORY` via `HostUnregister` where applicable
    * Resets `comfy_kitchen` CUDA workspace / empty-tensor caches (Method **2c**) after core cleanup so reload (including INT8 GEMM) still works after purge
    * Scans `sys.modules` for `nvfp4_runtime` and calls `clear_nvfp4_runtime_pools()` (Method **2c**) to clear HSWQ **NVFP4** runtime pools / CUDA graphs, preventing downstream ConvRot NVFP4 generation failures (`quantize_nvfp4` / `PyCapsule` / `pooled TC path failed`)
    * UI label is **`HSWQ`**; legacy workflow kwargs `"HSWQ INT8"` remain accepted; log prefix `HSWQ INT8/NVFP4:`
  * `Ollama`: Purge Ollama server VRAM loaded by **comfyui-ollama** and **comfyui-ollama-describer**:
    * Toggle appears directly below `HSWQ` in the node UI
    * Targets describer's default `keep_model_alive=-1` (model stays loaded until explicitly unloaded)
    * Harvests `api_host` / `url` from both custom-node packs; loops `GET /api/ps` until empty
    * Sends `/api/generate` and `/api/chat` with `keep_alive=0`; runs `ollama stop`; Client API fallback when available
    * Clears in-process `CHAT_SESSIONS` / `saved_context`; deletes `saved_context/` files; final `/api/ps` verification
* **Architecture & Design Rationale**:
  * **Workflow Continuity**: Preserves full compatibility for pipelines built on the discontinued LayerStyle node via class id `DisTorchPurgeVRAMV2`.
  * **Beyond Standard ComfyUI Unload**: Standard `model_management.unload_all_models()` only tracks models registered within ComfyUI's internal model registry. It cannot reach external frameworks, custom runner caches, or third-party background services.
  * **Unified Multi-Engine Purge**: Consolidates memory evacuation for SeedVR2's independent model cache, Qwen3-VL's multi-device auto-mapping, Nunchaku's transformer wrapper structures, HSWQ's pinned caches / kitchen workspaces / NVFP4 CUDA graphs, ModelPatchLoader auxiliary attachments, and external Ollama server instances into a single, comprehensive node without requiring multiple separate cleaner utilities.

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
    * `spargeattn`: **SpargeAttn-hswq** (v2.4.7): two-stage block-sparse attention based on SageAttention2++ quantized kernels via the `spas_sage_hswq_attn` package (Owner's fork, coexists with official `sageattention`). **Prebuilt Windows wheels are published on Hugging Face: [Sage-Attention-and-Sparge-Attention-HSWQ](https://huggingface.co/ussoewwin/Sage-Attention-and-Sparge-Attention-HSWQ)** — install the matching wheel (e.g. `pip install spas_sage_hswq_attn-1.0.0+cu132torch2.14.0cxx11abitrue-cp313-cp313-win_amd64.whl` for Python 3.13 / torch 2.14 / CUDA 13.2; pick the `cp3xx` matching your Python and the `torch2.xx`/`cu1xx` matching your environment). Constraint fallbacks to PyTorch/SDPA with logged notices: attention mask present, headdim not in 64/128, or seq_len < 128. Use the `sparge_topk` input to trade accuracy vs speed (0.5 default; lower = more sparse/faster).
  * `sparge_topk` (optional): KV block keep ratio for `spargeattn` mode only (default 0.5, valid range (0, 1.0]). Higher keeps more KV blocks (more accurate, less acceleration); lower skips more blocks (faster, less accurate). `sparge_topk=1.0` computes all blocks (no skipping). **0 or invalid values are sanitized to 0.5 at run time with a log line** (the kernel computes `selected = topk * K`, so 0 would select zero blocks). Ignored by all other modes.
  * `allow_compile`: Allow torch.compile for SageAttention function (requires sageattn 2.2.0 or higher, default: False)
* **Use Case**: Use this node to replace ComfyUI's attention mechanism with SageAttention for better memory efficiency and performance. The node patches attention on each model execution and automatically cleans up afterward.
* **Technical Details**:
  * Uses ComfyUI's callback system (ON_PRE_RUN, ON_CLEANUP) to patch attention dynamically
  * Automatically detects SageAttention version and logs detailed information
  * SpargeAttn mode: uses a dedicated `get_sparge_func_dm()` code path (completely separate from the SageAttention modes) calling `spas_sage_hswq_attn.spas_sage2_attn_meansim_topk_cuda` (INT8 QK + FP8 PV quantized kernels with two-stage block-sparse filtering); per-head hyperparameters use the fork's plug-and-play defaults, E2-a scale sweep available via `SPARGE_SCALE_SWEEP=1`
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

3. (For SageAttention / SpargeAttn nodes) Install prebuilt wheels for Windows from the [Sage-Attention-and-Sparge-Attention-HSWQ](https://huggingface.co/ussoewwin/Sage-Attention-and-Sparge-Attention-HSWQ) Hugging Face repo — `sageattention` (SA2), `sageattn3` (SA3), and `spas_sage_hswq_attn` (SpargeAttn-hswq, used by the `spargeattn` mode) are published there for multiple Python/torch/CUDA combinations. Pick the wheel matching your environment (`cp3xx` = Python version, `torch2.xx` / `cu1xx` = torch/CUDA build).
4. Restart ComfyUI
5. Nodes will appear in the "Memory" category in the node palette

## Usage

### Basic Usage

1. Add any memory management node to your workflow
2. Connect any data to the input
3. Configure options as needed
4. Connect output to the next node

### Recommended Workflow Placement

**For ModelPatchLoader workflows**:

```
[ModelPatchLoader] → [QwenImageDiffsynthControlnet] → [General Purge VRAM V2 (clear_model_patches=True)] → [Upscaling Node]
```

**For general memory management**:

```
[Previous Node] → [Memory Manager] → [Next Node]
```

### Recommended Settings

**For ModelPatchLoader workflows (patch model format)**:

* Use **General Purge VRAM V2** (`DisTorchPurgeVRAMV2`)
* `clear_model_patches: True`
* `purge_models: False` (optional, to keep base models loaded while clearing patches)
* `purge_cache: True`
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

1. For ModelPatchLoader workflows: Use **General Purge VRAM V2** (`clear_model_patches: True`) after ControlNet usage
2. For general workflows: Use **Memory Manager**
3. Enable `clean_gpu` and `reset_virtual_memory`
4. Enable `force_gc` if needed

### OOM During Upscaling After ModelPatchLoader Usage

**Solution**:

1. Add **General Purge VRAM V2** node after QwenImageDiffsynthControlnet (when using ModelPatchLoader)
2. Enable `clear_model_patches: True`
3. Optionally set `purge_models: False` if you wish to retain base models while discarding patch models
4. **Note**: This applies to patch model format loaded via ModelPatchLoader, not standard ControlNet models

### UI Corruption

**Solution**:

1. Use **General Purge VRAM V2** or **Memory Manager**
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
* For ModelPatchLoader workflows: Always use General Purge VRAM V2 (`clear_model_patches: True`) before upscaling to prevent OOM. Note that patch model format loaded via ModelPatchLoader is an exceptional format different from standard ControlNet models.
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
