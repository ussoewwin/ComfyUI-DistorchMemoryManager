# Release History

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../zhmd/changelog/changelog.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

* **v2.4.1** – DisTorchPurgeVRAMV2 **HSWQ** purge: reclaim full HSWQ residual VRAM (PinCache force-import + drain, in-place PromptExecutor/SEGS cache clear without mid-prompt `reset()`, PINNED_MEMORY HostUnregister, gc nuclear unload). Method **2c** resets `comfy_kitchen` CUDA workspace / empty-tensor caches after nuclear kill so reload (including INT8 GEMM) remains valid after purge+reload. UI toggle renamed to **`HSWQ`** (legacy kwargs `"HSWQ INT8"` still accepted). Fallback `nodes/purge_vram.py` synced to root. See [Release Notes v2.4.1](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.4.1) for details.

* **v2.4.0** – Added automated GPU-wide VRAM management patch executed at startup. Auto-detects non-PyTorch VRAM usage (browsers, Discord, OBS, etc.) via NVML and dynamically adjusts ComfyUI's memory management (General Manage VRAM) at load time. This prevents OOM errors caused by external GPU processes that PyTorch cannot detect, ensuring system-wide memory safety. Added `nvidia-ml-py` dependency. See [Release Notes v2.4.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/v240_complete_guide.md) for details.

* **v2.3.8** – Technical documentation describing Flash Attention-2 issues when using ComfyUI on PyTorch 2.11.0, including failure symptoms and environment-related constraints. See [Release Notes v2.3.8](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.8) for details.

* **v2.3.7** – Added external runtime SageAttention noise guard behavior for the known `Unsupported head_dim: 160` fallback path, reduced repeated error-log spam, and aligned patch documentation to repository-scoped changes only. See [Release Notes v2.3.7](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.7) for details.

* **v2.3.6** – Enhanced SageAttention3 (SA3) integration in Patch Sage Attention DM node. Added SA3-specific version detection function (`get_sage_attention3_info()`) with Blackwell support detection. Improved SA3 implementation with tensor layout conversion (NHD to HND) and constraint handling (headdim >= 256, attention mask support). Added automatic fallback to PyTorch SDPA when SA3 constraints are not met. Fixed SA2 version logging to skip when SA3 modes are selected. Supports both `sageattn3` and `sageattn3_per_block_mean` modes with proper per-block mean processing. See [Release Notes v2.3.6](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.6) for details.

* **v2.3.5** – Fixed duplicate sys import in purge_vram.py. Removed redundant import statement that was already imported at module level.

* **v2.3.4** – Removed Safe Memory Manager node. The node has been removed from `__init__.py` and `README.md` as it is no longer needed. Users should use Memory Manager instead.

* **v2.3.3** – Fixed import paths for nodes in `nodes/` directory. All nodes (Memory Manager, Purge VRAM V2, Patch Sage Attention DM, Model Patch Memory Cleaner) are now correctly registered and displayed in ComfyUI. Resolves [Issue #3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/issues/3). See [Release Notes v2.3.3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.3) for details.

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
