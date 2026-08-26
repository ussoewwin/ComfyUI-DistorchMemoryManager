# 发行历史

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../../changelog/changelog.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

* **v2.4.6** – DisTorchPurgeVRAMV2：修复 HSWQ 清理时的 unpin 警告刷屏（`[WARNING] Tried to unpin tensor not pinned by ComfyUI`）与 CUDA 驱动中止崩溃（`Fatal Python error: Aborted`）。在 `_kill_tensor_storage`、`Method 0s`/`0s2`（Detailer SEGS / PromptExecutor 扫描）及 `Method 3`（gc 核清理）中遇到的 pinned 张量，调用 `mm.unpin_memory()` 前均增加 `comfy.model_management.PINNED_MEMORY` 校验。非 ComfyUI 注册的 pinned 张量（如 PyTorch 页锁定内存或 Detailer SEGS 缓存）安全跳过，不再触发 ComfyUI 警告日志。彻底移除对非 ComfyUI 注册张量直接调用 `cudaHostUnregister()` 的非法操作，防止 CUDA 驱动页表损坏及后续 `clear_nvfp4_runtime_pools` / CUDA 缓存重置时的致命进程崩溃。根目录 `purge_vram.py` 与 `nodes/purge_vram.py` 严格同步。详见 [Release Notes v2.4.6](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v2.4.6.md)。
* **v2.4.5** – DisTorchPurgeVRAMV2：HSWQ 清理完整支持 **Krea2 ConvRot NVFP4**。递归清空容器结构（`list`/`tuple`/`dict`）内嵌套的 low-rank LoRA 残差张量（`_hswq_krea2_lora_res`、`_hswq_krea2_lora_res_gpu`），彻底消除显存泄露并防止模型运行间的残差交叉污染。完整剥离 `mixed_precision_ops`、`convert_old_quants` 及 `detect_unet_config`（`_hswq_krea2_txtlayers_fix` 闭包复原）上的 Krea2 ops 包装。通过 `uninstall_krea2_nvfp4_lora_bake()` 卸载动态 LoRA bake 钩子。重置已加载 NVFP4 子模块中的激活/旋转运行时池（`_ACT_Q_POOL`、`_ROT_OUT_POOL`）、CUDA graph 缓存（`_GRAPH_CACHE`）、Hadamard 矩阵查找表以及 forward/bake 诊断日志计数器。fallback `nodes/purge_vram.py` 与根目录严格同步。详见 [Release Notes v2.4.5](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v2.4.5.md)。
* **v2.4.4** – Soft purge / Memory Manager：在 **HSWQ** 关闭时回收 MultiGPU–NVFP4（如 Krea2）unload 后残留的 CUDA（约 9–12 GB）：对已加载模型强制清空残留 CUDA param/buffer storage，再对全部 CUDA 设备执行 `unload_all_models()` + `free_memory(1e30)`（`free_memory(0)` 为空操作）。storage 销毁为 **仅 CUDA**，避免 Ollama 邻近的 soft/HSWQ purge 后擦掉 CPU CLIP / Z Image TE 重载源（`Embedding.weight` 非 2D）。HSWQ Methods 0–2c 仍仅由开关门控（不自动武装）。相对 v2.4.3 另扩展 HSWQ Method **2c** / 核清理表面：完整 INT8/NVFP4/Detailer 残留清理、ZI ConvRot parity / bake 剥离（`restore_nvfp4_tc_product_stack` / `uninstall_zimage_nvfp4_lora_bake`）、Hadamard / inplace 池清理。详见 [Release Notes v2.4.4](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v2.4.4.md)。
* **v2.4.3** – DisTorchPurgeVRAMV2 **HSWQ** Method **2c**：在核清理后，除既有 `comfy_kitchen` `_cublas_workspaces` / `_empty_cuda_tensors` 重置外，另通过 `sys.modules` 扫描 `nvfp4_runtime` 并调用 `clear_nvfp4_runtime_pools()`，清空 HSWQ **NVFP4** 运行时池 / CUDA graphs，避免 purge 后第二次 ConvRot NVFP4 生成出现 `quantize_nvfp4` / `PyCapsule` / `pooled TC path failed`。优先从 `nodes/purge_vram.py` 导入 `DisTorchPurgeVRAMV2`；日志前缀 `HSWQ INT8/NVFP4:`。详见 [Release Notes v2.4.3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v2.4.3.md)。
* **v2.4.2** – DisTorchPurgeVRAMV2：新增 **`Ollama`** 开关（位于 **`HSWQ`** 下方），完整清理 **comfyui-ollama** 与 **comfyui-ollama-describer**（describer 默认 `keep_model_alive=-1`）残留的 Ollama 显存：循环 `GET /api/ps` 至空，配合 `/api/generate` 与 `/api/chat` 的 `keep_alive=0`、`ollama stop`、Client 回退；从两个包收集 `api_host`/`url`；清空 `CHAT_SESSIONS` / `saved_context`；删除 `saved_context/` 文件；最终 `/api/ps` 验证。fallback `nodes/purge_vram.py` 与 root 同步。详见 [Release Notes v2.4.2](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v2.4.2.md)。

* **v2.4.1** – DisTorchPurgeVRAMV2 **HSWQ** 清理：回收完整 HSWQ 残留显存（PinCache 强制导入与排空、不在 prompt 中途调用 `reset()` 的 PromptExecutor/SEGS 就地清空、PINNED_MEMORY HostUnregister、gc 核清理）。Method **2c** 在核清理后重置 `comfy_kitchen` CUDA workspace / empty-tensor 缓存，使 purge+reload（含 INT8 GEMM）仍可用。UI 开关更名为 **`HSWQ`**（仍接受旧 kwargs `"HSWQ INT8"`）。fallback `nodes/purge_vram.py` 与 root 同步。详见 [Release Notes v2.4.1](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v2.4.1.md)。

* **v2.4.0** – 新增在 ComfyUI 启动时执行的 GPU 全局 VRAM 管理自动补丁。通过 NVML 自动检测非 PyTorch 的 VRAM 占用（浏览器、Discord、OBS 等），并在加载时动态调整 ComfyUI 的显存管理（General Manage VRAM）。可避免 PyTorch 无法感知的外部 GPU 进程导致的 OOM，保障系统级显存安全。新增 `nvidia-ml-py` 依赖。详见 [Release Notes v2.4.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v240_complete_guide.md)。

* **v2.3.8** – 技术文档：说明在 PyTorch 2.11.0 下使用 ComfyUI 时的 Flash Attention-2 相关问题，包括故障现象与环境约束。详见 [Release Notes v2.3.8](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.8)。

* **v2.3.7** – 为已知的 `Unsupported head_dim: 160` 回退路径增加外部运行时 SageAttention 噪声防护，减少重复错误日志刷屏，并将补丁文档限定为仅描述本仓库范围内的变更。详见 [Release Notes v2.3.7](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.7)。

* **v2.3.6** – 增强 Patch Sage Attention DM 节点中的 SageAttention3（SA3）集成。新增 SA3 专用版本检测函数（`get_sage_attention3_info()`），支持 Blackwell 检测。改进 SA3 实现：张量布局转换（NHD 至 HND）、约束处理（headdim >= 256、attention mask 支持）。当不满足 SA3 约束时自动回退至 PyTorch SDPA。修复在选择 SA3 模式时仍输出 SA2 版本日志的问题。支持 `sageattn3` 与 `sageattn3_per_block_mean` 模式及正确的 per-block mean 处理。详见 [Release Notes v2.3.6](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.6)。

* **v2.3.5** – 修复 purge_vram.py 中重复的 sys 导入，移除模块级已存在的冗余 import。

* **v2.3.4** – 移除 Safe Memory Manager 节点。该节点已从 `__init__.py` 与 `README.md` 中删除，请改用 Memory Manager。

* **v2.3.3** – 修复 `nodes/` 目录下节点的导入路径。Memory Manager、Purge VRAM V2、Patch Sage Attention DM、Model Patch Memory Cleaner 均已正确注册并在 ComfyUI 中显示。解决 [Issue #3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/issues/3)。详见 [Release Notes v2.3.3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.3)。

* **v2.3.0** – 新增 Patch Sage Attention DM 节点，将 ComfyUI 的 attention 机制补丁为使用 SageAttention。支持多种 SageAttention 实现（auto、CUDA、Triton、SageAttention 3），通过 ComfyUI 回调系统动态打补丁。新增 Flash-Attention 与 SageAttention 的独立版本检测（与 model_management 模块完全独立）。禁用时自动加载 Flash-Attention（无需 CLI 参数）。每次生成时自动检测并记录 SageAttention 版本（含 CUDA/PyTorch 信息）及 Flash-Attention 版本（FA-2/FA-3 类型）。兼容 ComfyUI attention 函数格式。详见 [Release Notes v2.3.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.0)。

* **v2.2.0** – 新增 Patch Sage Attention DM 节点，将 ComfyUI 的 attention 机制补丁为使用 SageAttention。支持多种 SageAttention 实现（auto、CUDA、Triton、SageAttention 3），通过 ComfyUI 回调系统动态打补丁。自动检测并记录 SageAttention 版本（含 CUDA/PyTorch 信息）。兼容 ComfyUI attention 函数格式。详见 [Release Notes v2.2.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.2.0)。

* **v2.0.0** – DisTorchPurgeVRAMV2 节点新增 Qwen3-VL 与 Nunchaku 模型清理支持。Qwen3-VL 支持 device_map="auto" 下从 GPU 显存清理；Nunchaku 模型（FLUX/Z-Image/Qwen-Image）支持 CPU offload 处理。增强 CUDA 缓存清理以支持全部设备。修复 any() 与 AnyType 的命名冲突。增加完整调试日志。显示名称改为 ComfyUI-VRAM-Manager。详见 [Release Notes v2.0.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.0.0)。

* **v1.3.1** – 改进 SeedVR2 缓存检测与提示信息。移除重复消息。明确 cache_model=False（默认）表示模型不会缓存在 GlobalModelCache。增加缓存状态的详细调试信息。详见 [Release Notes v1.3.1](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v1.3.1)。

* **v1.3.0** – DisTorchPurgeVRAMV2 节点新增 SeedVR2 模型清理支持。修复 cleanup_models() 中「NoneType object is not callable」错误。修复虚拟内存重置时的 CPU 设备错误。改进 SeedVR2 自定义节点路径检测以适配不同用户环境。详见 [Release Notes v1.3.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v1.3.0)。

* **v1.2.0** – 新增 Model Patch Memory Cleaner 节点，用于 ModelPatchLoader 的模型补丁（patch model 格式）。防止 ModelPatchLoader 使用后放大时 OOM。处理与标准 ControlNet 不同的特殊 patch 模型格式。增强 DisTorchPurgeVRAMV2：更激进的模型卸载、改进错误处理与安全 None 检查。支持从缓存清理 SeedVR2 的 DiT 与 VAE 模型。修复虚拟内存重置时的 CPU 设备错误。改进 ComfyUI 核心中 cleanup_models() 与 is_dead() 的错误处理。详见 [Release Notes v1.2.0](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v1.2.0)。

* v1.10.1 – 热修复：确保 DisTorch Purge VRAM V2 节点随包一起发布。

* v1.10 – 在 DisTorch Memory Manager 内新增 LayerUtility: Purge VRAM V2 兼容节点。

* v1.1.0 – 新增 ANY 类型 I/O 支持，简化节点名称，分类移至「Memory」。

* v1.0.0 – 首发，提供核心显存管理功能。
