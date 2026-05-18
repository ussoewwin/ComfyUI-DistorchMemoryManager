# 技术文档：v2.4.0 VRAM 自动管理补丁

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/v240_complete_guide.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

本文分析 **ComfyUI-VRAM-Manager (v2.4.0)** 中实现的「启动时 VRAM 自动补丁（General Manage VRAM）」的行为，说明其如何覆盖 ComfyUI 标准显存管理器，并给出 v2.4.0 新增或修改的程序代码全文及技术说明。

*说明：本文仅聚焦 v2.4.0 新增/修改的程序文件，不包含未改动的遗留代码块或文档文件。*

---

## 1. 修改的技术意义

### ComfyUI 与 PyTorch 的架构局限
默认情况下，ComfyUI 的显存管理系统（`comfy.model_management`）仅识别通过 PyTorch 框架内部分配器占用的 VRAM。
PyTorch 无法内置检测其他活跃操作系统进程（如浏览器、Discord、OBS、桌面窗口管理器等）占用的物理 VRAM。这是标准 ComfyUI 的核心局限。

由于这一盲区，ComfyUI 的行为如下：
- 即使外部进程已占用大量 VRAM，ComfyUI 仍假定所有非 PyTorch 占用的 VRAM 完全空闲，从而规划过度的模型搬运调度。
- 这种偏差常导致操作系统层面的突发显存不足（OOM）与进程崩溃。

在多 GPU 配置中，若将 Windows 桌面渲染与浏览器加速卸载到核显（例如 Ryzen 9 7900 iGPU），而将 RTX 独显专用于 CUDA，则 RTX 卡上的外部 VRAM 占用通常极低（例如约 `0.02 GB`）。
相反，在 Windows 环境下 ComfyUI 默认保留约 `0.68 GB` 的固定安全余量，不必要地压缩可用于权重存储与推理的 VRAM 空间。

### 补丁的技术思路
v2.4.0 补丁通过以下运行时流程纠正上述问题：
1. **通过 NVML 查询物理 VRAM**：使用 `pynvml` 获取 GPU 的实时物理显存占用。
2. **计算外部开销**：从系统级已用 VRAM（`system_used`）中减去 PyTorch 活跃分配（`torch_used`），得到非 PyTorch 进程占用的精确字节数。
3. **动态注入余量**：在启动时将计算出的外部 VRAM 占用写入 ComfyUI 核心安全阈值变量（`comfy.model_management.EXTRA_RESERVED_VRAM`）。

---

## 2. ComfyUI 显存核心劫持的验证

以下说明该启动补丁如何拦截 ComfyUI 显存管理流水线，并引用 ComfyUI 核心代码（`comfy/model_management.py`）中的结构定义。

### I. ComfyUI 核心变量设置
在 `comfy/model_management.py` 中，保留余量变量 `EXTRA_RESERVED_VRAM` 定义如下：

```python
# Default value (non-Windows)
EXTRA_RESERVED_VRAM = 400 * 1024 * 1024        # ~0.39 GB

# Overridden under Windows environment
if WINDOWS:
    EXTRA_RESERVED_VRAM = 600 * 1024 * 1024     # ~0.59 GB
    if total_vram > (15 * 1024):                 # GPUs with >=16GB VRAM
        EXTRA_RESERVED_VRAM += 100 * 1024 * 1024 # Total ~0.68 GB (700MB)

# Overridden by command-line argument --reserve-vram
if args.reserve_vram is not None:
    EXTRA_RESERVED_VRAM = args.reserve_vram * 1024 * 1024 * 1024
```

### II. 依赖 `EXTRA_RESERVED_VRAM` 的核心决策点
该变量直接决定关键显存调度阈值：

| 宿主函数 | 运行影响 |
|---|---|
| `extra_reserved_memory()` | 计算并返回 `EXTRA_RESERVED_VRAM` 的当前有效值。 |
| `minimum_inference_memory()` | 计算 `0.8GB + extra_reserved_memory()`，设定安全运行推理所需的 VRAM 下限。 |
| `free_memory()` | 加载模型时评估 `memory_required + extra_reserved_memory()`，决定是否从 VRAM 卸载已有模型。 |
| `maximum_vram_for_weights()` | 计算 `total_vram * 0.88 - minimum_inference_memory()`，设定分配给模型权重的 VRAM 绝对上限。 |

### III. 拦截机制
`__init__.py` 中的补丁执行如下拦截：

```python
import comfy.model_management as mm
original_general_manage_vram = getattr(mm, "EXTRA_RESERVED_VRAM", 0)
mm.EXTRA_RESERVED_VRAM = non_torch
```

由于 Python 在进程命名空间内全局共享已导入模块属性，在启动时将物理非 PyTorch 开销（`non_torch`）赋给 `mm.EXTRA_RESERVED_VRAM` 后，`extra_reserved_memory()`、`minimum_inference_memory()`、`free_memory()` 与 `maximum_vram_for_weights()` 的后续求值都会动态适应物理 GPU 状态。

---

## 3. v2.4.0 新增与修改的源代码

### I. 新增至 `__init__.py` 的代码

```python
def _install_general_vram_management():
    """
    Startup patch: auto-detect non-PyTorch VRAM usage (browsers, other apps)
    via NVML and apply general manage VRAM patch to ComfyUI memory management at load time.
    """
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
        except Exception as e:
            print(f"[ComfyUI-VRAM-Manager] WARNING: pynvml NVML init failed: {e}")
            return
    except ImportError:
        print("[ComfyUI-VRAM-Manager] INFO: pynvml (nvidia-ml-py) not installed. Startup VRAM patch skipped.")
        return

    if not torch.cuda.is_available():
        return

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        nvml_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        system_used = nvml_info.used
        vram_total = nvml_info.total

        torch_free, torch_total = torch.cuda.mem_get_info()
        torch_used = torch_total - torch_free

        non_torch = max(0, system_used - torch_used)

        import comfy.model_management as mm
        original_general_manage_vram = getattr(mm, "EXTRA_RESERVED_VRAM", 0)
        mm.EXTRA_RESERVED_VRAM = non_torch

        # Detailed startup log
        to_gb = lambda b: b / (1024 * 1024 * 1024)
        print(f"[ComfyUI-VRAM-Manager] ── Startup VRAM Patch ──")
        print(f"[ComfyUI-VRAM-Manager]   GPU: {gpu_name}")
        print(f"[ComfyUI-VRAM-Manager]   VRAM Total:          {to_gb(vram_total):.2f} GB")
        print(f"[ComfyUI-VRAM-Manager]   System-wide used:    {to_gb(system_used):.2f} GB  (NVML)")
        print(f"[ComfyUI-VRAM-Manager]   PyTorch used:        {to_gb(torch_used):.2f} GB")
        print(f"[ComfyUI-VRAM-Manager]   Non-PyTorch used:    {to_gb(non_torch):.2f} GB  (browsers, other apps)")
        print(f"[ComfyUI-VRAM-Manager]   General Manage VRAM: {to_gb(original_general_manage_vram):.2f} GB → {to_gb(non_torch):.2f} GB")
        print(f"[ComfyUI-VRAM-Manager] ── Patch applied ──")
    except Exception as e:
        print(f"[ComfyUI-VRAM-Manager] Startup patch error: {e}")


_install_general_vram_management()
```

#### 新增代码的技术说明
- **`_install_general_vram_management()`**
  - ComfyUI 初始化时执行一次的启动钩子。通过 NVML 计算外部 VRAM 开销并调整 ComfyUI 核心阈值。
- **NVML 导入与初始化**
  - 安全导入 `pynvml` 并调用 `nvmlInit()`。若未安装 `nvidia-ml-py` 或初始化失败，捕获异常并干净退出，避免 ComfyUI 启动崩溃。
- **CUDA 校验**
  - 检查 `torch.cuda.is_available()`。在非 GPU/仅 CPU 环境下提前返回。
- **硬件信息获取**
  - 通过 `nvmlDeviceGetHandleByIndex(0)` 获取设备句柄，查询 GPU 型号、物理 VRAM 总容量与当前物理占用（`used`，字节）。
- **PyTorch 显存计算**
  - 调用 `torch.cuda.mem_get_info()`，计算 PyTorch 内活跃 VRAM（`total - free`）。
- **外部分配计算**
  - 用 NVML 物理占用减去 PyTorch 占用，并以 `max(0, ...)` 钳位，得到非 PyTorch 进程的精确 VRAM 开销。
- **动态覆盖动作**
  - 导入 `comfy.model_management`，将计算值直接写入核心保留参数 `EXTRA_RESERVED_VRAM`。
- **系统日志**
  - 将指标转换为 GB，在 ComfyUI 控制台输出格式化遥测报告。

---

### II. 新增至 `nodes/memory_manager.py`（及 `memory_manager.py`）的代码

```python
# Safe import for pynvml (NVIDIA Management Library)
try:
    import pynvml
    try:
        pynvml.nvmlInit()
        _pynvml_available = True
    except Exception as e:
        _pynvml_available = False
        pynvml = None
        print(f"[ComfyUI-VRAM-Manager] WARNING: pynvml imported but NVML init failed: {e}")
except ImportError:
    _pynvml_available = False
    pynvml = None
    print("[ComfyUI-VRAM-Manager] INFO: pynvml (nvidia-ml-py) not installed. general_manage_vram will be unavailable.")


def get_non_torch_vram_usage_bytes():
    """
    Returns the amount of VRAM (in bytes) consumed by non-PyTorch processes.
    This is the difference between system-wide GPU usage (via NVML) and
    PyTorch's own reported usage.
    Returns None if detection is not possible.
    """
    if not _pynvml_available or pynvml is None:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        nvml_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        system_used = nvml_info.used  # bytes, all processes

        torch_free, torch_total = torch.cuda.mem_get_info()
        torch_used = torch_total - torch_free  # bytes, PyTorch only

        non_torch = system_used - torch_used
        return max(0, non_torch)
    except Exception as e:
        print(f"[ComfyUI-VRAM-Manager] Error detecting non-PyTorch VRAM usage: {e}")
        return None
```

#### 新增代码的技术说明
- **NVML 安全导入块**
  - 在节点作用域内用 try-except 加载 NVML 绑定，定义 `_pynvml_available` 标志。库缺失或出错时干净禁用 NVML 功能，不中断自定义节点注册。
- **`get_non_torch_vram_usage_bytes()`**
  - 按需计算外部进程 VRAM 开销的工具函数。
  - NVML 不可用或 CUDA 未激活时返回 `None`，避免向调用节点返回无效遥测。
  - 动态用 NVML 报告的物理 GPU 占用减去活跃 PyTorch 显存，返回钳位后的非负字节数。
