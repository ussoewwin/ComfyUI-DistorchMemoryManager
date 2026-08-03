# Technical Documentation: v2.4.0 VRAM Automatic Management Patch

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/blob/main/zhmd/v240_complete_guide.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

This document analyzes the behavior of the "Startup VRAM Automatic Patch (General Manage VRAM)" implemented in **ComfyUI-VRAM-Manager (v2.4.0)**, validates how it overrides ComfyUI's standard memory manager, and provides the full text of the program code added or modified in v2.4.0 along with technical explanations.

*Note: This document focuses strictly on the newly added/modified program files in v2.4.0 and excludes unmodified legacy code blocks or documentation files.*

---

## 1. Technical Significance of the Modification

### Architectural Limitation of ComfyUI and PyTorch
By default, ComfyUI's memory management system (`comfy.model_management`) only recognizes VRAM allocations made through the PyTorch framework's internal allocator.
PyTorch has no built-in capability to detect physical VRAM allocated by other active OS processes (such as Web browsers, Discord, OBS, or Desktop Window Manager). This is a core limitation of standard ComfyUI.

Due to this blind spot, ComfyUI behaves as follows:
- Even when external processes consume a significant portion of VRAM, ComfyUI assumes that all non-PyTorch VRAM is fully free, leading it to plan excessive model transfer schedules.
- This discrepancy frequently results in sudden Out-of-Memory (OOM) failures and process crashes at the OS level.

Particularly in multi-GPU configurations where Windows desktop rendering and browser acceleration are offloaded to an integrated GPU (e.g., Ryzen 9 7900 iGPU) and the RTX discrete GPU is dedicated to CUDA, external VRAM consumption on the RTX card is extremely low (e.g., around `0.02 GB`).
Conversely, ComfyUI defaults to reserving a rigid safety margin of `0.68 GB` on Windows setups, needlessly restricting the VRAM space available for weight storage and inference.

### Technical Approach of the Patch
The v2.4.0 patch rectifies these issues through the following runtime flow:
1. **Physical VRAM Querying via NVML**: Uses `pynvml` to retrieve the real-time physical memory utilization of the GPU.
2. **External Overhead Calculation**: Subtracts PyTorch's active allocations from the system-wide utilized VRAM (`system_used - torch_used`) to determine the exact bytes allocated to non-PyTorch processes.
3. **Dynamic Margin Injection**: Overrides ComfyUI's core safety threshold variable (`comfy.model_management.EXTRA_RESERVED_VRAM`) with the calculated external VRAM usage at startup.

---

## 2. Validation of the ComfyUI Memory Core Hack

The following details how this startup patch intercepts ComfyUI's memory management pipeline, referencing the structural definitions within ComfyUI's core codebase (`comfy/model_management.py`).

### I. ComfyUI Core Variable Setup
In `comfy/model_management.py`, the reserve margin variable `EXTRA_RESERVED_VRAM` is defined as follows:

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

### II. Core Decision Points Dependent on `EXTRA_RESERVED_VRAM`
This variable directly dictates critical memory scheduling thresholds:

| Host Function | Operational Impact |
|---|---|
| `extra_reserved_memory()` | Evaluates and returns the active value of `EXTRA_RESERVED_VRAM`. |
| `minimum_inference_memory()` | Computes `0.8GB + extra_reserved_memory()`, setting the lower VRAM bound required to safely run inference. |
| `free_memory()` | When loading a model, evaluates `memory_required + extra_reserved_memory()` to decide whether to unload existing models from VRAM. |
| `maximum_vram_for_weights()` | Computes `total_vram * 0.88 - minimum_inference_memory()` to set the absolute upper bound of VRAM allocated to model weights. |

### III. Interception Mechanism
The patch execution in `__init__.py` intercepts this pipeline as follows:

```python
import comfy.model_management as mm
original_general_manage_vram = getattr(mm, "EXTRA_RESERVED_VRAM", 0)
mm.EXTRA_RESERVED_VRAM = non_torch
```

Because Python shares imported module attributes globally across the process namespace, assigning the physical non-PyTorch overhead (`non_torch`) to `mm.EXTRA_RESERVED_VRAM` at startup causes all subsequent evaluations in `extra_reserved_memory()`, `minimum_inference_memory()`, `free_memory()`, and `maximum_vram_for_weights()` to dynamically adapt to the physical GPU state.

---

## 3. Added and Modified Source Code in v2.4.0

### I. Code Added to `__init__.py`

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

#### Technical Explanation of the Added Code
- **`_install_general_vram_management()`**
  - A startup hook executed once when ComfyUI initializes. It calculates external VRAM overhead via NVML and adjusts ComfyUI's core thresholds.
- **NVML Import and Initialization**
  - Attempts to safely import `pynvml` and call `nvmlInit()`. If `nvidia-ml-py` is not installed or initialization fails, the function catches the exception and exits cleanly to prevent ComfyUI from crashing during startup.
- **CUDA Verification**
  - Checks `torch.cuda.is_available()`. Exits early on non-GPU/CPU environments.
- **Hardware Profile Retrieval**
  - Obtains the device handle for GPU 0 via `nvmlDeviceGetHandleByIndex(0)` and queries the GPU model name, total physical VRAM capacity, and active physical memory allocation (`used`) in bytes.
- **PyTorch Memory Computation**
  - Evaluates `torch.cuda.mem_get_info()` to compute the active VRAM allocated inside PyTorch (`total - free`).
- **External Allocation Calculation**
  - Subtracts PyTorch memory from NVML physical memory and clamps the result using `max(0, ...)` to determine the non-PyTorch processes' exact VRAM overhead.
- **Dynamic Override Action**
  - Imports `comfy.model_management` and assigns the computed value directly to the core reserve parameter `EXTRA_RESERVED_VRAM`.
- **System Logging**
  - Converts metrics to gigabytes and prints a formatted telemetry report to the ComfyUI console.

---

### II. Code Added to `nodes/memory_manager.py` (and `memory_manager.py`)

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

#### Technical Explanation of the Added Code
- **NVML Safe Import Block**
  - Implements a robust try-except wrapper to load the NVML bindings within the node scope, defining the `_pynvml_available` flag. If the library is missing or errors out, it cleanly disables NVML features without interrupting the custom node registration.
- **`get_non_torch_vram_usage_bytes()`**
  - A utility function designed to compute external process VRAM overhead on demand.
  - Returns `None` if NVML is unavailable or CUDA is inactive to prevent returning invalid telemetry to caller nodes.
  - Dynamically subtracts active PyTorch memory from NVML-reported physical GPU allocations and returns the clamped non-negative byte count.
