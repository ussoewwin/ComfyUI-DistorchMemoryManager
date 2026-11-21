"""Backend capability helpers for the non-CUDA arms of this pack.

Design rules, all of them measured against ComfyUI 0.46.0 and torch 2.15.0:

* Nothing imports ComfyUI at module import time; ``comfy.model_management`` is
  imported lazily inside the probes so that a CUDA run is never affected by
  importing this module.
* ComfyUI's own flags are authoritative (``is_intel_xpu`` / ``is_amd``) because
  they also require ``cpu_state == GPU``, i.e. that ComfyUI is really running on
  the GPU. ``get_torch_device()`` tests ``is_intel_xpu()`` *before* the CUDA
  fallback, so on a dual-backend build ComfyUI itself selects XPU; the probes
  mirror that order instead of inventing a CUDA-first rule.
* Every ``torch.xpu`` call is wrapped: a missing capability becomes ``False`` or
  ``0``, never an exception. ``hasattr`` alone is not a gate -- on a CUDA-only
  build ``torch.xpu.synchronize()`` raises ``AssertionError``.
* ``torch.cuda`` is touched only by the host-pin helpers, which describe the
  CUDA capability that has no XPU equivalent.
"""

import torch

_XPU_CACHE = None
_AMD_CACHE = None


def _comfy_model_management():
    try:
        import comfy.model_management as mm
    except Exception:
        return None
    return mm


def is_nvidia_build():
    """True when this torch build targets NVIDIA.

    This is a build flag, not a run state: ``torch.version.cuda`` is set on every
    CUDA build even when ComfyUI runs in CPU mode, so gating the NVML paths on it
    cannot change what a CUDA build does today.
    """
    try:
        return bool(torch.version.cuda)
    except Exception:
        return False


def is_amd():
    """True when ComfyUI runs on AMD / ROCm.

    ROCm exposes itself through ``torch.cuda`` (``torch.version.hip`` is set and
    ``torch.version.cuda`` is None), so this is a build-and-run probe, not a
    device namespace of its own.
    """
    global _AMD_CACHE
    if _AMD_CACHE is not None:
        return _AMD_CACHE
    mm = _comfy_model_management()
    if mm is not None and hasattr(mm, "is_amd"):
        try:
            value = bool(mm.is_amd())
        except Exception:
            value = False
        _AMD_CACHE = value
        return value
    try:
        if not bool(torch.version.hip):
            return False
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_xpu():
    """True when ComfyUI runs on an Intel XPU.

    Returns False on this CUDA build by execution, which is the direct evidence
    that the CUDA path is untouched.
    """
    global _XPU_CACHE
    if _XPU_CACHE is not None:
        return _XPU_CACHE
    mm = _comfy_model_management()
    if mm is not None and hasattr(mm, "is_intel_xpu"):
        try:
            value = bool(mm.is_intel_xpu())
        except Exception:
            value = False
        _XPU_CACHE = value
        return value
    if not hasattr(torch, "xpu"):
        return False
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def xpu_device_count():
    if not is_xpu():
        return 0
    try:
        return int(torch.xpu.device_count())
    except Exception:
        return 0


def xpu_current_device():
    try:
        return int(torch.xpu.current_device())
    except Exception:
        return 0


def xpu_set_device(index):
    try:
        torch.xpu.set_device(int(index))
        return True
    except Exception:
        return False


def xpu_empty_cache():
    try:
        torch.xpu.empty_cache()
        return True
    except Exception:
        return False


def xpu_synchronize():
    try:
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def xpu_mem_get_info(index=None):
    """Return (free, total) in bytes, or (0, 0) when unavailable."""
    try:
        if index is None:
            free, total = torch.xpu.mem_get_info()
        else:
            free, total = torch.xpu.mem_get_info(int(index))
        return int(free), int(total)
    except Exception:
        return 0, 0


def xpu_memory_allocated(index=None):
    try:
        return int(torch.xpu.memory_allocated(None if index is None else int(index)))
    except Exception:
        return 0


def xpu_memory_reserved(index=None):
    try:
        return int(torch.xpu.memory_reserved(None if index is None else int(index)))
    except Exception:
        return 0


def xpu_ipc_collect():
    """No XPU equivalent exists; reports the skip instead of raising."""
    return False


def xpu_device_string(index):
    return "xpu:%d" % int(index)


def xpu_clear_cache_all_devices():
    """Synchronise then empty the XPU allocator on every device.

    Returns the number of devices whose cache was actually emptied. A build that
    cannot synchronise is reported once instead of being counted as success.
    """
    if not is_xpu():
        return 0
    cleared = 0
    sync_failures = 0
    try:
        current = xpu_current_device()
    except Exception:
        current = 0
    for index in range(xpu_device_count()):
        if not xpu_set_device(index):
            continue
        try:
            if not xpu_synchronize():
                sync_failures += 1
            if xpu_empty_cache():
                cleared += 1
        finally:
            xpu_set_device(current)
    if sync_failures:
        print(
            "Backend: XPU synchronize unavailable on %d device(s); "
            "the cache was emptied without waiting for in-flight work" % sync_failures
        )
    return cleared


def host_pin_supported():
    """True when the CUDA runtime can release host pins on this build.

    ComfyUI registers those pins through the same symbol
    (``comfy/pinned_memory.py``), so a build without it has no pins to release.
    """
    try:
        runtime = torch.cuda.cudart()
    except Exception:
        return False
    if runtime is None:
        return False
    return hasattr(runtime, "cudaHostUnregister")


def unregister_host_pin(pointer):
    """Release one host pin. Returns True on success, False when unsupported."""
    if not host_pin_supported():
        return False
    try:
        return int(torch.cuda.cudart().cudaHostUnregister(int(pointer))) == 0
    except Exception:
        return False


def capability_report():
    """The measured facts behind the gates, for one log line."""
    report = {
        "is_xpu": False,
        "is_amd": False,
        "is_nvidia_build": False,
        "xpu_is_available": False,
        "xpu_device_count": 0,
        "hip": None,
        "cuda": None,
        "host_pin_supported": False,
    }
    try:
        report["is_xpu"] = bool(is_xpu())
    except Exception:
        pass
    try:
        report["is_amd"] = bool(is_amd())
    except Exception:
        pass
    try:
        report["is_nvidia_build"] = bool(is_nvidia_build())
    except Exception:
        pass
    try:
        report["xpu_is_available"] = bool(torch.xpu.is_available())
    except Exception:
        report["xpu_is_available"] = False
    try:
        report["xpu_device_count"] = int(torch.xpu.device_count())
    except Exception:
        report["xpu_device_count"] = 0
    try:
        report["hip"] = torch.version.hip
    except Exception:
        pass
    try:
        report["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        report["host_pin_supported"] = bool(host_pin_supported())
    except Exception:
        pass
    return report
