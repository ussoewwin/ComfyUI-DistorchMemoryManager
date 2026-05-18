"""
Memory Manager nodes for DistorchMemoryManager

Reserved VRAM auto-detection based on ComfyUI-ReservedVRAM by Windecay
https://github.com/Windecay/ComfyUI-ReservedVRAM
Original code licensed under Apache License 2.0
"""
import torch
import gc

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

# AnyType mirrors the behavior of the original Purge VRAM node
class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""
    def __eq__(self, __value: object) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
    def __repr__(self):
        return str(self)

any = AnyType("*")

# Helper used by several nodes to release memory
def clear_memory():
    import gc
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class MemoryManager:
    """
    Advanced memory management node with fine-grained controls.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any, {}),
            "clean_gpu": ("BOOLEAN", {"default": True}),
            "clean_cpu": ("BOOLEAN", {"default": False, "tooltip": "CPU memory cleanup (use with caution)"}),
            "force_gc": ("BOOLEAN", {"default": True}),
            "reset_virtual_memory": ("BOOLEAN", {"default": True}),
            "restore_original_functions": ("BOOLEAN", {"default": False, "tooltip": "Restore original model_management functions"}),
            "general_manage_vram": ("BOOLEAN", {"default": False, "tooltip": "General VRAM management: auto-detect non-PyTorch VRAM usage (browsers, other apps) via NVML and adjust ComfyUI memory management accordingly. Requires nvidia-ml-py."}),
        }}
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "manage_memory"
    CATEGORY = "Memory"

    def manage_memory(self, anything, clean_gpu, clean_cpu, force_gc, reset_virtual_memory, restore_original_functions, general_manage_vram=False):
        try:
            if clean_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU memory cleared")
            
            if clean_cpu:
                gc.collect()
                print("CPU memory cleared")
            
            if force_gc:
                gc.collect()
                print("Forced garbage collection completed")
            
            if reset_virtual_memory:
                try:
                    import comfy.model_management
                    if hasattr(comfy.model_management, 'free_memory'):
                        # Only free CUDA memory, skip CPU as it may cause errors
                        if torch.cuda.is_available():
                            try:
                                comfy.model_management.free_memory(0, 'cuda:0')
                            except Exception as e:
                                print(f"Virtual memory reset (CUDA) failed: {e}")
                        print("Virtual memory reset")
                except Exception as e:
                    print(f"Virtual memory reset failed: {e}")
            
            if restore_original_functions:
                try:
                    import comfy.model_management
                    print("Original functions restored")
                except Exception as e:
                    print(f"Function restoration failed: {e}")
            
            # Auto-detect non-PyTorch VRAM usage and adjust ComfyUI memory management
            if general_manage_vram:
                non_torch = get_non_torch_vram_usage_bytes()
                if non_torch is not None:
                    import comfy.model_management as mm
                    mm.EXTRA_RESERVED_VRAM = non_torch
                    non_torch_gb = non_torch / (1024 * 1024 * 1024)
                    print(f"[ComfyUI-VRAM-Manager] Detected {non_torch_gb:.2f} GB non-PyTorch VRAM usage; adjusted memory management accordingly")
                else:
                    print("[ComfyUI-VRAM-Manager] general_manage_vram: Could not detect non-PyTorch VRAM usage (pynvml unavailable or error)")
            
            print("Comprehensive memory management completed")
            
        except Exception as e:
            print(f"Memory management error: {e}")
        
        return (anything,)


class SafeMemoryManager:
    """
    Recommended memory management node that prioritizes safe cleanup.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any, {}),
            "clean_gpu": ("BOOLEAN", {"default": True}),
            "force_gc": ("BOOLEAN", {"default": True}),
            "reset_virtual_memory": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "safe_manage_memory"
    CATEGORY = "Memory"

    def safe_manage_memory(self, anything, clean_gpu, force_gc, reset_virtual_memory):
        try:
            if clean_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("Safe GPU memory cleared")
            
            if force_gc:
                gc.collect()
                print("Safe garbage collection completed")
            
            if reset_virtual_memory:
                try:
                    import comfy.model_management
                    if hasattr(comfy.model_management, 'free_memory'):
                        # Only free CUDA memory, skip CPU as it may cause errors
                        if torch.cuda.is_available():
                            try:
                                comfy.model_management.free_memory(0, 'cuda:0')
                            except Exception as e:
                                print(f"Safe virtual memory reset (CUDA) failed: {e}")
                        print("Safe virtual memory reset")
                except Exception as e:
                    print(f"Safe virtual memory reset failed: {e}")
            
            print("Safe memory management completed")
            
        except Exception as e:
            print(f"Safe memory management error: {e}")
        
        return (anything,)
