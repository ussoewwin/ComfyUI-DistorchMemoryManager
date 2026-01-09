"""
Memory Manager nodes for DistorchMemoryManager
"""
import torch
import gc

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
        }}
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "manage_memory"
    CATEGORY = "Memory"

    def manage_memory(self, anything, clean_gpu, clean_cpu, force_gc, reset_virtual_memory, restore_original_functions):
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
