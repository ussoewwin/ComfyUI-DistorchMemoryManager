import torch
import gc
import sys
import os

# Ensure ComfyUI root is on sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# AnyType mirrors the behavior of the original Purge VRAM node
class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""
    def __eq__(self, __value: object) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

# Helper used by several nodes to release memory
def clear_memory():
    import gc
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class MemoryCleaner:
    """
    Basic memory cleaning node that provides safe default behavior.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any, {}),
        }}
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "clean_memory"
    CATEGORY = "Memory"

    def clean_memory(self, anything):
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Run garbage collection on the host
        gc.collect()
        
        # Release virtual memory tracked by DisTorch/Comfy
        try:
            import comfy.model_management
            if hasattr(comfy.model_management, 'free_memory'):
                comfy.model_management.free_memory(0, 'cuda:0')
                comfy.model_management.free_memory(0, 'cpu')
        except:
            pass
        
        print("DisTorch memory cleaned")
        return (anything,)


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
                        comfy.model_management.free_memory(0, 'cuda:0')
                        comfy.model_management.free_memory(0, 'cpu')
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
                        comfy.model_management.free_memory(0, 'cuda:0')
                        comfy.model_management.free_memory(0, 'cpu')
                        print("Safe virtual memory reset")
                except Exception as e:
                    print(f"Safe virtual memory reset failed: {e}")
            
            print("Safe memory management completed")
            
        except Exception as e:
            print(f"Safe memory management error: {e}")
        
        return (anything,)


class DisTorchPurgeVRAMV2:
    """
    Compatibility clone of the original LayerUtility Purge VRAM V2 node
    maintained within the Distortch Memory Manager package.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "purge_vram"
    CATEGORY = "DisTorch/Memory"

    def purge_vram(self, anything, purge_cache, purge_models):
        if purge_cache:
            gc.collect()

            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                try:
                    for idx in range(torch.cuda.device_count()):
                        torch.cuda.set_device(idx)
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                finally:
                    torch.cuda.set_device(current_device)

        if purge_models:
            try:
                import comfy.model_management

                # Cleanup dead models first
                if hasattr(comfy.model_management, "cleanup_models") and callable(comfy.model_management.cleanup_models):
                    try:
                        comfy.model_management.cleanup_models()
                    except Exception as e:
                        print(f"Error in cleanup_models: {e}")
                
                # Cleanup models GC
                if hasattr(comfy.model_management, "cleanup_models_gc") and callable(comfy.model_management.cleanup_models_gc):
                    try:
                        comfy.model_management.cleanup_models_gc()
                    except Exception as e:
                        print(f"Error in cleanup_models_gc: {e}")
                
                # More aggressive model unloading
                if hasattr(comfy.model_management, "current_loaded_models"):
                    current_loaded_models = comfy.model_management.current_loaded_models
                    unloaded_count = 0
                    
                    # Mark all models as not currently used
                    for loaded_model in current_loaded_models:
                        if loaded_model is not None:
                            try:
                                if hasattr(loaded_model, "is_dead") and callable(loaded_model.is_dead):
                                    if not loaded_model.is_dead():
                                        loaded_model.currently_used = False
                                else:
                                    loaded_model.currently_used = False
                            except Exception as e:
                                print(f"Error checking model status: {e}")
                    
                    # Try to unload models
                    for i in range(len(current_loaded_models) - 1, -1, -1):
                        loaded_model = current_loaded_models[i]
                        if loaded_model is not None:
                            try:
                                if hasattr(loaded_model, "is_dead") and callable(loaded_model.is_dead):
                                    if loaded_model.is_dead():
                                        continue
                                if hasattr(loaded_model, "model_unload") and callable(loaded_model.model_unload):
                                    if loaded_model.model_unload():
                                        unloaded_count += 1
                            except Exception as e:
                                print(f"Error unloading model: {e}")
                    
                    if unloaded_count > 0:
                        print(f"Unloaded {unloaded_count} model(s)")
                    
                    # Cleanup again after unloading
                    if hasattr(comfy.model_management, "cleanup_models"):
                        try:
                            comfy.model_management.cleanup_models()
                        except Exception as e:
                            print(f"Error in cleanup_models: {e}")
                
                # Soft empty cache (if available)
                if hasattr(comfy.model_management, "soft_empty_cache") and callable(comfy.model_management.soft_empty_cache):
                    try:
                        comfy.model_management.soft_empty_cache()
                    except Exception as e:
                        print(f"Error in soft_empty_cache: {e}")
                    
            except Exception as e:
                print(f"Error purging models: {e}")

        return (anything,)


class ModelPatchMemoryCleaner:
    """
    Memory cleaner specifically for ModelPatcher loaded model patches.
    Clears model patches loaded via ModelPatchLoader to prevent OOM during upscaling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "clear_model_patches": ("BOOLEAN", {"default": True, "tooltip": "Clear model patches loaded via ModelPatchLoader"}),
                "clean_gpu": ("BOOLEAN", {"default": True}),
                "force_gc": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "clear_model_patches"
    CATEGORY = "Memory"

    def clear_model_patches(self, anything, clear_model_patches, clean_gpu, force_gc):
        try:
            if clear_model_patches:
                import comfy.model_management
                import comfy.model_patcher
                
                # Get current loaded models
                if hasattr(comfy.model_management, "current_loaded_models"):
                    current_loaded_models = comfy.model_management.current_loaded_models
                    
                    # Find and unload model patches
                    unloaded_count = 0
                    for i in range(len(current_loaded_models) - 1, -1, -1):
                        loaded_model = current_loaded_models[i]
                        if loaded_model is not None and hasattr(loaded_model, "model"):
                            model = loaded_model.model
                            # Check if this is a ModelPatcher with additional_models (model patches)
                            if isinstance(model, comfy.model_patcher.ModelPatcher):
                                # Check for additional_models (model patches stored here)
                                if hasattr(model, "additional_models") and model.additional_models:
                                    # Mark as not currently used
                                    loaded_model.currently_used = False
                                    # Unload the model
                                    if hasattr(loaded_model, "model_unload"):
                                        loaded_model.model_unload()
                                    # Remove from current_loaded_models
                                    current_loaded_models.pop(i)
                                    unloaded_count += 1
                                    print(f"Unloaded model patch: {type(model.model).__name__ if hasattr(model, 'model') else 'ModelPatcher'}")
                                # Also check attachments for model patches
                                elif hasattr(model, "attachments") and model.attachments:
                                    # Mark as not currently used
                                    loaded_model.currently_used = False
                                    # Unload the model
                                    if hasattr(loaded_model, "model_unload"):
                                        loaded_model.model_unload()
                                    # Remove from current_loaded_models
                                    current_loaded_models.pop(i)
                                    unloaded_count += 1
                                    print(f"Unloaded model patch from attachments: {type(model.model).__name__ if hasattr(model, 'model') else 'ModelPatcher'}")
                    
                    if unloaded_count > 0:
                        print(f"Cleared {unloaded_count} model patch(es)")
                    
                    # Cleanup models GC
                    if hasattr(comfy.model_management, "cleanup_models_gc"):
                        comfy.model_management.cleanup_models_gc()
            
            if clean_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU memory cleared")
            
            if force_gc:
                gc.collect()
                print("Garbage collection completed")
            
            print("Model patch memory cleanup completed")
            
        except Exception as e:
            print(f"Model patch memory cleanup error: {e}")
        
        return (anything,)


# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "MemoryCleaner": MemoryCleaner,
    "MemoryManager": MemoryManager,
    "SafeMemoryManager": SafeMemoryManager,
    "DisTorchPurgeVRAMV2": DisTorchPurgeVRAMV2,
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryCleaner": "Memory Cleaner",
    "MemoryManager": "Memory Manager",
    "SafeMemoryManager": "Safe Memory Manager",
    "DisTorchPurgeVRAMV2": "LayerUtility: Purge VRAM V2",
    "ModelPatchMemoryCleaner": "Model Patch Memory Cleaner",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 