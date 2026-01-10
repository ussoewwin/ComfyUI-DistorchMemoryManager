import sys
import os
import torch
import gc

# Ensure ComfyUI root is on sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Memory Manager nodes (including any for ModelPatchMemoryCleaner)
try:
    from .memory_manager import MemoryManager, SafeMemoryManager, any
except ImportError:
    try:
        from memory_manager import MemoryManager, SafeMemoryManager, any
    except ImportError:
        MemoryManager = None
        SafeMemoryManager = None
        # Fallback: define any locally if import fails
class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""
    def __eq__(self, __value: object) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
    def __repr__(self):
        return str(self)
any = AnyType("*")


# Import Purge VRAM V2 node
try:
    from .purge_vram import DisTorchPurgeVRAMV2
except ImportError:
    try:
        from purge_vram import DisTorchPurgeVRAMV2
    except ImportError:
        DisTorchPurgeVRAMV2 = None


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


# Import SageAttention patch node
try:
    from .sa import PatchSageAttentionDM
except ImportError:
    try:
        from sa import PatchSageAttentionDM
    except ImportError:
        PatchSageAttentionDM = None


# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,
}

# Register Memory Manager nodes if available
if MemoryManager is not None:
    NODE_CLASS_MAPPINGS["MemoryManager"] = MemoryManager
if SafeMemoryManager is not None:
    NODE_CLASS_MAPPINGS["SafeMemoryManager"] = SafeMemoryManager

# Register Purge VRAM V2 node if available
if DisTorchPurgeVRAMV2 is not None:
    NODE_CLASS_MAPPINGS["DisTorchPurgeVRAMV2"] = DisTorchPurgeVRAMV2

# Register SageAttention node if available
if PatchSageAttentionDM is not None:
    NODE_CLASS_MAPPINGS["PatchSageAttentionDM"] = PatchSageAttentionDM

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPatchMemoryCleaner": "Model Patch Memory Cleaner",
}

# Register Memory Manager node display names if available
if MemoryManager is not None:
    NODE_DISPLAY_NAME_MAPPINGS["MemoryManager"] = "Memory Manager"
if SafeMemoryManager is not None:
    NODE_DISPLAY_NAME_MAPPINGS["SafeMemoryManager"] = "Safe Memory Manager"

# Register Purge VRAM V2 node display name if available
if DisTorchPurgeVRAMV2 is not None:
    NODE_DISPLAY_NAME_MAPPINGS["DisTorchPurgeVRAMV2"] = "LayerUtility: Purge VRAM V2"

# Register SageAttention node display name if available
if PatchSageAttentionDM is not None:
    NODE_DISPLAY_NAME_MAPPINGS["PatchSageAttentionDM"] = "Patch Sage Attention DM"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 
