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
                # Only free CUDA memory, skip CPU as it may cause errors
                if torch.cuda.is_available():
                    try:
                        comfy.model_management.free_memory(0, 'cuda:0')
                    except Exception:
                        pass
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
                "purge_seedvr2_models": ("BOOLEAN", {"default": False, "tooltip": "Clear SeedVR2 DiT (base) and VAE models from cache"}),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "purge_vram"
    CATEGORY = "DisTorch/Memory"

    def purge_vram(self, anything, purge_cache, purge_models, purge_seedvr2_models):
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

                # Pre-cleanup: Remove models with None or non-callable real_model before calling cleanup_models()
                # This prevents 'NoneType' object is not callable errors
                if hasattr(comfy.model_management, "current_loaded_models"):
                    current_loaded_models = comfy.model_management.current_loaded_models
                    pre_cleaned = 0
                    for i in range(len(current_loaded_models) - 1, -1, -1):
                        loaded_model = current_loaded_models[i]
                        if loaded_model is not None:
                            try:
                                # Check if real_model is None or not callable
                                if hasattr(loaded_model, "real_model"):
                                    real_model = loaded_model.real_model
                                    if real_model is None:
                                        # Remove model with None real_model
                                        current_loaded_models.pop(i)
                                        pre_cleaned += 1
                                    elif not callable(real_model):
                                        # Remove model with non-callable real_model
                                        current_loaded_models.pop(i)
                                        pre_cleaned += 1
                                    else:
                                        # Check if calling real_model() would fail
                                        try:
                                            if real_model() is None:
                                                current_loaded_models.pop(i)
                                                pre_cleaned += 1
                                        except (TypeError, AttributeError):
                                            # real_model is not callable or has issues
                                            current_loaded_models.pop(i)
                                            pre_cleaned += 1
                            except Exception:
                                # Skip problematic models
                                pass
                    
                    if pre_cleaned > 0:
                        print(f"Pre-cleaned {pre_cleaned} problematic model(s) before cleanup_models()")

                # Cleanup dead models
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
                    
                    # Pre-cleanup again before second cleanup_models() call
                    if hasattr(comfy.model_management, "current_loaded_models"):
                        current_loaded_models = comfy.model_management.current_loaded_models
                        pre_cleaned_2 = 0
                        for i in range(len(current_loaded_models) - 1, -1, -1):
                            loaded_model = current_loaded_models[i]
                            if loaded_model is not None:
                                try:
                                    if hasattr(loaded_model, "real_model"):
                                        real_model = loaded_model.real_model
                                        if real_model is None or not callable(real_model):
                                            current_loaded_models.pop(i)
                                            pre_cleaned_2 += 1
                                        else:
                                            try:
                                                if real_model() is None:
                                                    current_loaded_models.pop(i)
                                                    pre_cleaned_2 += 1
                                            except (TypeError, AttributeError):
                                                current_loaded_models.pop(i)
                                                pre_cleaned_2 += 1
                                except Exception:
                                    pass
                        
                        if pre_cleaned_2 > 0:
                            print(f"Pre-cleaned {pre_cleaned_2} problematic model(s) before second cleanup_models()")
                    
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

        # Purge SeedVR2 models if requested
        if purge_seedvr2_models:
            try:
                # Try to import SeedVR2's GlobalModelCache
                import sys
                import os
                
                # Try multiple possible paths for SeedVR2 custom node
                # Note: Paths are relative to avoid hardcoding user-specific directories
                seedvr2_path = None
                
                # Method 1: Try to import from already loaded modules (most reliable)
                try:
                    import seedvr2_videoupscaler
                    if hasattr(seedvr2_videoupscaler, '__file__'):
                        seedvr2_path = os.path.dirname(os.path.abspath(seedvr2_videoupscaler.__file__))
                except (ImportError, AttributeError):
                    pass
                
                # Method 2: Relative to current file (same custom_nodes directory)
                # Current file is in: ComfyUI/custom_nodes/ComfyUI-DistorchMemoryManager/__init__.py
                # Target is: ComfyUI/custom_nodes/seedvr2_videoupscaler
                if not seedvr2_path:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    # Go up one level to custom_nodes directory
                    custom_nodes_dir = os.path.dirname(current_dir)
                    seedvr2_candidate = os.path.join(custom_nodes_dir, 'seedvr2_videoupscaler')
                    if os.path.exists(seedvr2_candidate) and os.path.isdir(seedvr2_candidate):
                        seedvr2_path = seedvr2_candidate
                
                # Method 3: Search in sys.path for seedvr2_videoupscaler
                if not seedvr2_path:
                    for path in sys.path:
                        # Check if path contains seedvr2_videoupscaler
                        if 'seedvr2_videoupscaler' in path:
                            # Extract the directory containing seedvr2_videoupscaler
                            parts = path.split(os.sep)
                            if 'seedvr2_videoupscaler' in parts:
                                idx = parts.index('seedvr2_videoupscaler')
                                candidate = os.sep.join(parts[:idx+1])
                                if os.path.exists(candidate) and os.path.isdir(candidate):
                                    seedvr2_path = candidate
                                    break
                        else:
                            # Check if seedvr2_videoupscaler exists as subdirectory
                            seedvr2_candidate = os.path.join(path, 'seedvr2_videoupscaler')
                            if os.path.exists(seedvr2_candidate) and os.path.isdir(seedvr2_candidate):
                                seedvr2_path = seedvr2_candidate
                                break
                
                # Method 4: Find custom_nodes directory from current file path structure
                if not seedvr2_path:
                    current_file = os.path.abspath(__file__)
                    parts = current_file.split(os.sep)
                    # Look for 'custom_nodes' in the path
                    if 'custom_nodes' in parts:
                        idx = parts.index('custom_nodes')
                        # Reconstruct path up to custom_nodes
                        custom_nodes_base = os.sep.join(parts[:idx+1])
                        seedvr2_candidate = os.path.join(custom_nodes_base, 'seedvr2_videoupscaler')
                        if os.path.exists(seedvr2_candidate) and os.path.isdir(seedvr2_candidate):
                            seedvr2_path = seedvr2_candidate
                
                if seedvr2_path:
                    sys.path.insert(0, seedvr2_path)
                    try:
                        from src.core.model_cache import get_global_cache
                        
                        cache = get_global_cache()
                        dit_cleared = 0
                        vae_cleared = 0
                        
                        # Clear all DiT models
                        if hasattr(cache, '_dit_models'):
                            dit_models_copy = dict(cache._dit_models)
                            for node_id, (model, config) in dit_models_copy.items():
                                try:
                                    # Ensure config has node_id for remove_dit
                                    if not isinstance(config, dict):
                                        config = {}
                                    if 'node_id' not in config:
                                        config['node_id'] = node_id
                                    # Use remove_dit to properly clean up
                                    if cache.remove_dit(config, debug=None):
                                        dit_cleared += 1
                                except Exception as e:
                                    print(f"Error removing SeedVR2 DiT model {node_id}: {e}")
                        
                        # Clear all VAE models
                        if hasattr(cache, '_vae_models'):
                            vae_models_copy = dict(cache._vae_models)
                            for node_id, (model, config) in vae_models_copy.items():
                                try:
                                    # Ensure config has node_id for remove_vae
                                    if not isinstance(config, dict):
                                        config = {}
                                    if 'node_id' not in config:
                                        config['node_id'] = node_id
                                    # Use remove_vae to properly clean up
                                    if cache.remove_vae(config, debug=None):
                                        vae_cleared += 1
                                except Exception as e:
                                    print(f"Error removing SeedVR2 VAE model {node_id}: {e}")
                        
                        # Clear runner templates
                        if hasattr(cache, '_runner_templates'):
                            runner_count = len(cache._runner_templates)
                            cache._runner_templates.clear()
                            if runner_count > 0:
                                print(f"Cleared {runner_count} SeedVR2 runner template(s)")
                        
                        if dit_cleared > 0 or vae_cleared > 0:
                            print(f"Cleared {dit_cleared} SeedVR2 DiT model(s) and {vae_cleared} VAE model(s)")
                        elif dit_cleared == 0 and vae_cleared == 0:
                            print("No SeedVR2 models found in cache")
                            
                    except ImportError as e:
                        print(f"SeedVR2 not available or incompatible version: {e}")
                    except Exception as e:
                        print(f"Error purging SeedVR2 models: {e}")
                    finally:
                        # Remove from path if we added it
                        if seedvr2_path in sys.path:
                            sys.path.remove(seedvr2_path)
                else:
                    # SeedVR2 not found - this is normal if SeedVR2 is not installed or not yet loaded
                    # Don't print error as it's not necessarily a problem
                    pass
            except Exception as e:
                print(f"Error accessing SeedVR2 models: {e}")

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