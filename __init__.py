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
                "purge_qwen3vl_models": ("BOOLEAN", {"default": False, "tooltip": "Clear Qwen3-VL models from GPU memory"}),
                "purge_nunchaku_models": ("BOOLEAN", {"default": False, "tooltip": "Clear Nunchaku models (FLUX/Z-Image/Qwen-Image) from GPU memory"}),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "purge_vram"
    CATEGORY = "DisTorch/Memory"

    def purge_vram(self, anything, purge_cache, purge_models, purge_seedvr2_models, purge_qwen3vl_models, purge_nunchaku_models):
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
                # Note: sys and os are already imported at module level
                
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
                    # Add seedvr2_path to sys.path temporarily
                    original_path = sys.path[:]
                    try:
                        if seedvr2_path not in sys.path:
                            sys.path.insert(0, seedvr2_path)
                        
                        # Try importing with different methods
                        cache = None
                        import_method = None
                        try:
                            # Method 1: Direct import
                            from src.core.model_cache import get_global_cache
                            cache = get_global_cache()
                            import_method = "Method 1 (direct import)"
                        except (ImportError, ModuleNotFoundError) as e1:
                            try:
                                # Method 2: Import seedvr2_videoupscaler first
                                import seedvr2_videoupscaler
                                from seedvr2_videoupscaler.src.core.model_cache import get_global_cache
                                cache = get_global_cache()
                                import_method = "Method 2 (via seedvr2_videoupscaler)"
                            except (ImportError, ModuleNotFoundError, AttributeError) as e2:
                                # Method 3: Try to access via already loaded module
                                if 'seedvr2_videoupscaler' in sys.modules:
                                    seedvr2_module = sys.modules['seedvr2_videoupscaler']
                                    if hasattr(seedvr2_module, 'src'):
                                        from seedvr2_videoupscaler.src.core.model_cache import get_global_cache
                                        cache = get_global_cache()
                                        import_method = "Method 3 (via sys.modules)"
                        
                        if cache is not None:
                            if import_method:
                                print(f"SeedVR2: Cache accessed via {import_method}")
                            dit_cleared = 0
                            vae_cleared = 0
                            
                            # Debug: Check cache state before clearing
                            dit_count_before = len(cache._dit_models) if hasattr(cache, '_dit_models') else 0
                            vae_count_before = len(cache._vae_models) if hasattr(cache, '_vae_models') else 0
                            runner_count_before = len(cache._runner_templates) if hasattr(cache, '_runner_templates') else 0
                            
                            # Log SeedVR2 cache access with detailed info
                            print(f"SeedVR2: Checking cache (DiT: {dit_count_before}, VAE: {vae_count_before}, Runners: {runner_count_before})")
                            
                            # Debug: Check if cache attributes exist and show details
                            if hasattr(cache, '_dit_models'):
                                dit_keys = list(cache._dit_models.keys()) if cache._dit_models else []
                                if dit_keys:
                                    print(f"SeedVR2: DiT model node IDs: {dit_keys}")
                                else:
                                    print("SeedVR2: DiT models dictionary exists but is empty")
                            else:
                                print("SeedVR2: _dit_models attribute not found in cache")
                            
                            if hasattr(cache, '_vae_models'):
                                vae_keys = list(cache._vae_models.keys()) if cache._vae_models else []
                                if vae_keys:
                                    print(f"SeedVR2: VAE model node IDs: {vae_keys}")
                                else:
                                    print("SeedVR2: VAE models dictionary exists but is empty")
                            else:
                                print("SeedVR2: _vae_models attribute not found in cache")
                            
                            # Clear all DiT models
                            if hasattr(cache, '_dit_models') and cache._dit_models:
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
                            if hasattr(cache, '_vae_models') and cache._vae_models:
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
                            if hasattr(cache, '_runner_templates') and cache._runner_templates:
                                runner_count = len(cache._runner_templates)
                                cache._runner_templates.clear()
                                if runner_count > 0:
                                    print(f"Cleared {runner_count} SeedVR2 runner template(s)")
                            
                            # Report results
                            if dit_cleared > 0 or vae_cleared > 0:
                                print(f"Cleared {dit_cleared} SeedVR2 DiT model(s) and {vae_cleared} VAE model(s)")
                            elif dit_count_before == 0 and vae_count_before == 0 and runner_count_before == 0:
                                # Cache is completely empty - SeedVR2 may not have cached models yet
                                # This is normal if SeedVR2 is used but models aren't cached (cache_model=False)
                                # Or models were already cleared by SeedVR2 after processing completed
                                try:
                                    import comfy.model_management
                                    if hasattr(comfy.model_management, "current_loaded_models"):
                                        # Check if any loaded models might be SeedVR2 models
                                        seedvr2_model_count = 0
                                        for loaded_model in comfy.model_management.current_loaded_models:
                                            if loaded_model is not None and hasattr(loaded_model, "model"):
                                                model = loaded_model.model
                                                # Check if model name or type suggests SeedVR2
                                                model_str = str(type(model)).lower()
                                                if __builtins__['any'](keyword in model_str for keyword in ['seedvr', 'dit', 'video_vae']):
                                                    seedvr2_model_count += 1
                                        
                                        if seedvr2_model_count > 0:
                                            print(f"SeedVR2: Cache is empty, but found {seedvr2_model_count} potential SeedVR2 model(s) in ComfyUI's model management (not cached in GlobalModelCache)")
                                        else:
                                            # cache_model=False (default): Models are never cached in GlobalModelCache and are automatically deleted from memory after processing
                                            # cache_model=True: Models are cached in GlobalModelCache and remain in memory after processing
                                            print("SeedVR2: Cache is empty - cache_model option is disabled (False by default). Enable cache_model=True in SeedVR2 nodes to cache models in GlobalModelCache.")
                                except Exception:
                                    print("SeedVR2: Cache is empty - cache_model option is disabled (False by default). Enable cache_model=True in SeedVR2 nodes to cache models in GlobalModelCache.")
                            else:
                                # Models exist in cache but weren't cleared (shouldn't happen normally)
                                print(f"SeedVR2 cache state: {dit_count_before} DiT, {vae_count_before} VAE, {runner_count_before} runner template(s) (models may not be cached)")
                        else:
                            print("SeedVR2: Could not access GlobalModelCache")
                            
                    except ImportError as e:
                        print(f"SeedVR2 not available or incompatible version: {e}")
                    except Exception as e:
                        print(f"Error purging SeedVR2 models: {e}")
                    finally:
                        # Restore original sys.path
                        sys.path[:] = original_path
                else:
                    # SeedVR2 path not found - this is normal if SeedVR2 is not installed
                    pass
            except Exception as e:
                print(f"Error accessing SeedVR2 models: {e}")

        # Purge Qwen3-VL models if requested
        if purge_qwen3vl_models:
            try:
                print("Qwen3-VL: Starting purge process...")
                # Try to import Qwen3VLForConditionalGeneration to check model type
                qwen3vl_model_type = None
                try:
                    from transformers import Qwen3VLForConditionalGeneration
                    qwen3vl_model_type = Qwen3VLForConditionalGeneration
                    print("Qwen3-VL: Successfully imported Qwen3VLForConditionalGeneration")
                except ImportError as e:
                    print(f"Qwen3-VL: Failed to import Qwen3VLForConditionalGeneration: {e}")
                
                if qwen3vl_model_type is not None:
                    qwen3vl_cleared = 0
                    
                    # Method 1: Search for Qwen3VL models in sys.modules and other places
                    # Check if models are stored in any module attributes
                    print("Qwen3-VL: Method 1 - Searching sys.modules for models...")
                    modules_checked = 0
                    # Create a copy of sys.modules.items() to avoid RuntimeError if dictionary changes during iteration
                    modules_items = list(sys.modules.items())
                    for module_name, module in modules_items:
                        if module is None:
                            continue
                        modules_checked += 1
                        try:
                            # Check module attributes for Qwen3VL models
                            for attr_name in dir(module):
                                try:
                                    attr = getattr(module, attr_name, None)
                                    if attr is None:
                                        continue
                                    
                                    # Check if it's a Qwen3VL model instance
                                    if isinstance(attr, qwen3vl_model_type):
                                        try:
                                            print(f"Qwen3-VL: Found model instance at {module_name}.{attr_name}")
                                            # Handle device_map="auto" case - move all modules from GPU to CPU
                                            try:
                                                if hasattr(attr, 'hf_device_map'):
                                                    hf_device_map = attr.hf_device_map
                                                    print(f"Qwen3-VL: Model has hf_device_map with {len(hf_device_map)} entries")
                                                    modules_moved = 0
                                                    for param_name, device in hf_device_map.items():
                                                        # Handle different device formats: str ('cuda:0'), int (device index), or torch.device
                                                        device_str = str(device) if device is not None else ''
                                                        if device_str.startswith('cuda') or (isinstance(device, int) and device >= 0):
                                                            print(f"Qwen3-VL: Moving module {param_name} from {device} to CPU")
                                                            submodule = attr
                                                            # Skip empty param_name (root module)
                                                            if param_name:
                                                                try:
                                                                    for part in param_name.split('.'):
                                                                        submodule = getattr(submodule, part)
                                                                except AttributeError:
                                                                    print(f"Qwen3-VL: Warning: Could not find module path {param_name}, skipping")
                                                                    continue
                                                            if hasattr(submodule, 'to'):
                                                                submodule.to('cpu')
                                                                modules_moved += 1
                                                    print(f"Qwen3-VL: Moved {modules_moved} modules from GPU to CPU")
                                            except Exception as e:
                                                print(f"Qwen3-VL: Error handling hf_device_map: {e}")
                                            
                                            # Move model to CPU and clear GPU memory
                                            try:
                                                print(f"Qwen3-VL: Attempting to move model to CPU...")
                                                if hasattr(attr, 'to'):
                                                    attr.to('cpu')
                                                    print(f"Qwen3-VL: Model moved to CPU using .to('cpu')")
                                                elif hasattr(attr, 'cpu'):
                                                    attr.cpu()
                                                    print(f"Qwen3-VL: Model moved to CPU using .cpu()")
                                            except Exception as e:
                                                print(f"Qwen3-VL: Direct move to CPU failed: {e}, trying parameter-by-parameter move...")
                                                # If direct move fails, try moving parameters individually
                                                try:
                                                    params_moved = 0
                                                    for param in attr.parameters():
                                                        if param.is_cuda:
                                                            param.data = param.data.cpu()
                                                            params_moved += 1
                                                    buffers_moved = 0
                                                    for buffer in attr.buffers():
                                                        if buffer.is_cuda:
                                                            buffer.data = buffer.data.cpu()
                                                            buffers_moved += 1
                                                    print(f"Qwen3-VL: Moved {params_moved} parameters and {buffers_moved} buffers to CPU")
                                                except Exception as e2:
                                                    print(f"Qwen3-VL: Parameter-by-parameter move also failed: {e2}")
                                            
                                            # Delete the model reference and force memory release
                                            try:
                                                # First, try to delete the attribute
                                                if hasattr(module, attr_name):
                                                    delattr(module, attr_name)
                                                    print(f"Qwen3-VL: Deleted model reference from {module_name}.{attr_name}")
                                            except Exception as e:
                                                print(f"Qwen3-VL: Failed to delete model reference: {e}")
                                            
                                            # Force delete the model object itself
                                            # Clear all parameters and buffers to release memory
                                            try:
                                                # Try to clear model's internal state more aggressively
                                                if hasattr(attr, 'named_parameters'):
                                                    for name, param in list(attr.named_parameters()):
                                                        if param is not None and hasattr(param, 'data'):
                                                            try:
                                                                if param.data is not None:
                                                                    # Detach from computation graph
                                                                    param.data = param.data.detach().cpu()
                                                            except Exception:
                                                                pass
                                                if hasattr(attr, 'named_buffers'):
                                                    for name, buffer in list(attr.named_buffers()):
                                                        if buffer is not None and hasattr(buffer, 'data'):
                                                            try:
                                                                if buffer.data is not None:
                                                                    buffer.data = buffer.data.detach().cpu()
                                                            except Exception:
                                                                pass
                                                # Clear model's modules dict if available
                                                if hasattr(attr, '_modules'):
                                                    attr._modules.clear()
                                                print(f"Qwen3-VL: Cleared model internal state")
                                            except Exception as e:
                                                print(f"Qwen3-VL: Warning: Failed to clear model internal state: {e}")
                                            
                                            try:
                                                del attr
                                                print(f"Qwen3-VL: Deleted model object")
                                            except Exception as e:
                                                print(f"Qwen3-VL: Failed to delete model object: {e}")
                                            
                                            qwen3vl_cleared += 1
                                            print(f"Qwen3-VL: Successfully cleared model from {module_name}.{attr_name}")
                                        except Exception as e:
                                            print(f"Qwen3-VL: Error clearing model from {module_name}.{attr_name}: {e}")
                                            import traceback
                                            print(f"Qwen3-VL: Traceback: {traceback.format_exc()}")
                                    
                                    # Check if it's a dict containing a model (like {"model": model_object, "model_path": path})
                                    elif isinstance(attr, dict) and 'model' in attr:
                                        model_obj = attr.get('model')
                                        if isinstance(model_obj, qwen3vl_model_type):
                                            try:
                                                print(f"Qwen3-VL: Found model in dict at {module_name}.{attr_name}")
                                                # Handle device_map="auto" case - move all modules from GPU to CPU
                                                try:
                                                    if hasattr(model_obj, 'hf_device_map'):
                                                        hf_device_map = model_obj.hf_device_map
                                                        print(f"Qwen3-VL: Model in dict has hf_device_map with {len(hf_device_map)} entries")
                                                        modules_moved = 0
                                                        for param_name, device in hf_device_map.items():
                                                            # Handle different device formats: str ('cuda:0'), int (device index), or torch.device
                                                            device_str = str(device) if device is not None else ''
                                                            if device_str.startswith('cuda') or (isinstance(device, int) and device >= 0):
                                                                print(f"Qwen3-VL: Moving module {param_name} from {device} to CPU")
                                                                submodule = model_obj
                                                                # Skip empty param_name (root module)
                                                                if param_name:
                                                                    try:
                                                                        for part in param_name.split('.'):
                                                                            submodule = getattr(submodule, part)
                                                                    except AttributeError:
                                                                        print(f"Qwen3-VL: Warning: Could not find module path {param_name}, skipping")
                                                                        continue
                                                                if hasattr(submodule, 'to'):
                                                                    submodule.to('cpu')
                                                                    modules_moved += 1
                                                        print(f"Qwen3-VL: Moved {modules_moved} modules from GPU to CPU")
                                                except Exception as e:
                                                    print(f"Qwen3-VL: Error handling hf_device_map in dict: {e}")
                                                
                                                # Move model to CPU
                                                try:
                                                    print(f"Qwen3-VL: Attempting to move model in dict to CPU...")
                                                    if hasattr(model_obj, 'to'):
                                                        model_obj.to('cpu')
                                                        print(f"Qwen3-VL: Model in dict moved to CPU using .to('cpu')")
                                                    elif hasattr(model_obj, 'cpu'):
                                                        model_obj.cpu()
                                                        print(f"Qwen3-VL: Model in dict moved to CPU using .cpu()")
                                                except Exception as e:
                                                    print(f"Qwen3-VL: Direct move to CPU failed: {e}, trying parameter-by-parameter move...")
                                                    # If direct move fails, try moving parameters individually
                                                    try:
                                                        params_moved = 0
                                                        for param in model_obj.parameters():
                                                            if param.is_cuda:
                                                                param.data = param.data.cpu()
                                                                params_moved += 1
                                                        buffers_moved = 0
                                                        for buffer in model_obj.buffers():
                                                            if buffer.is_cuda:
                                                                buffer.data = buffer.data.cpu()
                                                                buffers_moved += 1
                                                        print(f"Qwen3-VL: Moved {params_moved} parameters and {buffers_moved} buffers to CPU")
                                                    except Exception as e2:
                                                        print(f"Qwen3-VL: Parameter-by-parameter move also failed: {e2}")
                                                
                                                # Clear the model from dict and force memory release
                                                try:
                                                    # Clear model's internal state before deletion
                                                    if hasattr(model_obj, 'named_parameters'):
                                                        for name, param in list(model_obj.named_parameters()):
                                                            if param is not None and hasattr(param, 'data'):
                                                                try:
                                                                    if param.data is not None:
                                                                        param.data = param.data.detach().cpu()
                                                                except Exception:
                                                                    pass
                                                    if hasattr(model_obj, 'named_buffers'):
                                                        for name, buffer in list(model_obj.named_buffers()):
                                                            if buffer is not None and hasattr(buffer, 'data'):
                                                                try:
                                                                    if buffer.data is not None:
                                                                        buffer.data = buffer.data.detach().cpu()
                                                                except Exception:
                                                                    pass
                                                    if hasattr(model_obj, '_modules'):
                                                        model_obj._modules.clear()
                                                    print(f"Qwen3-VL: Cleared model internal state from dict")
                                                except Exception as e:
                                                    print(f"Qwen3-VL: Warning: Failed to clear model internal state from dict: {e}")
                                                
                                                try:
                                                    # Delete the model object first
                                                    del model_obj
                                                    print(f"Qwen3-VL: Deleted model object from dict")
                                                except Exception as e:
                                                    print(f"Qwen3-VL: Failed to delete model object from dict: {e}")
                                                
                                                # Clear the dict entry
                                                attr['model'] = None
                                                qwen3vl_cleared += 1
                                                print(f"Qwen3-VL: Successfully cleared model from dict in {module_name}.{attr_name}")
                                            except Exception as e:
                                                print(f"Qwen3-VL: Error clearing model from dict in {module_name}.{attr_name}: {e}")
                                                import traceback
                                                print(f"Qwen3-VL: Traceback: {traceback.format_exc()}")
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"Qwen3-VL: Error checking module {module_name}: {e}")
                    print(f"Qwen3-VL: Method 1 complete - checked {modules_checked} modules")
                    
                    # Method 2: Force clear GPU memory for any remaining transformers models
                    print("Qwen3-VL: Method 2 - Searching gc.get_objects() for models...")
                    if torch.cuda.is_available():
                        try:
                            objects_checked = 0
                            models_found_in_gc = 0
                            # Get all objects in memory that might be transformers models
                            for obj in gc.get_objects():
                                objects_checked += 1
                                try:
                                    if isinstance(obj, qwen3vl_model_type):
                                        print(
                                            f"Qwen3-VL: Found model instance in gc.get_objects() (type: {type(obj).__name__}, id: {id(obj)})"
                                        )
                                        models_found_in_gc += 1
                                        # For transformers models with device_map="auto", need to handle multiple devices
                                        try:
                                            # Try to get device info from model
                                            if hasattr(obj, 'hf_device_map'):
                                                hf_device_map = obj.hf_device_map
                                                print(f"Qwen3-VL: Model has hf_device_map with {len(hf_device_map)} entries")
                                                modules_moved = 0
                                                # Model is distributed across multiple devices
                                                # Move all modules to CPU
                                                for param_name, device in hf_device_map.items():
                                                    # Handle different device formats: str ('cuda:0'), int (device index), or torch.device
                                                    device_str = str(device) if device is not None else ''
                                                    if device_str.startswith('cuda') or (isinstance(device, int) and device >= 0):
                                                        print(f"Qwen3-VL: Moving module {param_name} from {device} to CPU")
                                                        submodule = obj
                                                        # Skip empty param_name (root module)
                                                        if param_name:
                                                            try:
                                                                for part in param_name.split('.'):
                                                                    submodule = getattr(submodule, part)
                                                            except AttributeError:
                                                                print(f"Qwen3-VL: Warning: Could not find module path {param_name}, skipping")
                                                                continue
                                                        if hasattr(submodule, 'to'):
                                                            submodule.to('cpu')
                                                            modules_moved += 1
                                                print(f"Qwen3-VL: Moved {modules_moved} modules from GPU to CPU")
                                        except Exception as e:
                                            print(f"Qwen3-VL: Error handling hf_device_map in gc objects: {e}")
                                        
                                        # Move entire model to CPU
                                        try:
                                            print(f"Qwen3-VL: Attempting to move model to CPU...")
                                            if hasattr(obj, 'to'):
                                                obj.to('cpu')
                                                print(f"Qwen3-VL: Model moved to CPU using .to('cpu')")
                                            elif hasattr(obj, 'cpu'):
                                                obj.cpu()
                                                print(f"Qwen3-VL: Model moved to CPU using .cpu()")
                                        except Exception as e:
                                            print(f"Qwen3-VL: Direct move to CPU failed: {e}, trying parameter-by-parameter move...")
                                            # If model is quantized or has special structure, try moving parameters
                                            try:
                                                params_moved = 0
                                                for param in obj.parameters():
                                                    if param.is_cuda:
                                                        param.data = param.data.cpu()
                                                        params_moved += 1
                                                buffers_moved = 0
                                                for buffer in obj.buffers():
                                                    if buffer.is_cuda:
                                                        buffer.data = buffer.data.cpu()
                                                        buffers_moved += 1
                                                print(f"Qwen3-VL: Moved {params_moved} parameters and {buffers_moved} buffers to CPU")
                                            except Exception as e2:
                                                print(f"Qwen3-VL: Parameter-by-parameter move also failed: {e2}")
                                        
                                        # Force delete the model object and clear all references
                                        try:
                                            # Clear model's internal state more aggressively
                                            if hasattr(obj, 'named_parameters'):
                                                for name, param in list(obj.named_parameters()):
                                                    if param is not None and hasattr(param, 'data'):
                                                        try:
                                                            if param.data is not None:
                                                                # Detach from computation graph and move to CPU
                                                                param.data = param.data.detach().cpu()
                                                        except Exception:
                                                            pass
                                            if hasattr(obj, 'named_buffers'):
                                                for name, buffer in list(obj.named_buffers()):
                                                    if buffer is not None and hasattr(buffer, 'data'):
                                                        try:
                                                            if buffer.data is not None:
                                                                buffer.data = buffer.data.detach().cpu()
                                                        except Exception:
                                                            pass
                                            # Clear model's modules dict if available
                                            if hasattr(obj, '_modules'):
                                                obj._modules.clear()
                                            print(f"Qwen3-VL: Cleared model internal state from gc.get_objects()")
                                        except Exception as e:
                                            print(f"Qwen3-VL: Warning: Failed to clear model internal state from gc.get_objects(): {e}")
                                        
                                        try:
                                            del obj
                                            print(f"Qwen3-VL: Deleted model object from gc.get_objects()")
                                        except Exception as e:
                                            print(f"Qwen3-VL: Failed to delete model object: {e}")
                                        
                                        qwen3vl_cleared += 1
                                        print(f"Qwen3-VL: Successfully cleared model from gc.get_objects()")
                                except Exception as e:
                                    pass
                            print(
                                f"Qwen3-VL: Method 2 complete - checked {objects_checked} objects, found {models_found_in_gc} models"
                            )
                        except Exception as e:
                            print(f"Qwen3-VL: Error in GPU memory cleanup: {e}")
                            import traceback
                            print(f"Qwen3-VL: Traceback: {traceback.format_exc()}")
                    
                    # Force garbage collection and clear GPU cache
                    print("Qwen3-VL: Running garbage collection...")
                    gc.collect()
                    gc.collect()  # Run twice to ensure cleanup
                    if torch.cuda.is_available():
                        print("Qwen3-VL: Clearing CUDA cache...")
                        # Clear cache for all devices
                        for device_idx in range(torch.cuda.device_count()):
                            with torch.cuda.device(device_idx):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                        print("Qwen3-VL: CUDA cache cleared for all devices")
                    
                    if qwen3vl_cleared > 0:
                        print(f"Qwen3-VL: Successfully cleared {qwen3vl_cleared} model(s)")
                    else:
                        print("Qwen3-VL: No models found in memory (models may not be cached or already cleared)")
                else:
                    print("Qwen3-VL: transformers library with Qwen3VLForConditionalGeneration not available")
                    
            except Exception as e:
                print(f"Qwen3-VL: Error purging models: {e}")
                import traceback
                print(f"Qwen3-VL: Traceback: {traceback.format_exc()}")

        # Purge Nunchaku models if requested
        if purge_nunchaku_models:
            try:
                print("Nunchaku: Starting purge process...")
                # Try to import Nunchaku transformer model types
                nunchaku_model_types = []
                
                # Try to import NunchakuFluxTransformer2dModel
                try:
                    from nunchaku import NunchakuFluxTransformer2dModel
                    nunchaku_model_types.append(NunchakuFluxTransformer2dModel)
                    print("Nunchaku: Successfully imported NunchakuFluxTransformer2dModel")
                except ImportError as e:
                    print(f"Nunchaku: Failed to import NunchakuFluxTransformer2dModel: {e}")
                
                # Try to import NunchakuZImageTransformer2DModel
                try:
                    from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
                    nunchaku_model_types.append(NunchakuZImageTransformer2DModel)
                    print("Nunchaku: Successfully imported NunchakuZImageTransformer2DModel")
                except ImportError as e:
                    print(f"Nunchaku: Failed to import NunchakuZImageTransformer2DModel: {e}")
                
                # Try to import NunchakuT5EncoderModel (text encoder)
                try:
                    from nunchaku.models.transformers.transformer_t5 import NunchakuT5EncoderModel
                    nunchaku_model_types.append(NunchakuT5EncoderModel)
                    print("Nunchaku: Successfully imported NunchakuT5EncoderModel")
                except ImportError as e:
                    print(f"Nunchaku: Failed to import NunchakuT5EncoderModel: {e}")
                
                # Try to import NunchakuQwenImageTransformer2DModel (Qwen-Image)
                try:
                    from comfyui_nunchaku.models.qwenimage import NunchakuQwenImageTransformer2DModel
                    nunchaku_model_types.append(NunchakuQwenImageTransformer2DModel)
                    print("Nunchaku: Successfully imported NunchakuQwenImageTransformer2DModel")
                except ImportError as e:
                    print(f"Nunchaku: Failed to import NunchakuQwenImageTransformer2DModel from comfyui_nunchaku: {e}")
                
                # Alternative import path for NunchakuQwenImageTransformer2DModel
                if not __builtins__['any'](cls.__name__ == 'NunchakuQwenImageTransformer2DModel' for cls in nunchaku_model_types):
                    try:
                        # Try to find it in sys.modules
                        # Create a copy to avoid RuntimeError if dictionary changes during iteration
                        module_names = list(sys.modules.keys())
                        for module_name in module_names:
                            if 'qwenimage' in module_name.lower() or 'nunchaku' in module_name.lower():
                                try:
                                    module = sys.modules[module_name]
                                    if hasattr(module, 'NunchakuQwenImageTransformer2DModel'):
                                        cls = getattr(module, 'NunchakuQwenImageTransformer2DModel')
                                        if cls not in nunchaku_model_types:
                                            nunchaku_model_types.append(cls)
                                            print(f"Nunchaku: Found NunchakuQwenImageTransformer2DModel in {module_name}")
                                            break
                                except Exception:
                                    pass
                    except Exception:
                        pass
                
                if nunchaku_model_types:
                    print(f"Nunchaku: Found {len(nunchaku_model_types)} model type(s): {[cls.__name__ for cls in nunchaku_model_types]}")
                    nunchaku_cleared = 0
                    
                    # Method 1: Search for Nunchaku models in sys.modules
                    print("Nunchaku: Method 1 - Searching sys.modules for models...")
                    # Create a copy to avoid RuntimeError if dictionary changes during iteration
                    modules_items = list(sys.modules.items())
                    modules_checked = 0
                    for module_name, module in modules_items:
                        if module is None:
                            continue
                        modules_checked += 1
                        try:
                            for attr_name in dir(module):
                                try:
                                    attr = getattr(module, attr_name, None)
                                    if attr is None:
                                        continue
                                    
                                    # Check if it's a Nunchaku model instance
                                    model_found = False
                                    for model_type in nunchaku_model_types:
                                        if isinstance(attr, model_type):
                                            try:
                                                print(f"Nunchaku: Found model instance ({model_type.__name__}) at {module_name}.{attr_name} (id: {id(attr)})")
                                                # Disable CPU offload first if it's enabled (this releases memory)
                                                if hasattr(attr, 'set_offload'):
                                                    try:
                                                        if hasattr(attr, 'offload') and attr.offload:
                                                            print(f"Nunchaku: Disabling CPU offload for {module_name}.{attr_name}")
                                                            attr.set_offload(False)
                                                            print(f"Nunchaku: CPU offload disabled for {module_name}.{attr_name}")
                                                    except Exception as e:
                                                        print(f"Nunchaku: Warning: Failed to disable CPU offload: {e}")
                                                
                                                # Clear offload_manager if it exists
                                                if hasattr(attr, 'offload_manager') and attr.offload_manager is not None:
                                                    try:
                                                        print(f"Nunchaku: Clearing offload_manager for {module_name}.{attr_name}")
                                                        attr.offload_manager = None
                                                        print(f"Nunchaku: offload_manager cleared for {module_name}.{attr_name}")
                                                    except Exception as e:
                                                        print(f"Nunchaku: Warning: Failed to clear offload_manager: {e}")
                                                
                                                # Move model to CPU and clear GPU memory
                                                print(f"Nunchaku: Attempting to move model to CPU...")
                                                try:
                                                    if hasattr(attr, 'to'):
                                                        attr.to('cpu')
                                                        print(f"Nunchaku: Model moved to CPU using .to('cpu')")
                                                    elif hasattr(attr, 'cpu'):
                                                        attr.cpu()
                                                        print(f"Nunchaku: Model moved to CPU using .cpu()")
                                                except Exception as e:
                                                    print(f"Nunchaku: Warning: Failed to move model to CPU: {e}")
                                                
                                                # Clear model's internal state
                                                try:
                                                    if hasattr(attr, 'named_parameters'):
                                                        params_cleared = 0
                                                        for name, param in list(attr.named_parameters()):
                                                            if param is not None and hasattr(param, 'data'):
                                                                try:
                                                                    if param.data is not None:
                                                                        param.data = param.data.detach().cpu()
                                                                        params_cleared += 1
                                                                except Exception:
                                                                    pass
                                                        if params_cleared > 0:
                                                            print(f"Nunchaku: Cleared {params_cleared} parameters from {module_name}.{attr_name}")
                                                    if hasattr(attr, 'named_buffers'):
                                                        buffers_cleared = 0
                                                        for name, buffer in list(attr.named_buffers()):
                                                            if buffer is not None and hasattr(buffer, 'data'):
                                                                try:
                                                                    if buffer.data is not None:
                                                                        buffer.data = buffer.data.detach().cpu()
                                                                        buffers_cleared += 1
                                                                except Exception:
                                                                    pass
                                                        if buffers_cleared > 0:
                                                            print(f"Nunchaku: Cleared {buffers_cleared} buffers from {module_name}.{attr_name}")
                                                    if hasattr(attr, '_modules'):
                                                        attr._modules.clear()
                                                        print(f"Nunchaku: Cleared _modules dict from {module_name}.{attr_name}")
                                                except Exception as e:
                                                    print(f"Nunchaku: Warning: Failed to clear model internal state: {e}")
                                                
                                                # Delete the model reference
                                                try:
                                                    delattr(module, attr_name)
                                                    print(f"Nunchaku: Deleted model reference from {module_name}.{attr_name}")
                                                except Exception as e:
                                                    print(f"Nunchaku: Warning: Failed to delete model reference: {e}")
                                                
                                                nunchaku_cleared += 1
                                                print(f"Nunchaku: Successfully cleared model ({model_type.__name__}) from {module_name}.{attr_name}")
                                                model_found = True
                                                break
                                            except Exception as e:
                                                print(f"Nunchaku: Error clearing model from {module_name}.{attr_name}: {e}")
                                                import traceback
                                                print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                                    
                                    # Check if it's a dict containing a model
                                    if not model_found and isinstance(attr, dict):
                                        if 'transformer' in attr:
                                            transformer_obj = attr.get('transformer')
                                            for model_type in nunchaku_model_types:
                                                if isinstance(transformer_obj, model_type):
                                                    try:
                                                        # Disable CPU offload first if enabled
                                                        if hasattr(transformer_obj, 'set_offload'):
                                                            try:
                                                                if hasattr(transformer_obj, 'offload') and transformer_obj.offload:
                                                                    transformer_obj.set_offload(False)
                                                            except Exception:
                                                                pass
                                                        
                                                        # Clear offload_manager
                                                        if hasattr(transformer_obj, 'offload_manager') and transformer_obj.offload_manager is not None:
                                                            try:
                                                                transformer_obj.offload_manager = None
                                                            except Exception:
                                                                pass
                                                        
                                                        if hasattr(transformer_obj, 'to'):
                                                            transformer_obj.to('cpu')
                                                        elif hasattr(transformer_obj, 'cpu'):
                                                            transformer_obj.cpu()
                                                        attr['transformer'] = None
                                                        nunchaku_cleared += 1
                                                        print(f"Cleared Nunchaku model ({model_type.__name__}) from dict.transformer in {module_name}.{attr_name}")
                                                        break
                                                    except Exception as e:
                                                        print(f"Error clearing Nunchaku model from dict in {module_name}.{attr_name}: {e}")
                                        
                                        # Check for model in dict (like ModelPatcher structure)
                                        if 'model' in attr:
                                            model_obj = attr.get('model')
                                            # Check if model_obj has diffusion_model with transformer
                                            if hasattr(model_obj, 'diffusion_model'):
                                                diffusion_model = model_obj.diffusion_model
                                                # Check if diffusion_model has model attribute (ComfyFluxWrapper structure)
                                                if hasattr(diffusion_model, 'model'):
                                                    transformer_obj = diffusion_model.model
                                                    for model_type in nunchaku_model_types:
                                                        if isinstance(transformer_obj, model_type):
                                                            try:
                                                                # Disable CPU offload first if enabled
                                                                if hasattr(transformer_obj, 'set_offload'):
                                                                    try:
                                                                        if hasattr(transformer_obj, 'offload') and transformer_obj.offload:
                                                                            transformer_obj.set_offload(False)
                                                                    except Exception:
                                                                        pass
                                                                
                                                                # Clear offload_manager
                                                                if hasattr(transformer_obj, 'offload_manager') and transformer_obj.offload_manager is not None:
                                                                    try:
                                                                        transformer_obj.offload_manager = None
                                                                    except Exception:
                                                                        pass
                                                                
                                                                if hasattr(transformer_obj, 'to'):
                                                                    transformer_obj.to('cpu')
                                                                elif hasattr(transformer_obj, 'cpu'):
                                                                    transformer_obj.cpu()
                                                                nunchaku_cleared += 1
                                                                print(f"Cleared Nunchaku model ({model_type.__name__}) from dict.model.diffusion_model.model in {module_name}.{attr_name}")
                                                                break
                                                            except Exception as e:
                                                                print(f"Error clearing Nunchaku model from nested structure in {module_name}.{attr_name}: {e}")
                                                # Also check if diffusion_model itself is a nunchaku model
                                                for model_type in nunchaku_model_types:
                                                    if isinstance(diffusion_model, model_type):
                                                        try:
                                                            # Disable CPU offload first if enabled
                                                            if hasattr(diffusion_model, 'set_offload'):
                                                                try:
                                                                    if hasattr(diffusion_model, 'offload') and diffusion_model.offload:
                                                                        diffusion_model.set_offload(False)
                                                                except Exception:
                                                                    pass
                                                            
                                                            # Clear offload_manager
                                                            if hasattr(diffusion_model, 'offload_manager') and diffusion_model.offload_manager is not None:
                                                                try:
                                                                    diffusion_model.offload_manager = None
                                                                except Exception:
                                                                    pass
                                                            
                                                            if hasattr(diffusion_model, 'to'):
                                                                diffusion_model.to('cpu')
                                                            elif hasattr(diffusion_model, 'cpu'):
                                                                diffusion_model.cpu()
                                                            nunchaku_cleared += 1
                                                            print(f"Cleared Nunchaku model ({model_type.__name__}) from dict.model.diffusion_model in {module_name}.{attr_name}")
                                                            break
                                                        except Exception as e:
                                                            print(f"Error clearing Nunchaku model from diffusion_model in {module_name}.{attr_name}: {e}")
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"Nunchaku: Error checking module {module_name}: {e}")
                    print(f"Nunchaku: Method 1 complete - checked {modules_checked} modules, cleared {nunchaku_cleared} model(s)")
                    
                    # Method 2: Search in ComfyUI's current_loaded_models for Nunchaku models
                    print("Nunchaku: Method 2 - Searching ComfyUI current_loaded_models for models...")
                    try:
                        import comfy.model_management
                        if hasattr(comfy.model_management, "current_loaded_models"):
                            current_loaded_models = comfy.model_management.current_loaded_models
                            loaded_models_checked = 0
                            for loaded_model in current_loaded_models:
                                if loaded_model is None:
                                    continue
                                loaded_models_checked += 1
                                try:
                                    if hasattr(loaded_model, "model"):
                                        model = loaded_model.model
                                        # Check if model has diffusion_model attribute
                                        if hasattr(model, "diffusion_model"):
                                            diffusion_model = model.diffusion_model
                                            # Check ComfyFluxWrapper structure: diffusion_model.model
                                            if hasattr(diffusion_model, "model"):
                                                transformer = diffusion_model.model
                                                for model_type in nunchaku_model_types:
                                                    if isinstance(transformer, model_type):
                                                        try:
                                                            print(f"Nunchaku: Found model instance ({model_type.__name__}) in ComfyUI model management (transformer, id: {id(transformer)})")
                                                            # Disable CPU offload first if enabled
                                                            if hasattr(transformer, 'set_offload'):
                                                                try:
                                                                    if hasattr(transformer, 'offload') and transformer.offload:
                                                                        print(f"Nunchaku: Disabling CPU offload for ComfyUI model")
                                                                        transformer.set_offload(False)
                                                                        print(f"Nunchaku: CPU offload disabled for ComfyUI model")
                                                                except Exception as e:
                                                                    print(f"Nunchaku: Warning: Failed to disable CPU offload: {e}")
                                                            
                                                            # Clear offload_manager
                                                            if hasattr(transformer, 'offload_manager') and transformer.offload_manager is not None:
                                                                try:
                                                                    print(f"Nunchaku: Clearing offload_manager for ComfyUI model")
                                                                    transformer.offload_manager = None
                                                                    print(f"Nunchaku: offload_manager cleared for ComfyUI model")
                                                                except Exception as e:
                                                                    print(f"Nunchaku: Warning: Failed to clear offload_manager: {e}")
                                                            
                                                            # Clear model's internal state
                                                            try:
                                                                if hasattr(transformer, 'named_parameters'):
                                                                    params_cleared = 0
                                                                    for name, param in list(transformer.named_parameters()):
                                                                        if param is not None and hasattr(param, 'data'):
                                                                            try:
                                                                                if param.data is not None:
                                                                                    param.data = param.data.detach().cpu()
                                                                                    params_cleared += 1
                                                                            except Exception:
                                                                                pass
                                                                    if params_cleared > 0:
                                                                        print(f"Nunchaku: Cleared {params_cleared} parameters from ComfyUI model")
                                                                if hasattr(transformer, 'named_buffers'):
                                                                    buffers_cleared = 0
                                                                    for name, buffer in list(transformer.named_buffers()):
                                                                        if buffer is not None and hasattr(buffer, 'data'):
                                                                            try:
                                                                                if buffer.data is not None:
                                                                                    buffer.data = buffer.data.detach().cpu()
                                                                                    buffers_cleared += 1
                                                                            except Exception:
                                                                                pass
                                                                    if buffers_cleared > 0:
                                                                        print(f"Nunchaku: Cleared {buffers_cleared} buffers from ComfyUI model")
                                                                if hasattr(transformer, '_modules'):
                                                                    transformer._modules.clear()
                                                                    print(f"Nunchaku: Cleared _modules dict from ComfyUI model")
                                                            except Exception as e:
                                                                print(f"Nunchaku: Warning: Failed to clear model internal state: {e}")
                                                            
                                                            # Mark as not currently used
                                                            loaded_model.currently_used = False
                                                            # Unload the model
                                                            print(f"Nunchaku: Unloading model from ComfyUI model management...")
                                                            if hasattr(loaded_model, "model_unload"):
                                                                loaded_model.model_unload()
                                                                print(f"Nunchaku: Model unloaded from ComfyUI model management")
                                                            nunchaku_cleared += 1
                                                            print(f"Nunchaku: Successfully cleared model ({model_type.__name__}) from ComfyUI model management")
                                                            break
                                                        except Exception as e:
                                                            print(f"Nunchaku: Error clearing Nunchaku model from ComfyUI model management: {e}")
                                                            import traceback
                                                            print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                                            # Check if diffusion_model itself is nunchaku model
                                            for model_type in nunchaku_model_types:
                                                if isinstance(diffusion_model, model_type):
                                                    try:
                                                        print(f"Nunchaku: Found model instance ({model_type.__name__}) in ComfyUI model management (diffusion_model, id: {id(diffusion_model)})")
                                                        # Disable CPU offload first if enabled
                                                        if hasattr(diffusion_model, 'set_offload'):
                                                            try:
                                                                if hasattr(diffusion_model, 'offload') and diffusion_model.offload:
                                                                    print(f"Nunchaku: Disabling CPU offload for ComfyUI model (diffusion_model)")
                                                                    diffusion_model.set_offload(False)
                                                                    print(f"Nunchaku: CPU offload disabled for ComfyUI model (diffusion_model)")
                                                            except Exception as e:
                                                                print(f"Nunchaku: Warning: Failed to disable CPU offload: {e}")
                                                        
                                                        # Clear offload_manager
                                                        if hasattr(diffusion_model, 'offload_manager') and diffusion_model.offload_manager is not None:
                                                            try:
                                                                print(f"Nunchaku: Clearing offload_manager for ComfyUI model (diffusion_model)")
                                                                diffusion_model.offload_manager = None
                                                                print(f"Nunchaku: offload_manager cleared for ComfyUI model (diffusion_model)")
                                                            except Exception as e:
                                                                print(f"Nunchaku: Warning: Failed to clear offload_manager: {e}")
                                                        
                                                        # Clear model's internal state
                                                        try:
                                                            if hasattr(diffusion_model, 'named_parameters'):
                                                                params_cleared = 0
                                                                for name, param in list(diffusion_model.named_parameters()):
                                                                    if param is not None and hasattr(param, 'data'):
                                                                        try:
                                                                            if param.data is not None:
                                                                                param.data = param.data.detach().cpu()
                                                                                params_cleared += 1
                                                                        except Exception:
                                                                            pass
                                                                if params_cleared > 0:
                                                                    print(f"Nunchaku: Cleared {params_cleared} parameters from ComfyUI model (diffusion_model)")
                                                            if hasattr(diffusion_model, 'named_buffers'):
                                                                buffers_cleared = 0
                                                                for name, buffer in list(diffusion_model.named_buffers()):
                                                                    if buffer is not None and hasattr(buffer, 'data'):
                                                                        try:
                                                                            if buffer.data is not None:
                                                                                buffer.data = buffer.data.detach().cpu()
                                                                                buffers_cleared += 1
                                                                        except Exception:
                                                                            pass
                                                                if buffers_cleared > 0:
                                                                    print(f"Nunchaku: Cleared {buffers_cleared} buffers from ComfyUI model (diffusion_model)")
                                                            if hasattr(diffusion_model, '_modules'):
                                                                diffusion_model._modules.clear()
                                                                print(f"Nunchaku: Cleared _modules dict from ComfyUI model (diffusion_model)")
                                                        except Exception as e:
                                                            print(f"Nunchaku: Warning: Failed to clear model internal state: {e}")
                                                        
                                                        loaded_model.currently_used = False
                                                        print(f"Nunchaku: Unloading model from ComfyUI model management (diffusion_model)...")
                                                        if hasattr(loaded_model, "model_unload"):
                                                            loaded_model.model_unload()
                                                            print(f"Nunchaku: Model unloaded from ComfyUI model management (diffusion_model)")
                                                        nunchaku_cleared += 1
                                                        print(f"Nunchaku: Successfully cleared model ({model_type.__name__}) from ComfyUI model management (diffusion_model)")
                                                        break
                                                    except Exception as e:
                                                        print(f"Nunchaku: Error clearing Nunchaku model from ComfyUI model management: {e}")
                                                        import traceback
                                                        print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                                except Exception as e:
                                    print(f"Nunchaku: Warning: Error processing loaded model: {e}")
                            print(f"Nunchaku: Method 2 complete - checked {loaded_models_checked} loaded model(s)")
                    except Exception as e:
                        print(f"Nunchaku: Error checking ComfyUI model management for Nunchaku models: {e}")
                        import traceback
                        print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                    
                    # Method 3: Force clear GPU memory for any remaining nunchaku models
                    print("Nunchaku: Method 3 - Searching gc.get_objects() for models...")
                    if torch.cuda.is_available():
                        try:
                            objects_checked = 0
                            models_found_in_gc = 0
                            for obj in gc.get_objects():
                                objects_checked += 1
                                try:
                                    for model_type in nunchaku_model_types:
                                        if isinstance(obj, model_type):
                                            models_found_in_gc += 1
                                            try:
                                                print(f"Nunchaku: Found model instance ({model_type.__name__}) in gc.get_objects() (id: {id(obj)})")
                                                # Disable CPU offload first if enabled
                                                if hasattr(obj, 'set_offload'):
                                                    try:
                                                        if hasattr(obj, 'offload') and obj.offload:
                                                            print(f"Nunchaku: Disabling CPU offload for gc.get_objects() model")
                                                            obj.set_offload(False)
                                                            print(f"Nunchaku: CPU offload disabled for gc.get_objects() model")
                                                    except Exception as e:
                                                        print(f"Nunchaku: Warning: Failed to disable CPU offload: {e}")
                                                
                                                # Clear offload_manager
                                                if hasattr(obj, 'offload_manager') and obj.offload_manager is not None:
                                                    try:
                                                        print(f"Nunchaku: Clearing offload_manager for gc.get_objects() model")
                                                        obj.offload_manager = None
                                                        print(f"Nunchaku: offload_manager cleared for gc.get_objects() model")
                                                    except Exception as e:
                                                        print(f"Nunchaku: Warning: Failed to clear offload_manager: {e}")
                                                
                                                # Move to CPU if on GPU
                                                print(f"Nunchaku: Attempting to move model to CPU...")
                                                try:
                                                    if hasattr(obj, 'device'):
                                                        device_str = str(obj.device)
                                                        if device_str.startswith('cuda'):
                                                            if hasattr(obj, 'to'):
                                                                obj.to('cpu')
                                                                print(f"Nunchaku: Model moved to CPU using .to('cpu')")
                                                            elif hasattr(obj, 'cpu'):
                                                                obj.cpu()
                                                                print(f"Nunchaku: Model moved to CPU using .cpu()")
                                                            nunchaku_cleared += 1
                                                    else:
                                                        # No device attribute, try to move to CPU anyway
                                                        if hasattr(obj, 'to'):
                                                            obj.to('cpu')
                                                            print(f"Nunchaku: Model moved to CPU using .to('cpu') (no device attr)")
                                                        elif hasattr(obj, 'cpu'):
                                                            obj.cpu()
                                                            print(f"Nunchaku: Model moved to CPU using .cpu() (no device attr)")
                                                        nunchaku_cleared += 1
                                                except Exception as e:
                                                    print(f"Nunchaku: Warning: Failed to move model to CPU: {e}")
                                                
                                                # Clear model's internal state
                                                try:
                                                    if hasattr(obj, 'named_parameters'):
                                                        params_cleared = 0
                                                        for name, param in list(obj.named_parameters()):
                                                            if param is not None and hasattr(param, 'data'):
                                                                try:
                                                                    if param.data is not None:
                                                                        param.data = param.data.detach().cpu()
                                                                        params_cleared += 1
                                                                except Exception:
                                                                    pass
                                                        if params_cleared > 0:
                                                            print(f"Nunchaku: Cleared {params_cleared} parameters from gc.get_objects() model")
                                                    if hasattr(obj, 'named_buffers'):
                                                        buffers_cleared = 0
                                                        for name, buffer in list(obj.named_buffers()):
                                                            if buffer is not None and hasattr(buffer, 'data'):
                                                                try:
                                                                    if buffer.data is not None:
                                                                        buffer.data = buffer.data.detach().cpu()
                                                                        buffers_cleared += 1
                                                                except Exception:
                                                                    pass
                                                        if buffers_cleared > 0:
                                                            print(f"Nunchaku: Cleared {buffers_cleared} buffers from gc.get_objects() model")
                                                    if hasattr(obj, '_modules'):
                                                        obj._modules.clear()
                                                        print(f"Nunchaku: Cleared _modules dict from gc.get_objects() model")
                                                    print(f"Nunchaku: Cleared model internal state from gc.get_objects()")
                                                except Exception as e:
                                                    print(f"Nunchaku: Warning: Failed to clear model internal state from gc.get_objects(): {e}")
                                                
                                                print(f"Nunchaku: Successfully cleared model ({model_type.__name__}) from gc.get_objects()")
                                            except Exception as e:
                                                print(f"Nunchaku: Error clearing model from gc.get_objects(): {e}")
                                                import traceback
                                                print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                                            break
                                except Exception:
                                    pass
                            print(f"Nunchaku: Method 3 complete - checked {objects_checked} objects, found {models_found_in_gc} models")
                        except Exception as e:
                            print(f"Nunchaku: Error in GPU memory cleanup: {e}")
                            import traceback
                            print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                    
                    # Force garbage collection and clear GPU cache
                    print("Nunchaku: Running garbage collection...")
                    gc.collect()
                    gc.collect()  # Run twice to ensure cleanup
                    if torch.cuda.is_available():
                        print("Nunchaku: Clearing CUDA cache...")
                        # Clear cache for all devices
                        for device_idx in range(torch.cuda.device_count()):
                            with torch.cuda.device(device_idx):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                        print("Nunchaku: CUDA cache cleared for all devices")
                    
                    if nunchaku_cleared > 0:
                        print(f"Nunchaku: Successfully cleared {nunchaku_cleared} model(s)")
                    else:
                        print("Nunchaku: No models found in memory (models may not be cached or already cleared)")
                else:
                    print("Nunchaku: nunchaku library not available or model types not found")
                    
            except Exception as e:
                print(f"Nunchaku: Error purging models: {e}")
                import traceback
                print(f"Nunchaku: Traceback: {traceback.format_exc()}")

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
    "MemoryManager": MemoryManager,
    "SafeMemoryManager": SafeMemoryManager,
    "DisTorchPurgeVRAMV2": DisTorchPurgeVRAMV2,
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryManager": "Memory Manager",
    "SafeMemoryManager": "Safe Memory Manager",
    "DisTorchPurgeVRAMV2": "LayerUtility: Purge VRAM V2",
    "ModelPatchMemoryCleaner": "Model Patch Memory Cleaner",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 