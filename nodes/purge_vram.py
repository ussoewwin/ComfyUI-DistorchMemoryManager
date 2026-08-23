"""
Purge VRAM V2 node for DistorchMemoryManager
"""
import torch
import gc
import sys
import os
import logging
import importlib

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
                "HSWQ": ("BOOLEAN", {"default": False, "tooltip": "Purge HSWQ residual VRAM (whole HSWQ path: models, PinCache, Detailer caches)"}),
                "Ollama": ("BOOLEAN", {"default": False, "tooltip": "Full purge of Ollama VRAM used by comfyui-ollama and comfyui-ollama-describer: unload every loaded model until /api/ps is empty (generate+chat keep_alive=0, ollama stop), clear CHAT_SESSIONS/saved_context, wipe saved_context files"}),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "purge_vram"
    CATEGORY = "Distorch/Memory"

    def purge_vram(self, anything, purge_cache, purge_models, purge_seedvr2_models, purge_qwen3vl_models, purge_nunchaku_models, **kwargs):
        # Toggle label is "HSWQ"; accept legacy "HSWQ INT8" for old workflows.
        purge_hswq_int8 = bool(kwargs.get("HSWQ", kwargs.get("HSWQ INT8", False)))
        purge_ollama = bool(kwargs.get("Ollama", False))
        global torch
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
                
                # Aggressive unload: MultiGPU Dynamic / NVFP4 (Krea2) often leave
                # ~9GB CUDA after model_unload()==True. Soft path alone is not enough.
                if hasattr(comfy.model_management, "current_loaded_models"):
                    current_loaded_models = comfy.model_management.current_loaded_models
                    unloaded_count = 0
                    bytes_force_killed = 0

                    def _force_empty_cuda_storage(t) -> int:
                        # NVFP4 / MultiGPU Dynamic: free leftover CUDA only.
                        # Never wipe CPU tensors — after model_unload() they are
                        # ComfyUI's reload source. Wiping to empty(0) made CLIP
                        # Embedding.weight non-2D (Ollama purge → CLIPTextEncode
                        # RuntimeError: 'weight' must be 2-D; reload logged 0.00 MB).
                        if t is None:
                            return 0
                        freed = 0
                        try:
                            data = getattr(t, "data", t)
                            if data is None:
                                return 0
                            nbytes = int(getattr(data, "nbytes", 0) or 0)
                            is_cuda = False
                            try:
                                is_cuda = bool(getattr(data, "is_cuda", False))
                                if not is_cuda:
                                    dev = getattr(data, "device", None)
                                    is_cuda = getattr(dev, "type", None) == "cuda"
                            except Exception:
                                pass
                            if not is_cuda:
                                return 0
                            dtype = getattr(data, "dtype", torch.float32)
                            empty = torch.empty(0, dtype=dtype, device="cpu")
                            if hasattr(t, "data"):
                                t.data = empty
                            freed = nbytes
                        except Exception:
                            pass
                        return freed

                    def _force_kill_nn_cuda(module) -> int:
                        if module is None:
                            return 0
                        freed = 0
                        try:
                            if hasattr(module, "to") and callable(module.to):
                                try:
                                    module.to("cpu")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            for _n, p in list(module.named_parameters()):
                                freed += _force_empty_cuda_storage(p)
                        except Exception:
                            pass
                        try:
                            for _n, b in list(module.named_buffers()):
                                freed += _force_empty_cuda_storage(b)
                        except Exception:
                            pass
                        return freed

                    def _unwrap_nn_soft(obj):
                        cur = obj
                        for _ in range(8):
                            if cur is None:
                                return None
                            try:
                                if isinstance(cur, torch.nn.Module):
                                    return cur
                            except Exception:
                                pass
                            nxt = getattr(cur, "model", None)
                            if nxt is None or nxt is cur:
                                nxt = getattr(cur, "diffusion_model", None)
                            if nxt is None or nxt is cur:
                                try:
                                    return cur if isinstance(cur, torch.nn.Module) else None
                                except Exception:
                                    return None
                            cur = nxt
                        try:
                            return cur if isinstance(cur, torch.nn.Module) else None
                        except Exception:
                            return None

                    # Mark unused, unload, kill CUDA storage, then remove from registry
                    for i in range(len(current_loaded_models) - 1, -1, -1):
                        loaded_model = current_loaded_models[i]
                        if loaded_model is None:
                            try:
                                current_loaded_models.pop(i)
                            except Exception:
                                pass
                            continue
                        try:
                            try:
                                loaded_model.currently_used = False
                            except Exception:
                                pass
                            try:
                                if hasattr(loaded_model, "partially_unload") and callable(loaded_model.partially_unload):
                                    try:
                                        loaded_model.partially_unload(None, 1e30)
                                    except Exception:
                                        loaded_model.partially_unload(torch.device("cpu"), 1e30)
                            except Exception:
                                pass
                            try:
                                if hasattr(loaded_model, "model_unload") and callable(loaded_model.model_unload):
                                    loaded_model.model_unload()
                                    unloaded_count += 1
                            except Exception as e:
                                print(f"Error unloading model: {e}")
                            try:
                                inner = getattr(loaded_model, "model", None)
                                nn = _unwrap_nn_soft(inner)
                                if nn is not None:
                                    bytes_force_killed += _force_kill_nn_cuda(nn)
                                elif inner is not None:
                                    bytes_force_killed += _force_kill_nn_cuda(_unwrap_nn_soft(inner))
                            except Exception:
                                pass
                            try:
                                current_loaded_models.pop(i)
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"Error force-unloading model[{i}]: {e}")
                            try:
                                current_loaded_models.pop(i)
                            except Exception:
                                pass

                    if unloaded_count > 0:
                        print(f"Unloaded {unloaded_count} model(s)")
                    if bytes_force_killed > 0:
                        print(
                            f"Force-killed ~{bytes_force_killed / (1024 ** 3):.2f} GB CUDA storage "
                            f"from loaded models (MultiGPU/NVFP4 soft-unload residue)"
                        )

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

                # Hard free: unload_all + free_memory(1e30). free_memory(0) does nothing.
                try:
                    mm = comfy.model_management
                    if hasattr(mm, "unload_all_models") and callable(mm.unload_all_models):
                        mm.unload_all_models()
                        print("unload_all_models() issued")
                    if torch.cuda.is_available() and hasattr(mm, "free_memory") and callable(mm.free_memory):
                        for di in range(torch.cuda.device_count()):
                            try:
                                mm.free_memory(1e30, torch.device(f"cuda:{di}"))
                            except Exception as e:
                                print(f"free_memory(cuda:{di}) warning: {e}")
                        print("free_memory(1e30) issued for all CUDA devices")
                except Exception as e:
                    print(f"Hard free after purge_models warning: {e}")
                
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
                                                # Try to clear model's internal state more aggressively (delete, not move to CPU)
                                                if hasattr(attr, 'named_parameters'):
                                                    for name, param in list(attr.named_parameters(recurse=False)):
                                                        if param is not None and hasattr(param, 'data'):
                                                            try:
                                                                if param.data is not None:
                                                                    # Delete data instead of moving to CPU
                                                                    del param.data
                                                            except Exception:
                                                                pass
                                                if hasattr(attr, 'named_buffers'):
                                                    for name, buffer in list(attr.named_buffers(recurse=False)):
                                                        if buffer is not None and hasattr(buffer, 'data'):
                                                            try:
                                                                if buffer.data is not None:
                                                                    # Delete data instead of moving to CPU
                                                                    del buffer.data
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
                                                    # Clear model's internal state before deletion (delete, not move to CPU)
                                                    if hasattr(model_obj, 'named_parameters'):
                                                        for name, param in list(model_obj.named_parameters(recurse=False)):
                                                            if param is not None and hasattr(param, 'data'):
                                                                try:
                                                                    if param.data is not None:
                                                                        # Delete data instead of moving to CPU
                                                                        del param.data
                                                                except Exception:
                                                                    pass
                                                    if hasattr(model_obj, 'named_buffers'):
                                                        for name, buffer in list(model_obj.named_buffers(recurse=False)):
                                                            if buffer is not None and hasattr(buffer, 'data'):
                                                                try:
                                                                    if buffer.data is not None:
                                                                        # Delete data instead of moving to CPU
                                                                        del buffer.data
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
                                            # Clear model's internal state more aggressively (delete, not move to CPU)
                                            if hasattr(obj, 'named_parameters'):
                                                for name, param in list(obj.named_parameters(recurse=False)):
                                                    if param is not None and hasattr(param, 'data'):
                                                        try:
                                                            if param.data is not None:
                                                                # Delete data instead of moving to CPU
                                                                del param.data
                                                        except Exception:
                                                            pass
                                            if hasattr(obj, 'named_buffers'):
                                                for name, buffer in list(obj.named_buffers(recurse=False)):
                                                    if buffer is not None and hasattr(buffer, 'data'):
                                                        try:
                                                            if buffer.data is not None:
                                                                # Delete data instead of moving to CPU
                                                                del buffer.data
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
                
                # Try to import NunchakuSDXLUNet2DConditionModel (SDXL)
                try:
                    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
                    nunchaku_model_types.append(NunchakuSDXLUNet2DConditionModel)
                    print("Nunchaku: Successfully imported NunchakuSDXLUNet2DConditionModel")
                except ImportError as e:
                    print(f"Nunchaku: Failed to import NunchakuSDXLUNet2DConditionModel: {e}")
                
                # Try to import NunchakuSDXL class (SDXL model wrapper)
                nunchaku_sdxl_class = None
                try:
                    # Try to import from model_base
                    try:
                        from model_base.sdxl import NunchakuSDXL
                        nunchaku_sdxl_class = NunchakuSDXL
                        print("Nunchaku: Successfully imported NunchakuSDXL from model_base.sdxl")
                    except ImportError:
                        # Try alternative import paths
                        try:
                            for module_name in list(sys.modules.keys()):
                                if 'nunchaku' in module_name.lower() and 'sdxl' in module_name.lower():
                                    module = sys.modules[module_name]
                                    if hasattr(module, 'NunchakuSDXL'):
                                        nunchaku_sdxl_class = getattr(module, 'NunchakuSDXL')
                                        print(f"Nunchaku: Found NunchakuSDXL in {module_name}")
                                        break
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Nunchaku: Failed to import NunchakuSDXL: {e}")
                
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
                                                
                                                # Delete model data directly (not moving to CPU)
                                                print(f"Nunchaku: Attempting to delete model data...")
                                                
                                                # Clear model's internal state by deleting data
                                                try:
                                                    params_deleted = 0
                                                    buffers_deleted = 0
                                                    
                                                    # Delete parameters (only top-level, not submodules)
                                                    if hasattr(attr, 'named_parameters'):
                                                        for name, param in list(attr.named_parameters(recurse=False)):
                                                            if param is not None:
                                                                try:
                                                                    if hasattr(param, 'data') and param.data is not None:
                                                                        del param.data
                                                                        params_deleted += 1
                                                                except Exception:
                                                                    pass
                                                    
                                                    # Delete buffers (only top-level, not submodules)
                                                    if hasattr(attr, 'named_buffers'):
                                                        for name, buffer in list(attr.named_buffers(recurse=False)):
                                                            if buffer is not None:
                                                                try:
                                                                    if hasattr(buffer, 'data') and buffer.data is not None:
                                                                        del buffer.data
                                                                        buffers_deleted += 1
                                                                except Exception:
                                                                    pass
                                                    
                                                    # Clear _parameters and _buffers dicts
                                                    if hasattr(attr, '_parameters'):
                                                        try:
                                                            for param_name in list(attr._parameters.keys()):
                                                                param = attr._parameters[param_name]
                                                                if param is not None and hasattr(param, 'data') and param.data is not None:
                                                                    try:
                                                                        del param.data
                                                                    except Exception:
                                                                        pass
                                                            attr._parameters.clear()
                                                        except Exception:
                                                            pass
                                                    
                                                    if hasattr(attr, '_buffers'):
                                                        try:
                                                            for buffer_name in list(attr._buffers.keys()):
                                                                buffer = attr._buffers[buffer_name]
                                                                if buffer is not None and hasattr(buffer, 'data') and buffer.data is not None:
                                                                    try:
                                                                        del buffer.data
                                                                    except Exception:
                                                                        pass
                                                            attr._buffers.clear()
                                                        except Exception:
                                                            pass
                                                    
                                                    # DO NOT recursively delete submodule parameters - it breaks model structure
                                                    # Submodules contain essential parameters like Linear.weight that must be preserved
                                                    # Only top-level parameters are deleted above (with recurse=False)
                                                    
                                                    if params_deleted > 0:
                                                        print(f"Nunchaku: Deleted {params_deleted} total parameters from {module_name}.{attr_name}")
                                                    if buffers_deleted > 0:
                                                        print(f"Nunchaku: Deleted {buffers_deleted} total buffers from {module_name}.{attr_name}")
                                                    # Module structure (_modules) is preserved for re-loading via load_state_dict
                                                except Exception as e:
                                                    print(f"Nunchaku: Warning: Failed to delete model internal state: {e}")
                                                
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
                                    
                                    # Check if it's a NunchakuSDXL instance (SDXL model wrapper)
                                    if not model_found and nunchaku_sdxl_class is not None:
                                        try:
                                            if isinstance(attr, nunchaku_sdxl_class):
                                                if hasattr(attr, 'diffusion_model'):
                                                    diffusion_model = attr.diffusion_model
                                                    for model_type in nunchaku_model_types:
                                                        if isinstance(diffusion_model, model_type):
                                                            try:
                                                                print(f"Nunchaku: Found NunchakuSDXL instance with {model_type.__name__} at {module_name}.{attr_name} (id: {id(attr)})")
                                                                # Disable CPU offload first if enabled
                                                                if hasattr(diffusion_model, 'set_offload'):
                                                                    try:
                                                                        if hasattr(diffusion_model, 'offload') and diffusion_model.offload:
                                                                            print(f"Nunchaku: Disabling CPU offload for SDXL diffusion_model")
                                                                            diffusion_model.set_offload(False)
                                                                            print(f"Nunchaku: CPU offload disabled for SDXL diffusion_model")
                                                                    except Exception as e:
                                                                        print(f"Nunchaku: Warning: Failed to disable CPU offload: {e}")
                                                                
                                                                # Clear offload_manager
                                                                if hasattr(diffusion_model, 'offload_manager') and diffusion_model.offload_manager is not None:
                                                                    try:
                                                                        print(f"Nunchaku: Clearing offload_manager for SDXL diffusion_model")
                                                                        diffusion_model.offload_manager = None
                                                                        print(f"Nunchaku: offload_manager cleared for SDXL diffusion_model")
                                                                    except Exception as e:
                                                                        print(f"Nunchaku: Warning: Failed to clear offload_manager: {e}")
                                                                
                                                                # Delete SDXL diffusion_model data directly (not moving to CPU)
                                                                print(f"Nunchaku: Attempting to delete SDXL diffusion_model data...")
                                                                
                                                                # Delete model's internal state
                                                                try:
                                                                    params_deleted = 0
                                                                    buffers_deleted = 0
                                                                    
                                                                    # Delete parameters (only top-level, not submodules)
                                                                    if hasattr(diffusion_model, 'named_parameters'):
                                                                        for name, param in list(diffusion_model.named_parameters(recurse=False)):
                                                                            if param is not None:
                                                                                try:
                                                                                    if hasattr(param, 'data') and param.data is not None:
                                                                                        del param.data
                                                                                        params_deleted += 1
                                                                                except Exception:
                                                                                    pass
                                                                    
                                                                    # Delete buffers (only top-level, not submodules)
                                                                    if hasattr(diffusion_model, 'named_buffers'):
                                                                        for name, buffer in list(diffusion_model.named_buffers(recurse=False)):
                                                                            if buffer is not None:
                                                                                try:
                                                                                    if hasattr(buffer, 'data') and buffer.data is not None:
                                                                                        del buffer.data
                                                                                        buffers_deleted += 1
                                                                                except Exception:
                                                                                    pass
                                                                    
                                                                    # Clear _parameters and _buffers dicts
                                                                    if hasattr(diffusion_model, '_parameters'):
                                                                        try:
                                                                            for param_name in list(diffusion_model._parameters.keys()):
                                                                                param = diffusion_model._parameters[param_name]
                                                                                if param is not None and hasattr(param, 'data') and param.data is not None:
                                                                                    try:
                                                                                        del param.data
                                                                                    except Exception:
                                                                                        pass
                                                                            diffusion_model._parameters.clear()
                                                                        except Exception:
                                                                            pass
                                                                    
                                                                    if hasattr(diffusion_model, '_buffers'):
                                                                        try:
                                                                            for buffer_name in list(diffusion_model._buffers.keys()):
                                                                                buffer = diffusion_model._buffers[buffer_name]
                                                                                if buffer is not None and hasattr(buffer, 'data') and buffer.data is not None:
                                                                                    try:
                                                                                        del buffer.data
                                                                                    except Exception:
                                                                                        pass
                                                                            diffusion_model._buffers.clear()
                                                                        except Exception:
                                                                            pass
                                                                    
                                                                    # DO NOT recursively delete submodule parameters - it breaks model structure
                                                                    # Submodules contain essential parameters like Linear.weight that must be preserved
                                                                    # Only top-level parameters are deleted above (with recurse=False)
                                                                    
                                                                    # Try to clear any cached data or temporary attributes that might hold VRAM
                                                                    try:
                                                                        # Clear any cache-related attributes
                                                                        cache_attrs = ['_cache', 'cache', '_state_dict_cache', 'state_dict_cache', '_non_persistent_buffers_set']
                                                                        for cache_attr in cache_attrs:
                                                                            if hasattr(diffusion_model, cache_attr):
                                                                                try:
                                                                                    cache_val = getattr(diffusion_model, cache_attr)
                                                                                    if cache_val is not None:
                                                                                        if isinstance(cache_val, (dict, set)):
                                                                                            cache_val.clear()
                                                                                        elif hasattr(cache_val, 'clear'):
                                                                                            cache_val.clear()
                                                                                        setattr(diffusion_model, cache_attr, None)
                                                                                        print(f"Nunchaku: Cleared {cache_attr} from SDXL diffusion_model")
                                                                                except Exception:
                                                                                    pass
                                                                    except Exception:
                                                                        pass
                                                                    
                                                                    if params_deleted > 0:
                                                                        print(f"Nunchaku: Deleted {params_deleted} total parameters from SDXL diffusion_model")
                                                                    if buffers_deleted > 0:
                                                                        print(f"Nunchaku: Deleted {buffers_deleted} total buffers from SDXL diffusion_model")
                                                                    # Module structure (_modules) is preserved for re-loading via load_state_dict
                                                                except Exception as e:
                                                                    print(f"Nunchaku: Warning: Failed to delete SDXL diffusion_model internal state: {e}")
                                                                
                                                                # Clear the diffusion_model reference
                                                                try:
                                                                    attr.diffusion_model = None
                                                                    print(f"Nunchaku: Cleared diffusion_model reference from NunchakuSDXL instance")
                                                                except Exception as e:
                                                                    print(f"Nunchaku: Warning: Failed to clear diffusion_model reference: {e}")
                                                                
                                                                nunchaku_cleared += 1
                                                                print(f"Nunchaku: Successfully cleared SDXL model ({model_type.__name__}) from {module_name}.{attr_name}")
                                                                model_found = True
                                                                break
                                                            except Exception as e:
                                                                print(f"Nunchaku: Error clearing SDXL model from {module_name}.{attr_name}: {e}")
                                                                import traceback
                                                                print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                                        except Exception as e:
                                            print(f"Nunchaku: Warning: Error checking NunchakuSDXL instance: {e}")
                                    
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
                                                        
                                                        # Delete transformer_obj data
                                                        try:
                                                            if hasattr(transformer_obj, 'named_parameters'):
                                                                for _, p in list(transformer_obj.named_parameters()):
                                                                    if p is not None and hasattr(p, 'data') and p.data is not None:
                                                                        try:
                                                                            del p.data
                                                                        except Exception:
                                                                            pass
                                                            if hasattr(transformer_obj, 'named_buffers'):
                                                                for _, b in list(transformer_obj.named_buffers()):
                                                                    if b is not None and hasattr(b, 'data') and b.data is not None:
                                                                        try:
                                                                            del b.data
                                                                        except Exception:
                                                                            pass
                                                            if hasattr(transformer_obj, '_modules'):
                                                                # DO NOT clear _modules - it contains module structure
                                                                # transformer_obj._modules.clear()  # Removed: clearing _modules breaks model structure
                                                                pass
                                                        except Exception:
                                                            pass
                                                        attr['transformer'] = None
                                                        nunchaku_cleared += 1
                                                        print(f"Deleted Nunchaku model ({model_type.__name__}) from dict.transformer in {module_name}.{attr_name}")
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
                                                                
                                                                # Delete transformer_obj data
                                                                try:
                                                                    if hasattr(transformer_obj, 'named_parameters'):
                                                                        for _, p in list(transformer_obj.named_parameters()):
                                                                            if p is not None and hasattr(p, 'data') and p.data is not None:
                                                                                try:
                                                                                    del p.data
                                                                                except Exception:
                                                                                    pass
                                                                    if hasattr(transformer_obj, 'named_buffers'):
                                                                        for _, b in list(transformer_obj.named_buffers()):
                                                                            if b is not None and hasattr(b, 'data') and b.data is not None:
                                                                                try:
                                                                                    del b.data
                                                                                except Exception:
                                                                                    pass
                                                                    if hasattr(transformer_obj, '_modules'):
                                                                        # DO NOT clear _modules - it contains module structure
                                                                        # transformer_obj._modules.clear()  # Removed: clearing _modules breaks model structure
                                                                        pass
                                                                except Exception:
                                                                    pass
                                                                nunchaku_cleared += 1
                                                                print(f"Deleted Nunchaku model ({model_type.__name__}) from dict.model.diffusion_model.model in {module_name}.{attr_name}")
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
                                                            
                                                            # Delete diffusion_model data recursively
                                                            try:
                                                                params_deleted = 0
                                                                buffers_deleted = 0
                                                                
                                                                # Recursively delete all parameters and buffers (including submodules)
                                                                # This fully frees VRAM while preserving module structure for re-loading
                                                                # DO NOT recursively delete submodule parameters - it breaks model structure
                                                                # Submodules contain essential parameters like Linear.weight that must be preserved
                                                                # Only top-level parameters are deleted above (with recurse=False)
                                                                
                                                                if params_deleted > 0:
                                                                    print(f"Nunchaku: Deleted {params_deleted} total parameters from diffusion_model in {module_name}.{attr_name}")
                                                                if buffers_deleted > 0:
                                                                    print(f"Nunchaku: Deleted {buffers_deleted} total buffers from diffusion_model in {module_name}.{attr_name}")
                                                                # Module structure (_modules) is preserved for re-loading via load_state_dict
                                                            except Exception:
                                                                pass
                                                            nunchaku_cleared += 1
                                                            print(f"Deleted Nunchaku model ({model_type.__name__}) from dict.model.diffusion_model in {module_name}.{attr_name}")
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
                                        
                                        # Check if model is a NunchakuSDXL instance (SDXL model wrapper)
                                        if nunchaku_sdxl_class is not None and isinstance(model, nunchaku_sdxl_class):
                                            if hasattr(model, "diffusion_model"):
                                                diffusion_model = model.diffusion_model
                                                for model_type in nunchaku_model_types:
                                                    if isinstance(diffusion_model, model_type):
                                                        try:
                                                            print(f"Nunchaku: Found NunchakuSDXL instance with {model_type.__name__} in ComfyUI model management (id: {id(model)})")
                                                            # Disable CPU offload first if enabled
                                                            if hasattr(diffusion_model, 'set_offload'):
                                                                try:
                                                                    if hasattr(diffusion_model, 'offload') and diffusion_model.offload:
                                                                        print(f"Nunchaku: Disabling CPU offload for SDXL diffusion_model in ComfyUI")
                                                                        diffusion_model.set_offload(False)
                                                                        print(f"Nunchaku: CPU offload disabled for SDXL diffusion_model in ComfyUI")
                                                                except Exception as e:
                                                                    print(f"Nunchaku: Warning: Failed to disable CPU offload: {e}")
                                                            
                                                            # Clear offload_manager
                                                            if hasattr(diffusion_model, 'offload_manager') and diffusion_model.offload_manager is not None:
                                                                try:
                                                                    print(f"Nunchaku: Clearing offload_manager for SDXL diffusion_model in ComfyUI")
                                                                    diffusion_model.offload_manager = None
                                                                    print(f"Nunchaku: offload_manager cleared for SDXL diffusion_model in ComfyUI")
                                                                except Exception as e:
                                                                    print(f"Nunchaku: Warning: Failed to clear offload_manager: {e}")
                                                            
                                                            # Delete SDXL diffusion_model data
                                                            print(f"Nunchaku: Attempting to delete SDXL diffusion_model data in ComfyUI...")
                                                            try:
                                                                params_deleted = 0
                                                                buffers_deleted = 0
                                                                
                                                                # Delete parameters (only top-level, not submodules)
                                                                if hasattr(diffusion_model, 'named_parameters'):
                                                                    for name, param in list(diffusion_model.named_parameters(recurse=False)):
                                                                        if param is not None:
                                                                            try:
                                                                                if hasattr(param, 'data') and param.data is not None:
                                                                                    del param.data
                                                                                    params_deleted += 1
                                                                            except Exception:
                                                                                pass
                                                                
                                                                # Delete buffers (only top-level, not submodules)
                                                                if hasattr(diffusion_model, 'named_buffers'):
                                                                    for name, buffer in list(diffusion_model.named_buffers(recurse=False)):
                                                                        if buffer is not None:
                                                                            try:
                                                                                if hasattr(buffer, 'data') and buffer.data is not None:
                                                                                    del buffer.data
                                                                                    buffers_deleted += 1
                                                                            except Exception:
                                                                                pass
                                                                
                                                                # Clear _parameters and _buffers dicts
                                                                if hasattr(diffusion_model, '_parameters'):
                                                                    try:
                                                                        for param_name in list(diffusion_model._parameters.keys()):
                                                                            param = diffusion_model._parameters[param_name]
                                                                            if param is not None and hasattr(param, 'data') and param.data is not None:
                                                                                try:
                                                                                    del param.data
                                                                                except Exception:
                                                                                    pass
                                                                        diffusion_model._parameters.clear()
                                                                    except Exception:
                                                                        pass
                                                                
                                                                if hasattr(diffusion_model, '_buffers'):
                                                                    try:
                                                                        for buffer_name in list(diffusion_model._buffers.keys()):
                                                                            buffer = diffusion_model._buffers[buffer_name]
                                                                            if buffer is not None and hasattr(buffer, 'data') and buffer.data is not None:
                                                                                try:
                                                                                    del buffer.data
                                                                                except Exception:
                                                                                    pass
                                                                        diffusion_model._buffers.clear()
                                                                    except Exception:
                                                                        pass
                                                                
                                                                # DO NOT recursively delete submodule parameters - it breaks model structure
                                                                # Submodules contain essential parameters like Linear.weight that must be preserved
                                                                # Only top-level parameters are deleted above (with recurse=False)
                                                                
                                                                # Try to clear any cached data or temporary attributes that might hold VRAM
                                                                try:
                                                                    # Clear any cache-related attributes
                                                                    cache_attrs = ['_cache', 'cache', '_state_dict_cache', 'state_dict_cache', '_non_persistent_buffers_set']
                                                                    for cache_attr in cache_attrs:
                                                                        if hasattr(diffusion_model, cache_attr):
                                                                            try:
                                                                                cache_val = getattr(diffusion_model, cache_attr)
                                                                                if cache_val is not None:
                                                                                    if isinstance(cache_val, (dict, set)):
                                                                                        cache_val.clear()
                                                                                    elif hasattr(cache_val, 'clear'):
                                                                                        cache_val.clear()
                                                                                    setattr(diffusion_model, cache_attr, None)
                                                                                    print(f"Nunchaku: Cleared {cache_attr} from SDXL diffusion_model in ComfyUI")
                                                                            except Exception:
                                                                                pass
                                                                except Exception:
                                                                    pass
                                                                
                                                                if params_deleted > 0:
                                                                    print(f"Nunchaku: Deleted {params_deleted} total parameters from SDXL diffusion_model in ComfyUI")
                                                                if buffers_deleted > 0:
                                                                    print(f"Nunchaku: Deleted {buffers_deleted} total buffers from SDXL diffusion_model in ComfyUI")
                                                                # Module structure (_modules) is preserved for re-loading via load_state_dict
                                                            except Exception as e:
                                                                print(f"Nunchaku: Warning: Failed to delete SDXL diffusion_model internal state: {e}")
                                                            
                                                            # Mark as not currently used and unload
                                                            loaded_model.currently_used = False
                                                            print(f"Nunchaku: Unloading SDXL model from ComfyUI model management...")
                                                            if hasattr(loaded_model, "model_unload"):
                                                                loaded_model.model_unload()
                                                                print(f"Nunchaku: SDXL model unloaded from ComfyUI model management")
                                                            nunchaku_cleared += 1
                                                            print(f"Nunchaku: Successfully cleared SDXL model ({model_type.__name__}) from ComfyUI model management")
                                                            break
                                                        except Exception as e:
                                                            print(f"Nunchaku: Error clearing SDXL model from ComfyUI model management: {e}")
                                                            import traceback
                                                            print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                                        
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
                                                            
                                                            # Delete model data
                                                            print(f"Nunchaku: Attempting to delete model data in ComfyUI...")
                                                            try:
                                                                params_deleted = 0
                                                                buffers_deleted = 0
                                                                
                                                                # Delete parameters (only top-level, not submodules)
                                                                if hasattr(transformer, 'named_parameters'):
                                                                    for name, param in list(transformer.named_parameters(recurse=False)):
                                                                        if param is not None:
                                                                            try:
                                                                                if hasattr(param, 'data') and param.data is not None:
                                                                                    del param.data
                                                                                    params_deleted += 1
                                                                            except Exception:
                                                                                pass
                                                                
                                                                # Delete buffers (only top-level, not submodules)
                                                                if hasattr(transformer, 'named_buffers'):
                                                                    for name, buffer in list(transformer.named_buffers(recurse=False)):
                                                                        if buffer is not None:
                                                                            try:
                                                                                if hasattr(buffer, 'data') and buffer.data is not None:
                                                                                    del buffer.data
                                                                                    buffers_deleted += 1
                                                                            except Exception:
                                                                                pass
                                                                
                                                                # Clear _parameters and _buffers dicts
                                                                if hasattr(transformer, '_parameters'):
                                                                    try:
                                                                        for param_name in list(transformer._parameters.keys()):
                                                                            param = transformer._parameters[param_name]
                                                                            if param is not None and hasattr(param, 'data') and param.data is not None:
                                                                                try:
                                                                                    del param.data
                                                                                except Exception:
                                                                                    pass
                                                                        transformer._parameters.clear()
                                                                    except Exception:
                                                                        pass
                                                                
                                                                if hasattr(transformer, '_buffers'):
                                                                    try:
                                                                        for buffer_name in list(transformer._buffers.keys()):
                                                                            buffer = transformer._buffers[buffer_name]
                                                                            if buffer is not None and hasattr(buffer, 'data') and buffer.data is not None:
                                                                                try:
                                                                                    del buffer.data
                                                                                except Exception:
                                                                                    pass
                                                                        transformer._buffers.clear()
                                                                    except Exception:
                                                                        pass
                                                                
                                                                # Recursively delete all parameters and buffers (including submodules)
                                                                # This fully frees VRAM while preserving module structure for re-loading
                                                                # DO NOT recursively delete submodule parameters - it breaks model structure
                                                                # Submodules contain essential parameters like Linear.weight that must be preserved
                                                                # Only top-level parameters are deleted above (with recurse=False)
                                                                
                                                                if params_deleted > 0:
                                                                    print(f"Nunchaku: Deleted {params_deleted} total parameters from ComfyUI model")
                                                                if buffers_deleted > 0:
                                                                    print(f"Nunchaku: Deleted {buffers_deleted} total buffers from ComfyUI model")
                                                                # Module structure (_modules) is preserved for re-loading via load_state_dict
                                                            except Exception as e:
                                                                print(f"Nunchaku: Warning: Failed to delete model internal state: {e}")
                                                            
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
                                                        
                                                        # Delete model data
                                                        print(f"Nunchaku: Attempting to delete model data in ComfyUI (diffusion_model)...")
                                                        try:
                                                            params_deleted = 0
                                                            buffers_deleted = 0
                                                            
                                                            # Delete parameters (only top-level, not submodules)
                                                            if hasattr(diffusion_model, 'named_parameters'):
                                                                for name, param in list(diffusion_model.named_parameters(recurse=False)):
                                                                    if param is not None:
                                                                        try:
                                                                            if hasattr(param, 'data') and param.data is not None:
                                                                                del param.data
                                                                                params_deleted += 1
                                                                        except Exception:
                                                                            pass
                                                            
                                                            # Delete buffers (only top-level, not submodules)
                                                            if hasattr(diffusion_model, 'named_buffers'):
                                                                for name, buffer in list(diffusion_model.named_buffers(recurse=False)):
                                                                    if buffer is not None:
                                                                        try:
                                                                            if hasattr(buffer, 'data') and buffer.data is not None:
                                                                                del buffer.data
                                                                                buffers_deleted += 1
                                                                        except Exception:
                                                                            pass
                                                            
                                                            # Clear _parameters and _buffers dicts
                                                            if hasattr(diffusion_model, '_parameters'):
                                                                try:
                                                                    for param_name in list(diffusion_model._parameters.keys()):
                                                                        param = diffusion_model._parameters[param_name]
                                                                        if param is not None and hasattr(param, 'data') and param.data is not None:
                                                                            try:
                                                                                del param.data
                                                                            except Exception:
                                                                                pass
                                                                    diffusion_model._parameters.clear()
                                                                except Exception:
                                                                    pass
                                                            
                                                            if hasattr(diffusion_model, '_buffers'):
                                                                try:
                                                                    for buffer_name in list(diffusion_model._buffers.keys()):
                                                                        buffer = diffusion_model._buffers[buffer_name]
                                                                        if buffer is not None and hasattr(buffer, 'data') and buffer.data is not None:
                                                                            try:
                                                                                del buffer.data
                                                                            except Exception:
                                                                                pass
                                                                    diffusion_model._buffers.clear()
                                                                except Exception:
                                                                    pass
                                                            
                                                            # DO NOT recursively delete submodule parameters - it breaks model structure
                                                            # Submodules contain essential parameters like Linear.weight that must be preserved
                                                            # Only top-level parameters are deleted above (with recurse=False)
                                                                
                                                            if params_deleted > 0:
                                                                print(f"Nunchaku: Deleted {params_deleted} total parameters from ComfyUI model (diffusion_model)")
                                                            if buffers_deleted > 0:
                                                                print(f"Nunchaku: Deleted {buffers_deleted} total buffers from ComfyUI model (diffusion_model)")
                                                            # Module structure (_modules) is preserved for re-loading via load_state_dict
                                                        except Exception as e:
                                                            print(f"Nunchaku: Warning: Failed to delete model internal state: {e}")
                                                        
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
                                    # Check if obj is a NunchakuSDXL instance (SDXL model wrapper)
                                    if nunchaku_sdxl_class is not None and isinstance(obj, nunchaku_sdxl_class):
                                        if hasattr(obj, 'diffusion_model'):
                                            diffusion_model = obj.diffusion_model
                                            for model_type in nunchaku_model_types:
                                                if isinstance(diffusion_model, model_type):
                                                    models_found_in_gc += 1
                                                    try:
                                                        print(f"Nunchaku: Found NunchakuSDXL instance with {model_type.__name__} in gc.get_objects() (id: {id(obj)})")
                                                        # Disable CPU offload first if enabled
                                                        if hasattr(diffusion_model, 'set_offload'):
                                                            try:
                                                                if hasattr(diffusion_model, 'offload') and diffusion_model.offload:
                                                                    print(f"Nunchaku: Disabling CPU offload for SDXL diffusion_model in gc.get_objects()")
                                                                    diffusion_model.set_offload(False)
                                                                    print(f"Nunchaku: CPU offload disabled for SDXL diffusion_model in gc.get_objects()")
                                                            except Exception as e:
                                                                print(f"Nunchaku: Warning: Failed to disable CPU offload: {e}")
                                                        
                                                        # Clear offload_manager
                                                        if hasattr(diffusion_model, 'offload_manager') and diffusion_model.offload_manager is not None:
                                                            try:
                                                                print(f"Nunchaku: Clearing offload_manager for SDXL diffusion_model in gc.get_objects()")
                                                                diffusion_model.offload_manager = None
                                                                print(f"Nunchaku: offload_manager cleared for SDXL diffusion_model in gc.get_objects()")
                                                            except Exception as e:
                                                                print(f"Nunchaku: Warning: Failed to clear offload_manager: {e}")
                                                        
                                                        # Delete SDXL diffusion_model data directly
                                                        print(f"Nunchaku: Attempting to delete SDXL diffusion_model data...")
                                                        
                                                        # Delete SDXL diffusion_model data
                                                        try:
                                                            params_deleted = 0
                                                            buffers_deleted = 0
                                                            
                                                            # Delete parameters (only top-level, not submodules)
                                                            if hasattr(diffusion_model, 'named_parameters'):
                                                                for name, param in list(diffusion_model.named_parameters(recurse=False)):
                                                                    if param is not None:
                                                                        try:
                                                                            if hasattr(param, 'data') and param.data is not None:
                                                                                del param.data
                                                                                params_deleted += 1
                                                                        except Exception:
                                                                            pass
                                                            
                                                            # Delete buffers (only top-level, not submodules)
                                                            if hasattr(diffusion_model, 'named_buffers'):
                                                                for name, buffer in list(diffusion_model.named_buffers(recurse=False)):
                                                                    if buffer is not None:
                                                                        try:
                                                                            if hasattr(buffer, 'data') and buffer.data is not None:
                                                                                del buffer.data
                                                                                buffers_deleted += 1
                                                                        except Exception:
                                                                            pass
                                                            
                                                            # Clear _parameters and _buffers dicts
                                                            if hasattr(diffusion_model, '_parameters'):
                                                                try:
                                                                    for param_name in list(diffusion_model._parameters.keys()):
                                                                        param = diffusion_model._parameters[param_name]
                                                                        if param is not None and hasattr(param, 'data') and param.data is not None:
                                                                            try:
                                                                                del param.data
                                                                            except Exception:
                                                                                pass
                                                                    diffusion_model._parameters.clear()
                                                                except Exception:
                                                                    pass
                                                            
                                                            if hasattr(diffusion_model, '_buffers'):
                                                                try:
                                                                    for buffer_name in list(diffusion_model._buffers.keys()):
                                                                        buffer = diffusion_model._buffers[buffer_name]
                                                                        if buffer is not None and hasattr(buffer, 'data') and buffer.data is not None:
                                                                            try:
                                                                                del buffer.data
                                                                            except Exception:
                                                                                pass
                                                                    diffusion_model._buffers.clear()
                                                                except Exception:
                                                                    pass
                                                            
                                                            # DO NOT recursively delete submodule parameters - it breaks model structure
                                                            # Submodules contain essential parameters like Linear.weight that must be preserved
                                                            # Only top-level parameters are deleted above (with recurse=False)
                                                                
                                                            if params_deleted > 0:
                                                                print(f"Nunchaku: Deleted {params_deleted} total parameters from SDXL diffusion_model in gc.get_objects()")
                                                            if buffers_deleted > 0:
                                                                print(f"Nunchaku: Deleted {buffers_deleted} total buffers from SDXL diffusion_model in gc.get_objects()")
                                                            # Module structure (_modules) is preserved for re-loading via load_state_dict
                                                            print(f"Nunchaku: Deleted SDXL diffusion_model internal state from gc.get_objects()")
                                                        except Exception as e:
                                                            print(f"Nunchaku: Warning: Failed to delete SDXL diffusion_model internal state from gc.get_objects(): {e}")
                                                        
                                                        print(f"Nunchaku: Successfully cleared SDXL model ({model_type.__name__}) from gc.get_objects()")
                                                    except Exception as e:
                                                        print(f"Nunchaku: Error clearing SDXL model from gc.get_objects(): {e}")
                                                        import traceback
                                                        print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                                    
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
                                                
                                                # Delete model data directly
                                                print(f"Nunchaku: Attempting to delete model data...")
                                                
                                                # Delete model's internal state
                                                try:
                                                    params_deleted = 0
                                                    buffers_deleted = 0
                                                    
                                                    # Delete parameters (only top-level, not submodules)
                                                    if hasattr(obj, 'named_parameters'):
                                                        for name, param in list(obj.named_parameters(recurse=False)):
                                                            if param is not None:
                                                                try:
                                                                    if hasattr(param, 'data') and param.data is not None:
                                                                        del param.data
                                                                        params_deleted += 1
                                                                except Exception:
                                                                    pass
                                                        
                                                    # Delete buffers (only top-level, not submodules)
                                                    if hasattr(obj, 'named_buffers'):
                                                        for name, buffer in list(obj.named_buffers(recurse=False)):
                                                            if buffer is not None:
                                                                try:
                                                                    if hasattr(buffer, 'data') and buffer.data is not None:
                                                                        del buffer.data
                                                                        buffers_deleted += 1
                                                                except Exception:
                                                                    pass
                                                    
                                                    # DO NOT recursively delete submodule parameters - it breaks model structure
                                                    # Submodules contain essential parameters like Linear.weight that must be preserved
                                                    # Only top-level parameters are deleted above (with recurse=False)
                                                    
                                                    # Clear _parameters and _buffers dicts directly
                                                    if hasattr(obj, '_parameters'):
                                                        try:
                                                            for param_name in list(obj._parameters.keys()):
                                                                param = obj._parameters[param_name]
                                                                if param is not None and hasattr(param, 'data') and param.data is not None:
                                                                    try:
                                                                        del param.data
                                                                    except Exception:
                                                                        pass
                                                            obj._parameters.clear()
                                                        except Exception:
                                                            pass
                                                    
                                                    if hasattr(obj, '_buffers'):
                                                        try:
                                                            for buffer_name in list(obj._buffers.keys()):
                                                                buffer = obj._buffers[buffer_name]
                                                                if buffer is not None and hasattr(buffer, 'data') and buffer.data is not None:
                                                                    try:
                                                                        del buffer.data
                                                                    except Exception:
                                                                        pass
                                                            obj._buffers.clear()
                                                        except Exception:
                                                            pass
                                                    
                                                    # Try to clear any cached data or temporary attributes that might hold VRAM
                                                    try:
                                                        # Clear any cache-related attributes
                                                        cache_attrs = ['_cache', 'cache', '_state_dict_cache', 'state_dict_cache', '_non_persistent_buffers_set']
                                                        for cache_attr in cache_attrs:
                                                            if hasattr(obj, cache_attr):
                                                                try:
                                                                    cache_val = getattr(obj, cache_attr)
                                                                    if cache_val is not None:
                                                                        if isinstance(cache_val, (dict, set)):
                                                                            cache_val.clear()
                                                                        elif hasattr(cache_val, 'clear'):
                                                                            cache_val.clear()
                                                                        setattr(obj, cache_attr, None)
                                                                        print(f"Nunchaku: Cleared {cache_attr} from gc.get_objects() model")
                                                                except Exception:
                                                                    pass
                                                    except Exception:
                                                        pass
                                                    
                                                    if params_deleted > 0:
                                                        print(f"Nunchaku: Deleted {params_deleted} parameters from gc.get_objects() model")
                                                    if buffers_deleted > 0:
                                                        print(f"Nunchaku: Deleted {buffers_deleted} buffers from gc.get_objects() model")
                                                    # DO NOT clear _modules - it contains module structure
                                                    # if hasattr(obj, '_modules'):  # Removed
                                                    #     print(f"Nunchaku: Cleared _modules dict from gc.get_objects() model")  # Removed
                                                    print(f"Nunchaku: Cleared model internal state from gc.get_objects()")
                                                except Exception as e:
                                                    print(f"Nunchaku: Warning: Failed to clear model internal state from gc.get_objects(): {e}")
                                                    import traceback
                                                    print(f"Nunchaku: Traceback: {traceback.format_exc()}")
                                                
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
                    
                    # Force garbage collection and clear GPU cache (more aggressively)
                    print("Nunchaku: Running garbage collection...")
                    gc.collect()
                    gc.collect()  # Run twice to ensure cleanup
                    gc.collect()  # Run third time for more aggressive cleanup
                    if torch.cuda.is_available():
                        print("Nunchaku: Clearing CUDA cache...")
                        # Clear cache for all devices (more aggressively)
                        for device_idx in range(torch.cuda.device_count()):
                            with torch.cuda.device(device_idx):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                torch.cuda.empty_cache()  # Run twice
                        torch.cuda.synchronize()
                        # Additional cleanup: try to free memory fragments
                        try:
                            import torch._C
                            if hasattr(torch._C, '_cuda_emptyCache'):
                                torch._C._cuda_emptyCache()
                        except Exception:
                            pass
                        print("Nunchaku: CUDA cache cleared for all devices (aggressive cleanup)")
                    
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

        # Purge HSWQ whole path (models / INT8 / PinCache / Detailer / pins / kitchen)
        # + orphaned ComfyUI cudaHostRegister / CUDA tensors (Task Manager dedicated GPU mem)
        # HSWQ nuclear runs ONLY when the HSWQ toggle is explicitly ON — never auto-arm.
        if purge_hswq_int8:
            try:
                print("HSWQ INT8/NVFP4: Starting purge process...")
                hswq_cleared = 0
                bytes_killed = 0
                pins_unregistered = 0
                cuda_tensors_killed = 0
                patchers_unloaded = 0

                def _sys_modules():
                    # Never use a local "import sys" in purge_vram — it shadows module-level sys.
                    return list(sys.modules.items())

                def _mem_diag(tag: str) -> None:
                    try:
                        import comfy.model_management as mm
                        total_pin = int(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0)
                        pin_entries = len(getattr(mm, "PINNED_MEMORY", {}) or {})
                        print(
                            f"HSWQ INT8/NVFP4: [{tag}] TOTAL_PINNED_MEMORY="
                            f"{total_pin / (1024 * 1024):.1f} MB entries={pin_entries}"
                        )
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: [{tag}] pin diag failed: {e}")
                    if torch.cuda.is_available():
                        try:
                            for di in range(torch.cuda.device_count()):
                                alloc = torch.cuda.memory_allocated(di)
                                reserved = torch.cuda.memory_reserved(di)
                                free_b, total_b = torch.cuda.mem_get_info(di)
                                used_sys = max(0, total_b - free_b)
                                print(
                                    f"HSWQ INT8/NVFP4: [{tag}] cuda:{di} "
                                    f"allocated={alloc / (1024 ** 3):.2f}GB "
                                    f"reserved={reserved / (1024 ** 3):.2f}GB "
                                    f"sys_used={used_sys / (1024 ** 3):.2f}GB"
                                )
                        except Exception as e:
                            print(f"HSWQ INT8/NVFP4: [{tag}] cuda diag failed: {e}")

                def _drain_hswq_pin_cache() -> int:
                    drained = 0

                    def _call_purge(mod, mod_name: str) -> int:
                        fn = getattr(mod, "purge_pin_cache", None)
                        if callable(fn):
                            got = int(fn() or 0)
                            print(
                                f"HSWQ INT8/NVFP4: PinCache purged via {mod_name}: "
                                f"{got / (1024 * 1024):.1f} MB"
                            )
                            return got
                        pool = getattr(mod, "_PIN_BUFFER_POOL", None)
                        total = int(getattr(mod, "_PIN_CACHE_TOTAL", 0) or 0)
                        drain = getattr(mod, "_drain_pool", None)
                        if callable(drain):
                            setattr(mod, "_active", False)
                            setattr(mod, "_depth", 0)
                            drain()
                            print(
                                f"HSWQ INT8/NVFP4: PinCache _drain_pool via {mod_name}: "
                                f"{total / (1024 * 1024):.1f} MB"
                            )
                            return total
                        if pool is not None:
                            pool.clear()
                            setattr(mod, "_PIN_CACHE_TOTAL", 0)
                            print(f"HSWQ INT8/NVFP4: PinCache pool cleared via {mod_name}")
                            return total
                        return 0

                    found = False
                    for mod_name, mod in _sys_modules():
                        if mod is None or "hswq_pin_cache" not in str(mod_name):
                            continue
                        try:
                            drained += _call_purge(mod, str(mod_name))
                            found = True
                        except Exception as e:
                            print(f"HSWQ INT8/NVFP4: PinCache purge via {mod_name} failed: {e}")
                    if found:
                        return drained

                    # Force-import: Detailer scope may have ended (deactivate drained
                    # tracking) or module never stayed in sys.modules under expected name.
                    try:
                        import importlib.util
                        pkg_dir = os.path.dirname(os.path.abspath(__file__))
                        # purge_vram.py at DistTorch root → custom_nodes is parent
                        cn_root = os.path.dirname(pkg_dir)
                        if os.path.basename(pkg_dir) == "nodes":
                            cn_root = os.path.dirname(os.path.dirname(pkg_dir))
                        candidates = [
                            os.path.join(
                                cn_root,
                                "ComfyUI-nunchaku-unofficial-loader",
                                "nodes",
                                "hswq_pin_cache.py",
                            ),
                            os.path.join(
                                cn_root,
                                "comfyui-nunchaku-unofficial-loader",
                                "nodes",
                                "hswq_pin_cache.py",
                            ),
                        ]
                        for pin_py in candidates:
                            if not os.path.isfile(pin_py):
                                continue
                            print(f"HSWQ INT8/NVFP4: Force-import PinCache from {pin_py}")
                            spec = importlib.util.spec_from_file_location(
                                "hswq_pin_cache_force_purge", pin_py
                            )
                            if spec is None or spec.loader is None:
                                continue
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            sys.modules["hswq_pin_cache_force_purge"] = mod
                            drained += _call_purge(mod, pin_py)
                            return drained
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: PinCache force-import failed: {e}")

                    print("HSWQ INT8/NVFP4: PinCache module not loaded (nothing to drain)")
                    return drained

                def _purge_detailer_segs_and_executor_cache() -> int:
                    """Drop Impact SEGS / large IMAGE held in PromptExecutor caches now.

                    Do NOT call PromptExecutor.reset() mid-prompt: reset() replaces
                    CacheSet with a fresh RAMPressureCache that has never run
                    set_prompt(), so cache_key_set is missing and the next
                    caches.outputs.get() raises AttributeError.

                    MultiGPU correctly only sets free_memory (reset after prompt).
                    Here we clear .cache / .subcaches in place so the current
                    prompt's cache_key_set / initialized state stay valid.
                    """
                    freed_hint = 0
                    cleared_entries = 0
                    executor_n = 0
                    try:
                        for obj in gc.get_objects():
                            if type(obj).__name__ != "PromptExecutor":
                                continue
                            executor_n += 1
                            try:
                                caches = getattr(obj, "caches", None)
                                if caches is None:
                                    continue
                                for cache in getattr(caches, "all", None) or []:
                                    try:
                                        cdict = getattr(cache, "cache", None)
                                        if isinstance(cdict, dict) and cdict:
                                            cleared_entries += len(cdict)
                                            cdict.clear()
                                        sub = getattr(cache, "subcaches", None)
                                        if isinstance(sub, dict) and sub:
                                            cleared_entries += len(sub)
                                            sub.clear()
                                        for attr in (
                                            "timestamps",
                                            "used_generation",
                                            "children",
                                        ):
                                            bag = getattr(cache, attr, None)
                                            if isinstance(bag, dict) and bag:
                                                bag.clear()
                                    except Exception as e:
                                        print(
                                            f"HSWQ INT8/NVFP4: in-place cache clear "
                                            f"failed: {e}"
                                        )
                            except Exception as e:
                                print(
                                    f"HSWQ INT8/NVFP4: PromptExecutor cache clear "
                                    f"failed: {e}"
                                )
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: PromptExecutor scan failed: {e}")
                    print(
                        f"HSWQ INT8/NVFP4: PromptExecutor in-place cache clear "
                        f"executors={executor_n} entries={cleared_entries}"
                    )

                    impact_cleared = 0
                    for mod_name, mod in _sys_modules():
                        if mod is None:
                            continue
                        n = str(mod_name).replace("\\", "/")
                        if "impact/core" not in n and not n.endswith("impact.core"):
                            if "impact.core" not in n:
                                continue
                        d = getattr(mod, "__dict__", None)
                        if not isinstance(d, dict):
                            continue
                        for attr in (
                            "preview_bridge_cache",
                            "preview_bridge_last_mask_cache",
                            "preview_bridge_image_id_map",
                            "preview_bridge_image_name_map",
                        ):
                            bag = d.get(attr)
                            if isinstance(bag, dict) and bag:
                                impact_cleared += len(bag)
                                bag.clear()
                    if impact_cleared:
                        print(
                            f"HSWQ INT8/NVFP4: Impact preview/SEG bridge caches cleared "
                            f"entries={impact_cleared}"
                        )

                    # HSWQ Batched Detailer / Impact Detailer instances holding SEGS crops
                    detailer_n = 0
                    detailer_tensors = 0
                    try:
                        for obj in gc.get_objects():
                            try:
                                tn = type(obj).__name__
                                if (
                                    "Detailer" not in tn
                                    and "SEGS" not in tn
                                    and "Segs" not in tn
                                ):
                                    continue
                                detailer_n += 1
                                d = getattr(obj, "__dict__", None)
                                if not isinstance(d, dict):
                                    continue
                                for attr, val in list(d.items()):
                                    try:
                                        if torch.is_tensor(val):
                                            nbytes = int(getattr(val, "nbytes", 0) or 0)
                                            if nbytes < 1024 * 1024:
                                                continue
                                            freed_hint += _kill_tensor_storage(val)
                                            detailer_tensors += 1
                                        elif isinstance(val, (list, tuple)):
                                            for item in val:
                                                if not torch.is_tensor(item):
                                                    continue
                                                nbytes = int(
                                                    getattr(item, "nbytes", 0) or 0
                                                )
                                                if nbytes < 1024 * 1024:
                                                    continue
                                                freed_hint += _kill_tensor_storage(item)
                                                detailer_tensors += 1
                                    except Exception:
                                        pass
                            except Exception:
                                continue
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: Detailer object sweep failed: {e}")
                    if detailer_n or detailer_tensors:
                        print(
                            f"HSWQ INT8/NVFP4: Detailer/SEGS object sweep "
                            f"objects={detailer_n} tensors={detailer_tensors}"
                        )

                    # Kill large CUDA / pinned tensors still reachable (SEG crops etc.)
                    tensor_killed = 0
                    try:
                        for obj in gc.get_objects():
                            try:
                                if not torch.is_tensor(obj):
                                    continue
                                nbytes = int(getattr(obj, "nbytes", 0) or 0)
                                if nbytes < 4 * 1024 * 1024:
                                    continue
                                pinned = False
                                try:
                                    pinned = bool(obj.is_pinned())
                                except Exception:
                                    pass
                                on_cuda = False
                                try:
                                    on_cuda = bool(obj.is_cuda)
                                except Exception:
                                    pass
                                if not pinned and not on_cuda:
                                    continue
                                freed_hint += _kill_tensor_storage(obj)
                                tensor_killed += 1
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: SEGS tensor sweep failed: {e}")
                    print(
                        f"HSWQ INT8/NVFP4: Detailer SEGS/cache sweep "
                        f"tensors_touched={tensor_killed} "
                        f"approx={freed_hint / (1024 * 1024):.1f} MB"
                    )
                    return freed_hint

                def _reset_comfy_kitchen_cuda_caches() -> None:
                    """Drop ALL HSWQ residual pools / module caches after nuclear kill.

                    Covers the whole HSWQ surface — not NVFP4 alone:

                    - INT8 (Linear protect ConvRot + Conv2d ``_hswq_convrot`` + LoRA bake)
                    - NVFP4 / ZI ConvRot (parity H, TC arms, runtime pools)
                    - Detailer path leaves SEGS via PromptExecutor (cleared separately)
                    - PinCache / kitchen / Hadamard globals (INT8 + NVFP4)

                    Independent of HSWQ Loader clear API presence or return values.
                    Loader ``clear_*`` is best-effort; local in-place + gc always run.
                    """
                    cleared = []

                    def _safe_hasattr(obj, name: str) -> bool:
                        # Some third-party modules (e.g. seedvr2 compatibility wrappers)
                        # raise ImportError from __getattr__; bare hasattr aborts purge.
                        try:
                            return hasattr(obj, name)
                        except Exception:
                            return False

                    def _safe_getattr(obj, name: str, default=None):
                        try:
                            return getattr(obj, name, default)
                        except Exception:
                            return default

                    def _drop_attr(obj, name: str) -> bool:
                        if not _safe_hasattr(obj, name):
                            return False
                        try:
                            delattr(obj, name)
                            return True
                        except Exception:
                            try:
                                setattr(obj, name, None)
                                return True
                            except Exception:
                                return False

                    def _clear_dict_attr(mod, attr: str) -> int:
                        bag = _safe_getattr(mod, attr, None)
                        if isinstance(bag, dict) and bag:
                            n = len(bag)
                            bag.clear()
                            return n
                        return 0

                    def _empty_cuda_tensor(t) -> None:
                        if t is None:
                            return
                        if torch.is_tensor(t):
                            try:
                                data = getattr(t, "data", t)
                                if not bool(getattr(data, "is_cuda", False)):
                                    return
                                empty = torch.empty(0, dtype=data.dtype, device=data.device)
                                t.data = empty
                            except Exception:
                                pass
                            return
                        # Nested containers (e.g. _hswq_krea2_lora_res = [(mat_dn, mat_up, scale), ...])
                        if isinstance(t, (list, tuple)):
                            for item in t:
                                if item is not None:
                                    _empty_cuda_tensor(item)
                        elif isinstance(t, dict):
                            for v in t.values():
                                if v is not None:
                                    _empty_cuda_tensor(v)

                    # Known residual names (INT8 + NVFP4 + bake + forward caches).
                    # Stray walk below also drops every other ``_hswq_*`` on Modules.
                    _hswq_drop_attrs = (
                        # NVFP4 / ZI ConvRot
                        "_hswq_nvfp4_parity_H",
                        "_hswq_nvfp4_H",
                        "_hswq_nvfp4_w_plain",
                        "_hswq_nvfp4_alpha",
                        "_hswq_nvfp4_no_cudagraph",
                        "_hswq_nvfp4_convrot",
                        "_hswq_nvfp4_convrot_groupsize",
                        "_hswq_nvfp4_convrot_parity",
                        "_hswq_nvfp4",
                        "_hswq_nvfp4_act_scale",
                        "_hswq_nvfp4_scale_placeholder",
                        "_hswq_nvfp4_scale_from_ckpt",
                        "_hswq_zi_nvfp4_baked_keys",
                        "_hswq_zi_nvfp4_baked_uuid",
                        # INT8 Linear protect ConvRot
                        "_hswq_int8_convrot",
                        "_hswq_int8_convrot_groupsize",
                        "_hswq_int8_baked_keys",
                        "_hswq_int8_baked_uuid",
                        # INT8 Conv2d ConvRot (comfy_quant_int8 QuantConv2d)
                        "_hswq_convrot",
                        "_hswq_convrot_groupsize",
                        # Krea2 ConvRot NVFP4 (bake bookkeeping / pack stamps / LoRA residuals)
                        "_hswq_krea2_nvfp4_pack",
                        "_hswq_krea2_nvfp4_baked_keys",
                        "_hswq_krea2_nvfp4_baked_uuid",
                        "_hswq_krea2_lora_res",
                        "_hswq_krea2_lora_res_gpu",
                        "_hswq_krea2_tc",
                        "_hswq_krea2_stack",
                        "_hswq_krea2_full_load",
                        "_hswq_krea2_oldquants",
                        "_hswq_krea2_prev_oldquants",
                        "_hswq_krea2_txtlayers_fix",
                        "_hswq_krea2_prev_dynamic_load",
                    )

                    # --- comfy_kitchen ---
                    try:
                        import comfy_kitchen.backends.cuda as ck_cuda
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: comfy_kitchen cuda import skipped: {e}")
                        ck_cuda = None
                    if ck_cuda is not None:
                        for attr in ("_cublas_workspaces", "_empty_cuda_tensors"):
                            n = _clear_dict_attr(ck_cuda, attr)
                            if n:
                                cleared.append(f"{attr}={n}")

                    # --- Peel Z Image parity / bake hooks so SDXL TC is not poisoned ---
                    # Loader also does this on SDXL load; Distorch must do it on purge so
                    # the next model (SDXL after ZI) never inherits comfy_parity / ZI bake.
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        nlow = str(name).replace("\\", "/").lower()
                        if not (
                            "nvfp4" in nlow
                            or "zimage_nvfp4" in nlow
                            or "comfy_quant_nvfp4" in nlow
                        ):
                            continue
                        for api_name in (
                            "_clear_zimage_parity_contamination_for_sdxl",
                            "restore_nvfp4_tc_product_stack",
                            "uninstall_zimage_nvfp4_lora_bake",
                            # Krea2 ConvRot NVFP4 (Dynamic.load / load_models_gpu bake hooks)
                            "uninstall_krea2_nvfp4_lora_bake",
                            "reset_krea2_nvfp4_lora_bake_log_counters",
                        ):
                            fn = _safe_getattr(mod, api_name, None)
                            if not callable(fn):
                                continue
                            try:
                                ret = fn()
                                cleared.append(f"{api_name}@{name}={ret!r}")
                                print(
                                    "HSWQ INT8/NVFP4: HSWQ stack peel "
                                    f"{api_name} via {name} -> {ret!r}"
                                )
                            except Exception as e_peel:
                                print(
                                    f"HSWQ INT8/NVFP4: {api_name} failed "
                                    f"({name}): {e_peel}"
                                )

                    # --- Peel ZI INT8-protect load arm on ops._load_quantized_module ---
                    # SDXL INT8 ConvRot and ZI INT8 protect share conf shape
                    # (int8_tensorwise + convrot). Loader peel may leave
                    # _hswq_int8_protect_in_load / _hswq_int8_protect_arm_v2
                    # (arm freevar is ``cur``, not orig_load) or PRODUCT wrapping
                    # that arm — then _arm_int8_protect_convrot_after_stock_load
                    # fires on SDXL load (Params.convrot=False / VER=8 bake).
                    try:
                        import comfy.ops as _ops_peel_load

                        def _closure_load_cell(fn, name: str):
                            try:
                                cells = fn.__closure__ or ()
                                for n, c in zip(fn.__code__.co_freevars, cells):
                                    if n == name:
                                        return c.cell_contents
                            except Exception:
                                return None
                            return None

                        def _is_foreign_int8_protect_load(fn) -> bool:
                            return bool(
                                getattr(fn, "_hswq_nvfp4_comfy_only", False)
                                or getattr(fn, "_hswq_int8_protect_in_load", False)
                                or getattr(fn, "_hswq_int8_protect_arm_v2", False)
                                or getattr(fn, "_hswq_int8_decode_patched", False)
                                or getattr(fn, "_hswq_krea2_full_load", False)
                                or (
                                    getattr(fn, "_hswq_nvfp4_full_load", False)
                                    and not getattr(
                                        fn, "_hswq_nvfp4_product_tc", False
                                    )
                                )
                            )

                        def _next_load_under(fn):
                            for name in (
                                "cur",
                                "orig_load",
                                "original_load",
                                "_orig_load",
                            ):
                                nxt = _closure_load_cell(fn, name)
                                if nxt is not None:
                                    return nxt
                            return getattr(fn, "_hswq_nvfp4_orig_load", None)

                        _peeled_load_n = 0
                        _cur_l = getattr(
                            _ops_peel_load, "_load_quantized_module", None
                        )
                        _seen_l: set[int] = set()
                        while (
                            _cur_l is not None
                            and id(_cur_l) not in _seen_l
                            and _peeled_load_n < 16
                        ):
                            _seen_l.add(id(_cur_l))
                            if getattr(_cur_l, "_hswq_nvfp4_product_tc", False):
                                _under = _next_load_under(_cur_l)
                                if _under is not None and _is_foreign_int8_protect_load(
                                    _under
                                ):
                                    _ops_peel_load._load_quantized_module = _under
                                    _peeled_load_n += 1
                                    _cur_l = _under
                                    continue
                                break
                            if not _is_foreign_int8_protect_load(_cur_l):
                                break
                            _nxt_l = _next_load_under(_cur_l)
                            if _nxt_l is None or _nxt_l is _cur_l:
                                break
                            _ops_peel_load._load_quantized_module = _nxt_l
                            _peeled_load_n += 1
                            _cur_l = _nxt_l
                        if _peeled_load_n:
                            cleared.append(
                                f"int8_protect_load_peel={_peeled_load_n}"
                            )
                            print(
                                "HSWQ INT8/NVFP4: peeled ZI INT8-protect "
                                f"load overlay layers={_peeled_load_n}"
                            )
                    except Exception as e_load_peel:
                        print(
                            "HSWQ INT8/NVFP4: INT8-protect load peel failed: "
                            f"{e_load_peel}"
                        )

                    # --- Peel Krea2 ConvRot NVFP4 stack (mixed_precision_ops +
                    #     convert_old_quants + detect_unet_config). uninstall_krea2_nvfp4_lora_bake only
                    #     peels Dynamic.load / load_models_gpu, not these ops wraps. ---
                    try:
                        import comfy.ops as _ops_peel_krea2_mp
                        import comfy.utils as _utils_peel_krea2
                        import comfy.model_detection as _md_peel_krea2

                        def _peel_krea2_mp_once():
                            cur = getattr(_ops_peel_krea2_mp, "mixed_precision_ops", None)
                            seen = set()
                            peeled = 0
                            while cur is not None and callable(cur) and id(cur) not in seen:
                                seen.add(id(cur))
                                if not getattr(cur, "_hswq_krea2_stack", False):
                                    break
                                nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None)
                                if nxt is None or nxt is cur:
                                    break
                                _ops_peel_krea2_mp.mixed_precision_ops = nxt
                                peeled += 1
                                cur = nxt
                            return peeled

                        def _peel_krea2_oldquants_once():
                            cur = getattr(_utils_peel_krea2, "convert_old_quants", None)
                            seen = set()
                            peeled = 0
                            while cur is not None and callable(cur) and id(cur) not in seen:
                                seen.add(id(cur))
                                if not getattr(cur, "_hswq_krea2_oldquants", False):
                                    break
                                nxt = getattr(cur, "_hswq_krea2_prev_oldquants", None)
                                if nxt is None or nxt is cur:
                                    break
                                _utils_peel_krea2.convert_old_quants = nxt
                                peeled += 1
                                cur = nxt
                            return peeled

                        def _peel_krea2_txtlayers_once():
                            cur = getattr(_md_peel_krea2, "detect_unet_config", None)
                            seen = set()
                            peeled = 0
                            while cur is not None and callable(cur) and id(cur) not in seen:
                                seen.add(id(cur))
                                if not getattr(cur, "_hswq_krea2_txtlayers_fix", False):
                                    break
                                nxt = _closure_load_cell(cur, "_prev_detect_txt")
                                if nxt is None or nxt is cur:
                                    break
                                _md_peel_krea2.detect_unet_config = nxt
                                peeled += 1
                                cur = nxt
                            return peeled

                        _mp_peeled_k2 = _peel_krea2_mp_once()
                        _oq_peeled_k2 = _peel_krea2_oldquants_once()
                        _txt_peeled_k2 = _peel_krea2_txtlayers_once()
                        if _mp_peeled_k2:
                            cleared.append(f"krea2_mp_stack_peel={_mp_peeled_k2}")
                            print(
                                "HSWQ INT8/NVFP4: peeled Krea2 mixed_precision_ops "
                                f"stack layers={_mp_peeled_k2}"
                            )
                        if _oq_peeled_k2:
                            cleared.append(f"krea2_oldquants_peel={_oq_peeled_k2}")
                            print(
                                "HSWQ INT8/NVFP4: peeled Krea2 convert_old_quants "
                                f"layers={_oq_peeled_k2}"
                            )
                        if _txt_peeled_k2:
                            cleared.append(f"krea2_txtlayers_peel={_txt_peeled_k2}")
                            print(
                                "HSWQ INT8/NVFP4: peeled Krea2 detect_unet_config txtlayers "
                                f"fix layers={_txt_peeled_k2}"
                            )
                    except Exception as e_krea2_peel:
                        print(
                            "HSWQ INT8/NVFP4: Krea2 stack peel failed: "
                            f"{e_krea2_peel}"
                        )

                    # --- Peel ZI/NVFP4 Linear.convert_weight / set_weight wraps ---
                    # uninstall_zimage_nvfp4_lora_bake only peels Dynamic.load /
                    # load_models_gpu. ZI attach_nvfp4_linear_lora_bake mutates
                    # MixedPrecision Linear in place; after ZI→SDXL the third SDXL
                    # bake still logs ConvRot int8_protect convert/set and LoRA
                    # strength dies. Restore stock (INT8) convert/set here.
                    def _peel_lora_bake_wrap_local(fn):
                        cur = fn
                        for _ in range(8):
                            if not callable(cur):
                                return cur
                            if int(
                                getattr(cur, "_hswq_nvfp4_lora_bake_ver", 0) or 0
                            ) <= 0:
                                return cur
                            stock = getattr(
                                cur, "_hswq_nvfp4_lora_bake_stock", None
                            )
                            if stock is not None and stock is not cur:
                                cur = stock
                                continue
                            closure = getattr(cur, "__closure__", None)
                            code = getattr(cur, "__code__", None)
                            if closure is None or code is None:
                                return cur
                            names = code.co_freevars
                            nxt = None
                            for i, nfree in enumerate(names):
                                if nfree in (
                                    "stock_convert_weight",
                                    "stock_set_weight",
                                ):
                                    nxt = closure[i].cell_contents
                                    break
                            if nxt is None or nxt is cur:
                                return cur
                            cur = nxt
                        return cur

                    peel_fn = _peel_lora_bake_wrap_local
                    for _pname, _pmod in list(__import__("sys").modules.items()):
                        if _pmod is None:
                            continue
                        _plow = str(_pname).replace("\\", "/").lower()
                        if (
                            "zi_nvfp4_forward" not in _plow
                            and not _plow.endswith("nvfp4_forward")
                            and ".nvfp4_forward" not in _plow
                        ):
                            continue
                        _helper = _safe_getattr(
                            _pmod, "_peel_lora_bake_wrap", None
                        )
                        if callable(_helper):
                            peel_fn = _helper
                            break

                    _peeled_lin_ids = set()

                    def _peel_linear_lora_bake(Lin, label: str) -> None:
                        if Lin is None or not isinstance(Lin, type):
                            return
                        lid = id(Lin)
                        if lid in _peeled_lin_ids:
                            return
                        for meth in ("convert_weight", "set_weight"):
                            fn = getattr(Lin, meth, None)
                            if not callable(fn):
                                continue
                            ver = int(
                                getattr(fn, "_hswq_nvfp4_lora_bake_ver", 0) or 0
                            )
                            if ver <= 0:
                                continue
                            try:
                                peeled = peel_fn(fn)
                            except Exception as e_peel_fn:
                                print(
                                    "HSWQ INT8/NVFP4: Linear LoRA bake peel "
                                    f"helper failed ({label}.{meth}): {e_peel_fn}"
                                )
                                continue
                            if peeled is fn or not callable(peeled):
                                continue
                            try:
                                setattr(Lin, meth, peeled)
                                _peeled_lin_ids.add(lid)
                                cleared.append(
                                    f"linear_lora_bake_peel@{label}.{meth}"
                                    f"=ver{ver}"
                                )
                                print(
                                    "HSWQ INT8/NVFP4: Peeled NVFP4/ZI Linear "
                                    f"LoRA bake wrap {label}.{meth} "
                                    f"(was ver={ver})"
                                )
                            except Exception as e_set:
                                print(
                                    "HSWQ INT8/NVFP4: Linear LoRA bake peel "
                                    f"setattr failed ({label}.{meth}): {e_set}"
                                )

                    try:
                        import comfy.ops as _comfy_ops

                        _peel_linear_lora_bake(
                            getattr(_comfy_ops, "Linear", None),
                            "comfy.ops.Linear",
                        )
                        for _an in dir(_comfy_ops):
                            try:
                                _obj = getattr(_comfy_ops, _an, None)
                            except Exception:
                                continue
                            if isinstance(_obj, type) and _safe_hasattr(
                                _obj, "convert_weight"
                            ):
                                _peel_linear_lora_bake(
                                    _obj, f"comfy.ops.{_an}"
                                )
                    except Exception as e_ops:
                        print(
                            "HSWQ INT8/NVFP4: comfy.ops Linear LoRA bake peel "
                            f"skipped: {e_ops}"
                        )

                    for _name, _mod in list(__import__("sys").modules.items()):
                        if _mod is None:
                            continue
                        _nlow = str(_name).replace("\\", "/").lower()
                        if not (
                            "comfy.ops" in _nlow
                            or "nvfp4" in _nlow
                            or "comfy_quant" in _nlow
                            or "zimage_nvfp4" in _nlow
                            or "hswq" in _nlow
                        ):
                            continue
                        try:
                            _Lin = _safe_getattr(_mod, "Linear", None)
                            if isinstance(_Lin, type):
                                _cvt = getattr(_Lin, "convert_weight", None)
                                if callable(_cvt) and int(
                                    getattr(
                                        _cvt, "_hswq_nvfp4_lora_bake_ver", 0
                                    )
                                    or 0
                                ) > 0:
                                    _peel_linear_lora_bake(
                                        _Lin, f"{_name}.Linear"
                                    )
                        except Exception:
                            pass

                    # Nested MixedPrecisionOps.Linear may not sit on a module.
                    try:
                        import gc as _gc_peel

                        for _obj in _gc_peel.get_objects():
                            if not isinstance(_obj, type):
                                continue
                            if getattr(_obj, "__name__", "") != "Linear":
                                continue
                            _modn = str(
                                getattr(_obj, "__module__", "") or ""
                            ).lower()
                            if not (
                                "comfy" in _modn
                                or "ops" in _modn
                                or "quant" in _modn
                            ):
                                continue
                            _cvt = getattr(_obj, "convert_weight", None)
                            if not (
                                callable(_cvt)
                                and int(
                                    getattr(
                                        _cvt, "_hswq_nvfp4_lora_bake_ver", 0
                                    )
                                    or 0
                                )
                                > 0
                            ):
                                continue
                            _peel_linear_lora_bake(
                                _obj, f"gc:{_modn}.Linear"
                            )
                    except Exception as e_gc_peel:
                        print(
                            "HSWQ INT8/NVFP4: gc Linear LoRA bake peel "
                            f"skipped: {e_gc_peel}"
                        )

                    # --- NVFP4 runtime pools + scale caches (SDXL + any twin) ---
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        try:
                            nlow = str(name).replace("\\", "/").lower()
                            name_hit = (
                                name.endswith("nvfp4_runtime")
                                or ".nvfp4_runtime" in name
                                or "nvfp4_runtime" in nlow
                            )
                            # Name-first: avoid probing unrelated modules whose
                            # __getattr__ raises (seedvr2 flashattention stubs).
                            if not name_hit and "nvfp4" not in nlow and "hswq" not in nlow:
                                continue
                            has_pool = (
                                _safe_hasattr(mod, "_ACT_Q_POOL")
                                or _safe_hasattr(mod, "_ROT_OUT_POOL")
                                or _safe_hasattr(mod, "_GRAPH_CACHE")
                                or _safe_hasattr(mod, "clear_nvfp4_runtime_pools")
                            )
                            if not name_hit and not has_pool:
                                continue
                            api_ok = False
                            fn = _safe_getattr(mod, "clear_nvfp4_runtime_pools", None)
                            if callable(fn):
                                try:
                                    fn()
                                    api_ok = True
                                    cleared.append(f"nvfp4_runtime_pools@{name}")
                                    print(
                                        "HSWQ INT8/NVFP4: Cleared HSWQ NVFP4 runtime pools / "
                                        f"CUDA graphs via {name}"
                                    )
                                except Exception as e2:
                                    print(
                                        f"HSWQ INT8/NVFP4: NVFP4 runtime pool clear failed "
                                        f"({name}): {e2}"
                                    )
                            n_pool = 0
                            for attr in (
                                "_ACT_Q_POOL",
                                "_ROT_OUT_POOL",
                                "_GRAPH_CACHE",
                                "_INV_NVFP4_AMAX_DENOM",
                                "_ONES_SCALE",
                            ):
                                n_pool += _clear_dict_attr(mod, attr)
                            if not api_ok:
                                cg = _safe_getattr(mod, "clear_nvfp4_cudagraphs", None)
                                if callable(cg):
                                    try:
                                        cg()
                                    except Exception:
                                        pass
                            if n_pool:
                                cleared.append(f"nvfp4_runtime_inplace={n_pool}@{name}")
                                if not api_ok:
                                    print(
                                        "HSWQ INT8/NVFP4: Cleared NVFP4 runtime dicts "
                                        f"in-place (n={n_pool}) via {name}"
                                    )
                        except Exception as e_pool:
                            print(
                                f"HSWQ INT8/NVFP4: NVFP4 runtime scan skip "
                                f"({name}): {e_pool}"
                            )

                    # --- Hadamard globals: SDXL nvfp4 + ZI zi_nvfp4 + INT8 ---
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        try:
                            nlow = str(name).replace("\\", "/").lower()
                            is_nv_h = (
                                name.endswith("nvfp4_hadamard")
                                or ".nvfp4_hadamard" in name
                                or "zi_nvfp4_hadamard" in nlow
                                or nlow.endswith("nvfp4_hadamard")
                            )
                            is_int8_h = (
                                "native_convert_int8" in nlow
                                or nlow.endswith("native_convert_int8")
                            )
                            if not is_nv_h and not is_int8_h:
                                if not (
                                    "hswq" in nlow
                                    or "nvfp4" in nlow
                                    or "zimage" in nlow
                                ):
                                    continue
                                has_h = (
                                    _safe_hasattr(mod, "_HADAMARD_CACHE")
                                    or _safe_hasattr(mod, "_H4_CACHE")
                                    or _safe_hasattr(
                                        mod, "clear_hadamard_global_caches"
                                    )
                                )
                                if not has_h:
                                    continue
                            fn = _safe_getattr(
                                mod, "clear_hadamard_global_caches", None
                            )
                            if callable(fn):
                                n_h = int(fn() or 0)
                                cleared.append(f"hadamard_api={n_h}@{name}")
                                print(
                                    "HSWQ INT8/NVFP4: Cleared Hadamard global caches "
                                    f"(n={n_h}) via {name}"
                                )
                            n_h2 = 0
                            for attr in ("_HADAMARD_CACHE", "_H4_CACHE"):
                                n_h2 += _clear_dict_attr(mod, attr)
                            if n_h2:
                                tag = "int8" if is_int8_h and not is_nv_h else "nvfp4"
                                cleared.append(
                                    f"hadamard_inplace_{tag}={n_h2}@{name}"
                                )
                                if not callable(fn):
                                    print(
                                        "HSWQ INT8/NVFP4: Cleared Hadamard dicts "
                                        f"in-place (n={n_h2}, {tag}) via {name}"
                                    )
                        except Exception as e_h:
                            print(
                                f"HSWQ INT8/NVFP4: Hadamard global clear "
                                f"failed ({name}): {e_h}"
                            )

                    # --- Loader parity / stats clear (best-effort; never gates local gc) ---
                    for name, mod in list(__import__("sys").modules.items()):
                        if mod is None:
                            continue
                        nlow = str(name).replace("\\", "/").lower()
                        if not (
                            "nvfp4" in nlow
                            or "zi_nvfp4" in nlow
                            or "comfy_quant" in nlow
                            or "hswq" in nlow
                            or "patches" in nlow
                        ):
                            continue
                        for api_name in (
                            "clear_nvfp4_parity_hadamard_caches",
                            "reset_nvfp4_forward_stats",
                            "reset_nvfp4_lora_log_counters",
                            "reset_krea2_nvfp4_lora_bake_log_counters",
                            "reset_int8_lora_log_counters",
                            "clear_nvfp4_runtime_pools",
                            "clear_nvfp4_cudagraphs",
                        ):
                            fn = _safe_getattr(mod, api_name, None)
                            if not callable(fn):
                                continue
                            try:
                                ret = fn()
                                cleared.append(f"{api_name}@{name}={ret!r}")
                                print(
                                    "HSWQ INT8/NVFP4: "
                                    f"{api_name} via {name} -> {ret!r}"
                                )
                            except Exception as e2:
                                print(
                                    f"HSWQ INT8/NVFP4: {api_name} failed "
                                    f"({name}): {e2}"
                                )

                    # --- ALWAYS local gc: every HSWQ module residual (INT8+NVFP4+Detailer models) ---
                    local_dropped = 0
                    try:
                        for obj in gc.get_objects():
                            try:
                                if not isinstance(obj, torch.nn.Module):
                                    continue
                                for attr in _hswq_drop_attrs:
                                    try:
                                        if not hasattr(obj, attr):
                                            continue
                                        _empty_cuda_tensor(getattr(obj, attr, None))
                                        if _drop_attr(obj, attr):
                                            local_dropped += 1
                                    except Exception:
                                        pass
                                # Every ``_hswq_*`` residual (bool arms, sets, tensors, …)
                                try:
                                    for k, v in list(vars(obj).items()):
                                        if not (
                                            isinstance(k, str) and k.startswith("_hswq_")
                                        ):
                                            continue
                                        if k in _hswq_drop_attrs:
                                            continue
                                        if torch.is_tensor(v):
                                            _empty_cuda_tensor(v)
                                        if _drop_attr(obj, k):
                                            local_dropped += 1
                                except Exception:
                                    pass
                            except Exception:
                                continue
                        if local_dropped:
                            cleared.append(f"hswq_module_attrs_gc={local_dropped}")
                            print(
                                "HSWQ INT8/NVFP4: Local gc dropped HSWQ module attrs "
                                f"(n={local_dropped}; INT8+NVFP4 H/bake/arm + all _hswq_*)"
                            )
                    except Exception as e3:
                        print(
                            f"HSWQ INT8/NVFP4: HSWQ module attr gc clear skipped: {e3}"
                        )

                    if cleared:
                        print(
                            "HSWQ INT8/NVFP4: Reset HSWQ/comfy_kitchen caches "
                            + ", ".join(cleared)
                        )
                    else:
                        print("HSWQ INT8/NVFP4: HSWQ/comfy_kitchen caches already empty")

                def _force_unregister_comfy_pins() -> int:
                    """Unregister every cudaHostRegister tracked by ComfyUI PINNED_MEMORY."""
                    nonlocal pins_unregistered
                    freed = 0
                    try:
                        import comfy.model_management as mm
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: cannot import model_management for pin nuke: {e}")
                        return 0
                    pinned = getattr(mm, "PINNED_MEMORY", None)
                    if not isinstance(pinned, dict):
                        print("HSWQ INT8/NVFP4: PINNED_MEMORY dict missing")
                        return 0
                    before = int(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0)
                    print(
                        f"HSWQ INT8/NVFP4: Force-unregister PINNED_MEMORY "
                        f"before={before / (1024 * 1024):.1f} MB entries={len(pinned)}"
                    )
                    for ptr, size in list(pinned.items()):
                        try:
                            if torch.cuda.cudart().cudaHostUnregister(int(ptr)) == 0:
                                pins_unregistered += 1
                                freed += int(size or 0)
                            else:
                                try:
                                    mm.discard_cuda_async_error()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            pinned.pop(ptr, None)
                        except Exception:
                            pass
                    try:
                        mm.TOTAL_PINNED_MEMORY = 0
                    except Exception:
                        pass
                    print(
                        f"HSWQ INT8/NVFP4: Force-unregister done "
                        f"unregistered={pins_unregistered} "
                        f"approx={freed / (1024 * 1024):.1f} MB"
                    )
                    return freed

                def _is_real_nn(obj) -> bool:
                    try:
                        return isinstance(obj, torch.nn.Module)
                    except Exception:
                        return False

                def _unwrap_nn(obj):
                    cur = obj
                    for _ in range(8):
                        if cur is None:
                            return None
                        if _is_real_nn(cur):
                            return cur
                        nxt = getattr(cur, "model", None)
                        if nxt is None or nxt is cur:
                            nxt = getattr(cur, "diffusion_model", None)
                        if nxt is None or nxt is cur:
                            return cur if _is_real_nn(cur) else None
                        cur = nxt
                    return cur if _is_real_nn(cur) else None

                def _is_hswq_int8_nn(module) -> bool:
                    """True for HSWQ INT8 and/or NVFP4 (incl. ZI / Krea2 ConvRot) UNet modules.

                    Pure NVFP4 packs have ``format=nvfp4`` comfy_quant markers and
                    ``_hswq_nvfp4_convrot`` arms — they are not ``int8_tensorwise``.
                    Detecting only INT8 left ZI ConvRot models half-purged: Method 3
                    nuked CUDA tensors while live Modules kept dead
                    ``_hswq_nvfp4_parity_H`` → 2nd gen noise.
                    """
                    if module is None or not _is_real_nn(module):
                        return False
                    baked = getattr(module, "_hswq_int8_baked_keys", None)
                    if baked:
                        return True
                    if getattr(module, "_hswq_int8_baked_uuid", None) is not None:
                        return True
                    # Z Image Dynamic LoRA NVFP4 bake bookkeeping (separate from INT8).
                    if getattr(module, "_hswq_zi_nvfp4_baked_keys", None):
                        return True
                    if getattr(module, "_hswq_zi_nvfp4_baked_uuid", None) is not None:
                        return True
                    # Krea2 ConvRot NVFP4 bake bookkeeping / pack stamp / LoRA residual.
                    if getattr(module, "_hswq_krea2_nvfp4_baked_keys", None):
                        return True
                    if getattr(module, "_hswq_krea2_nvfp4_baked_uuid", None) is not None:
                        return True
                    if getattr(module, "_hswq_krea2_nvfp4_pack", False):
                        return True
                    if getattr(module, "_hswq_krea2_lora_res", None) is not None:
                        return True
                    if getattr(module, "_hswq_krea2_lora_res_gpu", None) is not None:
                        return True
                    try:
                        for m in module.modules():
                            if (
                                getattr(m, "_hswq_nvfp4_convrot", False)
                                or getattr(m, "_hswq_nvfp4", False)
                                or getattr(m, "_hswq_int8_convrot", False)
                                # INT8 Conv2d ConvRot (QuantConv2d) — not Linear protect
                                or getattr(m, "_hswq_convrot", False)
                                or getattr(m, "_hswq_nvfp4_parity_H", None) is not None
                                or getattr(m, "_hswq_nvfp4_H", None) is not None
                                or getattr(m, "_hswq_nvfp4_w_plain", None) is not None
                                or getattr(m, "_hswq_zi_nvfp4_baked_keys", None)
                                or getattr(m, "_hswq_zi_nvfp4_baked_uuid", None) is not None
                                or getattr(m, "_hswq_krea2_nvfp4_baked_keys", None)
                                or getattr(m, "_hswq_krea2_nvfp4_baked_uuid", None) is not None
                                or getattr(m, "_hswq_krea2_nvfp4_pack", False)
                                or getattr(m, "_hswq_krea2_lora_res", None) is not None
                                or getattr(m, "_hswq_krea2_lora_res_gpu", None) is not None
                            ):
                                return True
                            # Any residual ``_hswq_*`` (INT8 / NVFP4 / bake / TC caches)
                            try:
                                for k in vars(m):
                                    if isinstance(k, str) and k.startswith("_hswq_"):
                                        return True
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        for name, buf in module.named_buffers():
                            if not (name.endswith("comfy_quant") or name.endswith(".comfy_quant")):
                                continue
                            try:
                                raw = buf.detach().cpu()
                                if raw.dtype == torch.uint8 and raw.numel() > 0:
                                    import json
                                    conf = json.loads(bytes(raw.tolist()).decode("utf-8", errors="ignore"))
                                    if isinstance(conf, dict):
                                        fmt = conf.get("format")
                                        if fmt in ("int8_tensorwise", "nvfp4"):
                                            return True
                                        if "format" in conf:
                                            # Known non-HSWQ format → keep scanning
                                            continue
                            except Exception:
                                pass
                            # Unparseable comfy_quant on an HSWQ pack → treat as hit
                            return True
                    except Exception:
                        pass
                    return False

                def _loaded_holds_hswq_int8(loaded_model) -> bool:
                    if loaded_model is None:
                        return False
                    for attr in ("model", "real_model"):
                        try:
                            v = getattr(loaded_model, attr, None)
                            if callable(v):
                                try:
                                    v = v()
                                except Exception:
                                    v = None
                            nn = _unwrap_nn(v)
                            if _is_hswq_int8_nn(nn):
                                return True
                            inner = getattr(v, "model", None) if v is not None else None
                            if _is_hswq_int8_nn(_unwrap_nn(inner)):
                                return True
                        except Exception:
                            pass
                    return False

                def _kill_tensor_storage(t) -> int:
                    # Same rule as _force_empty_cuda_storage: unpin/CPU-safe only;
                    # empty(0) only for CUDA. CPU wipe broke ZI TE after Ollama purge.
                    if t is None:
                        return 0
                    if isinstance(t, (list, tuple)):
                        freed_nested = 0
                        for item in t:
                            if item is not None:
                                freed_nested += _kill_tensor_storage(item)
                        return freed_nested
                    if isinstance(t, dict):
                        freed_nested = 0
                        for v in t.values():
                            if v is not None:
                                freed_nested += _kill_tensor_storage(v)
                        return freed_nested
                    freed = 0
                    try:
                        data = getattr(t, "data", t)
                        if data is None:
                            return 0
                        nbytes = int(getattr(data, "nbytes", 0) or 0)
                        is_cuda = False
                        try:
                            is_cuda = bool(getattr(data, "is_cuda", False))
                            if not is_cuda:
                                dev = getattr(data, "device", None)
                                is_cuda = getattr(dev, "type", None) == "cuda"
                        except Exception:
                            pass
                        try:
                            if bool(getattr(data, "is_pinned", lambda: False)()):
                                try:
                                    import comfy.model_management as mm
                                    mm.unpin_memory(data)
                                except Exception:
                                    try:
                                        torch.cuda.cudart().cudaHostUnregister(data.data_ptr())
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        if not is_cuda:
                            return 0
                        dtype = getattr(data, "dtype", torch.float32)
                        empty = torch.empty(0, dtype=dtype, device="cpu")
                        if hasattr(t, "data"):
                            t.data = empty
                        freed = nbytes
                    except Exception:
                        pass
                    return freed

                def _kill_module_vram(module, label: str) -> int:
                    freed = 0
                    print(f"HSWQ INT8/NVFP4: Killing module VRAM ({label}) type={type(module).__name__}")
                    try:
                        if hasattr(module, "to") and callable(module.to):
                            try:
                                module.to("cpu")
                            except Exception as e:
                                print(f"HSWQ INT8/NVFP4: .to('cpu') warning ({label}): {e}")
                    except Exception:
                        pass
                    try:
                        for _n, p in list(module.named_parameters()):
                            freed += _kill_tensor_storage(p)
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: param kill warning ({label}): {e}")
                    try:
                        for _n, b in list(module.named_buffers()):
                            freed += _kill_tensor_storage(b)
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: buffer kill warning ({label}): {e}")
                    # Drop ALL HSWQ module residuals (cache + bake + arm), every submodule.
                    # Loader presence does not matter — next load re-arms ConvRot/TC.
                    # INT8 (Linear + Conv2d) + NVFP4 / ZI + TC forward caches
                    _hswq_kill_attrs = (
                        "_hswq_nvfp4_parity_H",
                        "_hswq_nvfp4_H",
                        "_hswq_nvfp4_w_plain",
                        "_hswq_nvfp4_alpha",
                        "_hswq_nvfp4_no_cudagraph",
                        "_hswq_int8_baked_keys",
                        "_hswq_int8_baked_uuid",
                        "_hswq_zi_nvfp4_baked_keys",
                        "_hswq_zi_nvfp4_baked_uuid",
                        "_hswq_nvfp4_convrot",
                        "_hswq_nvfp4_convrot_groupsize",
                        "_hswq_nvfp4",
                        "_hswq_int8_convrot",
                        "_hswq_int8_convrot_groupsize",
                        "_hswq_convrot",
                        "_hswq_convrot_groupsize",
                        # Krea2 ConvRot NVFP4 (bake bookkeeping / pack stamps / LoRA residuals)
                        "_hswq_krea2_nvfp4_pack",
                        "_hswq_krea2_nvfp4_baked_keys",
                        "_hswq_krea2_nvfp4_baked_uuid",
                        "_hswq_krea2_lora_res",
                        "_hswq_krea2_lora_res_gpu",
                        "_hswq_krea2_tc",
                        "_hswq_krea2_stack",
                        "_hswq_krea2_full_load",
                        "_hswq_krea2_oldquants",
                        "_hswq_krea2_prev_oldquants",
                        "_hswq_krea2_txtlayers_fix",
                        "_hswq_krea2_prev_dynamic_load",
                    )
                    try:
                        for m in module.modules():
                            for attr in _hswq_kill_attrs:
                                if not hasattr(m, attr):
                                    continue
                                try:
                                    val = getattr(m, attr, None)
                                    if val is not None:
                                        freed += _kill_tensor_storage(val)
                                except Exception:
                                    pass
                                try:
                                    delattr(m, attr)
                                except Exception:
                                    try:
                                        setattr(m, attr, None)
                                    except Exception:
                                        pass
                            # Every residual ``_hswq_*`` (bool arms, sets, tensors, …)
                            try:
                                for k, v in list(vars(m).items()):
                                    if not (isinstance(k, str) and k.startswith("_hswq_")):
                                        continue
                                    if k in _hswq_kill_attrs:
                                        continue
                                    if torch.is_tensor(v):
                                        freed += _kill_tensor_storage(v)
                                    try:
                                        delattr(m, k)
                                    except Exception:
                                        try:
                                            setattr(m, k, None)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: HSWQ attr kill warning ({label}): {e}")
                    print(f"HSWQ INT8/NVFP4: Killed ~{freed / (1024 * 1024):.1f} MB CUDA storage ({label})")
                    return freed

                def _unload_patcher(obj) -> int:
                    nonlocal patchers_unloaded
                    freed = 0
                    try:
                        if hasattr(obj, "partially_unload_ram") and callable(obj.partially_unload_ram):
                            try:
                                freed += int(obj.partially_unload_ram(1e30) or 0)
                            except Exception:
                                pass
                        if hasattr(obj, "unregister_inactive_pins") and callable(obj.unregister_inactive_pins):
                            try:
                                freed += int(obj.unregister_inactive_pins(1e30) or 0)
                            except Exception:
                                pass
                        if hasattr(obj, "partially_unload") and callable(obj.partially_unload):
                            try:
                                obj.partially_unload(None, 1e30)
                            except Exception:
                                try:
                                    obj.partially_unload(torch.device("cpu"), 1e30)
                                except Exception:
                                    pass
                        if hasattr(obj, "unpatch_model") and callable(obj.unpatch_model):
                            try:
                                obj.unpatch_model(torch.device("cpu"), unpatch_weights=True)
                            except Exception:
                                pass
                        if hasattr(obj, "model_unload") and callable(obj.model_unload):
                            try:
                                obj.model_unload()
                            except Exception:
                                pass
                        patchers_unloaded += 1
                    except Exception:
                        pass
                    return freed

                _mem_diag("before")

                # 0) Batched Detailer pin pool
                print("HSWQ INT8/NVFP4: Method 0 - Draining HSWQ Batched Detailer PinCache...")
                bytes_killed += _drain_hswq_pin_cache()
                print("HSWQ INT8/NVFP4: Method 0s - Detailer SEGS / PromptExecutor cache...")
                bytes_killed += _purge_detailer_segs_and_executor_cache()

                # 1) ComfyUI loaded models (INT8 first, then unload everything)
                print("HSWQ INT8/NVFP4: Method 1 - current_loaded_models...")
                models_checked_mm = 0
                models_found_mm = 0
                try:
                    import comfy.model_management as mm
                    if hasattr(mm, "current_loaded_models"):
                        current_loaded_models = mm.current_loaded_models
                        print(f"HSWQ INT8/NVFP4: current_loaded_models count={len(current_loaded_models)}")
                        for i in range(len(current_loaded_models) - 1, -1, -1):
                            loaded_model = current_loaded_models[i]
                            models_checked_mm += 1
                            try:
                                is_int8 = _loaded_holds_hswq_int8(loaded_model)
                                if is_int8:
                                    models_found_mm += 1
                                    print(
                                        f"HSWQ INT8/NVFP4: Found HSWQ INT8/NVFP4 at current_loaded_models[{i}] "
                                        f"type={type(loaded_model).__name__}"
                                    )
                                    try:
                                        loaded_model.currently_used = False
                                    except Exception:
                                        pass
                                    nn = None
                                    try:
                                        nn = _unwrap_nn(getattr(loaded_model, "model", None))
                                    except Exception:
                                        nn = None
                                    if nn is not None:
                                        bytes_killed += _kill_module_vram(nn, f"current_loaded_models[{i}]")
                                    hswq_cleared += 1
                                # Always tear down pin/hostbuf on every loaded model
                                try:
                                    inner = getattr(loaded_model, "model", None)
                                    if inner is not None:
                                        bytes_killed += _unload_patcher(inner)
                                except Exception:
                                    pass
                                try:
                                    if hasattr(loaded_model, "model_unload") and callable(loaded_model.model_unload):
                                        loaded_model.model_unload()
                                except Exception as e:
                                    print(f"HSWQ INT8/NVFP4: model_unload warning: {e}")
                                current_loaded_models.pop(i)
                                print(f"HSWQ INT8/NVFP4: Removed current_loaded_models[{i}] (int8={is_int8})")
                            except Exception as e:
                                print(f"HSWQ INT8/NVFP4: Error at current_loaded_models[{i}]: {e}")
                    try:
                        if hasattr(mm, "unload_all_models") and callable(mm.unload_all_models):
                            mm.unload_all_models()
                            print("HSWQ INT8/NVFP4: unload_all_models() issued")
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: unload_all_models warning: {e}")
                    try:
                        if torch.cuda.is_available() and hasattr(mm, "free_memory"):
                            for di in range(torch.cuda.device_count()):
                                mm.free_memory(1e30, torch.device(f"cuda:{di}"))
                            print("HSWQ INT8/NVFP4: free_memory(1e30) issued for all CUDA devices")
                    except Exception as e:
                        print(f"HSWQ INT8/NVFP4: free_memory warning: {e}")
                    if hasattr(mm, "cleanup_models_gc") and callable(mm.cleanup_models_gc):
                        try:
                            mm.cleanup_models_gc()
                        except Exception as e:
                            print(f"HSWQ INT8/NVFP4: cleanup_models_gc warning: {e}")
                except Exception as e:
                    print(f"HSWQ INT8/NVFP4: Error in Method 1: {e}")
                    import traceback
                    print(f"HSWQ INT8/NVFP4: Traceback: {traceback.format_exc()}")
                print(
                    f"HSWQ INT8/NVFP4: Method 1 complete - checked {models_checked_mm}, found {models_found_mm}"
                )
                _mem_diag("after_method1")

                # 2) Force HostUnregister every ComfyUI-tracked pin (NOT sys.modules dir/getattr —
                #    that triggers kornia LazyLoader basicsr install prompts)
                print("HSWQ INT8/NVFP4: Method 2 - Force HostUnregister PINNED_MEMORY...")
                bytes_killed += _force_unregister_comfy_pins()
                _mem_diag("after_method2")

                # 3) gc nuclear: INT8 modules + ModelPatchers + pinned/CUDA tensors
                print("HSWQ INT8/NVFP4: Method 3 - gc nuclear (no sys.modules getattr)...")
                objects_checked = 0
                models_found_in_gc = 0
                try:
                    import comfy.model_management as mm
                except Exception:
                    mm = None
                try:
                    for obj in gc.get_objects():
                        objects_checked += 1
                        if objects_checked > 500000:
                            print("HSWQ INT8/NVFP4: gc scan limit 500000")
                            break
                        try:
                            tname = type(obj).__name__
                            if tname in (
                                "ModelPatcher",
                                "ModelPatcherDynamic",
                                "LoadedModel",
                            ):
                                bytes_killed += _unload_patcher(obj)
                                continue
                            if _is_real_nn(obj) and _is_hswq_int8_nn(obj):
                                models_found_in_gc += 1
                                hswq_cleared += 1
                                print(f"HSWQ INT8/NVFP4: Found HSWQ INT8/NVFP4 in gc type={tname}")
                                bytes_killed += _kill_module_vram(obj, f"gc:{tname}")
                                continue
                            if torch.is_tensor(obj):
                                nbytes = int(getattr(obj, "nbytes", 0) or 0)
                                if nbytes < 1024 * 1024:
                                    continue
                                try:
                                    if bool(obj.is_pinned()):
                                        try:
                                            if mm is not None:
                                                mm.unpin_memory(obj)
                                        except Exception:
                                            try:
                                                torch.cuda.cudart().cudaHostUnregister(obj.data_ptr())
                                            except Exception:
                                                pass
                                        pins_unregistered += 1
                                        bytes_killed += nbytes
                                except Exception:
                                    pass
                                try:
                                    if bool(obj.is_cuda):
                                        cuda_tensors_killed += 1
                                        bytes_killed += _kill_tensor_storage(obj)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception as e:
                    print(f"HSWQ INT8/NVFP4: Error in Method 3: {e}")
                    import traceback
                    print(f"HSWQ INT8/NVFP4: Traceback: {traceback.format_exc()}")
                print(
                    f"HSWQ INT8/NVFP4: Method 3 complete - checked {objects_checked}, "
                    f"int8={models_found_in_gc}, patchers={patchers_unloaded}, "
                    f"cuda_tensors={cuda_tensors_killed}"
                )

                # Second PinCache drain + second PINNED_MEMORY sweep
                print("HSWQ INT8/NVFP4: Method 0b - Second PinCache drain...")
                bytes_killed += _drain_hswq_pin_cache()
                print("HSWQ INT8/NVFP4: Method 0s2 - Second Detailer SEGS / executor sweep...")
                bytes_killed += _purge_detailer_segs_and_executor_cache()
                print("HSWQ INT8/NVFP4: Method 2b - Second PINNED_MEMORY sweep...")
                bytes_killed += _force_unregister_comfy_pins()

                # Nuclear CUDA tensor kill may have destroyed kitchen workspaces
                # while leaving dead refs in module-level dicts — clear them.
                print("HSWQ INT8/NVFP4: Method 2c - Reset comfy_kitchen CUDA caches...")
                _reset_comfy_kitchen_cuda_caches()

                # Reset INT8 LoRA counters (dict-only, no dir())
                print("HSWQ INT8/NVFP4: Resetting comfy_quant_int8 counters...")
                try:
                    for mod_name, mod in _sys_modules():
                        if mod is None or "comfy_quant_int8" not in str(mod_name):
                            continue
                        d = getattr(mod, "__dict__", None)
                        if not isinstance(d, dict):
                            continue
                        reset_fn = d.get("reset_int8_lora_log_counters")
                        if callable(reset_fn):
                            print(f"HSWQ INT8/NVFP4: Calling reset_int8_lora_log_counters via {mod_name}")
                            reset_fn()
                            break
                except Exception as e:
                    print(f"HSWQ INT8/NVFP4: counter reset skipped: {e}")

                print("HSWQ INT8/NVFP4: Running garbage collection...")
                gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    print("HSWQ INT8/NVFP4: Clearing CUDA cache...")
                    for device_idx in range(torch.cuda.device_count()):
                        with torch.cuda.device(device_idx):
                            torch.cuda.empty_cache()
                            try:
                                torch.cuda.ipc_collect()
                            except Exception:
                                pass
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    print("HSWQ INT8/NVFP4: CUDA cache cleared for all devices")
                else:
                    print("HSWQ INT8/NVFP4: CUDA not available, skipped CUDA cache clear")

                try:
                    import comfy.model_management as mm
                    if hasattr(mm, "soft_empty_cache") and callable(mm.soft_empty_cache):
                        try:
                            mm.soft_empty_cache(True)
                        except TypeError:
                            mm.soft_empty_cache()
                except Exception:
                    pass

                _mem_diag("after")
                print(
                    f"HSWQ INT8/NVFP4: Done — cleared {hswq_cleared} HSWQ INT8/NVFP4 ref(s), "
                    f"pins_unregistered={pins_unregistered}, "
                    f"patchers={patchers_unloaded}, "
                    f"cuda_tensors={cuda_tensors_killed}, "
                    f"approx {bytes_killed / (1024 * 1024):.1f} MB tracked"
                )

            except Exception as e:
                print(f"HSWQ INT8/NVFP4: Error purging models: {e}")
                import traceback
                print(f"HSWQ INT8/NVFP4: Traceback: {traceback.format_exc()}")

        # Purge Ollama VRAM for comfyui-ollama + comfyui-ollama-describer.
        # Both packs talk to the external Ollama server (no in-process torch models).
        # describer defaults keep_model_alive=-1 (indefinite load) — must empty /api/ps.
        if purge_ollama:
            try:
                print("Ollama: Starting FULL purge (zero residual; includes comfyui-ollama-describer)...")
                import json
                import shutil
                import subprocess
                import time
                import urllib.error
                import urllib.request

                def _is_comfyui_ollama_module(mod_name: str) -> bool:
                    n = str(mod_name).replace("\\", "/").lower()
                    return (
                        "compfyuiollama" in n
                        or "comfyui-ollama" in n
                        or "comfyui_ollama" in n
                        or "ollama-describer" in n
                        or "ollama_describer" in n
                        or (n.endswith("deprecated_nodes") and "ollama" in n)
                    )

                def _normalize_ollama_url(host: str) -> str:
                    host = str(host).strip().rstrip("/")
                    if not host:
                        return ""
                    if not host.startswith("http"):
                        host = "http://" + host
                    return host.rstrip("/")

                def _ollama_urls() -> list:
                    urls = []
                    env_host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL")
                    if env_host:
                        u = _normalize_ollama_url(env_host)
                        if u:
                            urls.append(u)
                    # Harvest URLs from loaded ollama custom-node modules / node state
                    for mod_name, mod in list(sys.modules.items()):
                        if mod is None or not _is_comfyui_ollama_module(mod_name):
                            continue
                        print(f"Ollama: scanning module {mod_name}")
                        for attr in ("DEFAULT_URL", "OLLAMA_URL", "url", "base_url", "api_host"):
                            v = getattr(mod, attr, None)
                            if isinstance(v, str) and ("http://" in v or "https://" in v or "11434" in v):
                                u = _normalize_ollama_url(v)
                                if u:
                                    urls.append(u)
                        # comfyui-ollama-describer: class INPUT_TYPES defaults for api_host
                        for cls_name in (
                            "OllamaImageCaptioner",
                            "OllamaImageDescriber",
                            "OllamaTextDescriber",
                            "OllamaConnectivityV2",
                            "OllamaGenerateV2",
                        ):
                            cls = getattr(mod, cls_name, None)
                            if cls is None or not hasattr(cls, "INPUT_TYPES"):
                                continue
                            try:
                                spec = cls.INPUT_TYPES()
                                required = (spec or {}).get("required") or {}
                                for key in ("api_host", "url"):
                                    entry = required.get(key)
                                    if (
                                        isinstance(entry, tuple)
                                        and len(entry) >= 2
                                        and isinstance(entry[1], dict)
                                    ):
                                        default = entry[1].get("default")
                                        if isinstance(default, str) and default.strip():
                                            u = _normalize_ollama_url(default)
                                            if u:
                                                urls.append(u)
                            except Exception:
                                pass
                    try:
                        for obj in gc.get_objects():
                            try:
                                if not hasattr(obj, "__class__"):
                                    continue
                                cname = type(obj).__name__
                                # Client from ollama package (used by both packs)
                                if cname == "Client" and hasattr(obj, "host"):
                                    v = getattr(obj, "host", None)
                                    if isinstance(v, str) and v.strip():
                                        u = _normalize_ollama_url(v)
                                        if u:
                                            urls.append(u)
                                if "Ollama" not in cname and "ollama" not in cname.lower():
                                    if not (cname == "OllamaUtil"):
                                        continue
                                # describer uses api_host; comfyui-ollama uses url
                                for attr in ("url", "_url", "host", "base_url", "api_host"):
                                    v = getattr(obj, attr, None)
                                    if isinstance(v, str) and v.strip():
                                        u = _normalize_ollama_url(v)
                                        if u:
                                            urls.append(u)
                                # Nested client.host on OllamaUtil
                                client = getattr(obj, "client", None)
                                if client is not None:
                                    for attr in ("host", "base_url", "url"):
                                        v = getattr(client, attr, None)
                                        if isinstance(v, str) and v.strip():
                                            u = _normalize_ollama_url(v)
                                            if u:
                                                urls.append(u)
                                # OllamaConnectivityV2 returns dict via execute; instances may stash meta
                                for attr in ("saved_meta", "meta", "connectivity"):
                                    bag = getattr(obj, attr, None)
                                    if isinstance(bag, dict):
                                        u = bag.get("url") or bag.get("api_host")
                                        if not u and isinstance(bag.get("connectivity"), dict):
                                            u = bag["connectivity"].get("url") or bag["connectivity"].get(
                                                "api_host"
                                            )
                                        if isinstance(u, str) and u.strip():
                                            nu = _normalize_ollama_url(u)
                                            if nu:
                                                urls.append(nu)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Ollama: URL harvest via gc skipped: {e}")
                    urls.append("http://127.0.0.1:11434")
                    urls.append("http://localhost:11434")
                    seen = set()
                    out = []
                    for u in urls:
                        u = str(u).rstrip("/")
                        if u and u not in seen:
                            seen.add(u)
                            out.append(u)
                    return out

                def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 30.0):
                    data = None
                    headers = {"Accept": "application/json"}
                    if body is not None:
                        data = json.dumps(body).encode("utf-8")
                        headers["Content-Type"] = "application/json"
                    req = urllib.request.Request(url, data=data, headers=headers, method=method)
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        raw = resp.read()
                        if not raw:
                            return None
                        return json.loads(raw.decode("utf-8", errors="ignore"))

                def _model_names_from_ps(payload) -> list:
                    names = []
                    if payload is None:
                        return names
                    if isinstance(payload, dict):
                        models = payload.get("models")
                    elif isinstance(payload, list):
                        models = payload
                    else:
                        models = getattr(payload, "models", None)
                    for m in models or []:
                        name = None
                        if isinstance(m, dict):
                            name = m.get("model") or m.get("name")
                        else:
                            name = getattr(m, "model", None) or getattr(m, "name", None)
                        if name:
                            names.append(str(name))
                    # unique, stable
                    seen = set()
                    out = []
                    for n in names:
                        if n not in seen:
                            seen.add(n)
                            out.append(n)
                    return out

                def _ps_names(base_url: str) -> list:
                    ps_url = base_url.rstrip("/") + "/api/ps"
                    try:
                        return _model_names_from_ps(_http_json("GET", ps_url, None, timeout=10.0))
                    except Exception as e:
                        print(f"Ollama: GET {ps_url} failed: {e}")
                        return []

                def _unload_one_http(base_url: str, name: str) -> None:
                    root = base_url.rstrip("/")
                    # Official unload: empty prompt/messages + keep_alive 0 (int and "0"), stream false
                    bodies_gen = (
                        {"model": name, "prompt": "", "keep_alive": 0, "stream": False},
                        {"model": name, "keep_alive": 0, "stream": False},
                        {"model": name, "prompt": "", "keep_alive": "0", "stream": False},
                    )
                    for body in bodies_gen:
                        try:
                            _http_json("POST", root + "/api/generate", body, timeout=120.0)
                            print(f"Ollama: /api/generate keep_alive=0 -> {name}")
                            break
                        except Exception as e:
                            print(f"Ollama: /api/generate unload try failed for {name}: {e}")
                    bodies_chat = (
                        {"model": name, "messages": [], "keep_alive": 0, "stream": False},
                        {"model": name, "messages": [], "keep_alive": "0", "stream": False},
                    )
                    for body in bodies_chat:
                        try:
                            _http_json("POST", root + "/api/chat", body, timeout=120.0)
                            print(f"Ollama: /api/chat keep_alive=0 -> {name}")
                            break
                        except Exception as e:
                            print(f"Ollama: /api/chat unload try failed for {name}: {e}")

                def _unload_one_client(client, name: str) -> None:
                    for kwargs in (
                        {"model": name, "prompt": "", "keep_alive": 0},
                        {"model": name, "prompt": "", "keep_alive": "0"},
                    ):
                        try:
                            client.generate(**kwargs)
                            print(f"Ollama: Client.generate keep_alive=0 -> {name}")
                            break
                        except Exception as e:
                            print(f"Ollama: Client.generate failed for {name}: {e}")
                    if hasattr(client, "chat") and callable(client.chat):
                        for kwargs in (
                            {"model": name, "messages": [], "keep_alive": 0},
                            {"model": name, "messages": [], "keep_alive": "0"},
                        ):
                            try:
                                client.chat(**kwargs)
                                print(f"Ollama: Client.chat keep_alive=0 -> {name}")
                                break
                            except Exception as e:
                                print(f"Ollama: Client.chat failed for {name}: {e}")

                def _ollama_stop_cli(name: str) -> None:
                    try:
                        r = subprocess.run(
                            ["ollama", "stop", name],
                            capture_output=True,
                            text=True,
                            timeout=60,
                            shell=False,
                        )
                        print(
                            f"Ollama: CLI stop {name} rc={r.returncode} "
                            f"stdout={(r.stdout or '').strip()[:200]} "
                            f"stderr={(r.stderr or '').strip()[:200]}"
                        )
                    except FileNotFoundError:
                        print("Ollama: CLI `ollama` not on PATH (skipped)")
                    except Exception as e:
                        print(f"Ollama: CLI stop failed for {name}: {e}")

                def _purge_server_until_empty(base_url: str, max_rounds: int = 8) -> tuple:
                    """Return (unload_attempts, residual_names). residual must be [] for success."""
                    attempts = 0
                    client = None
                    try:
                        from ollama import Client

                        client = Client(host=base_url)
                    except Exception as e:
                        print(f"Ollama: Client optional ({base_url}): {e}")

                    residual = _ps_names(base_url)
                    print(f"Ollama: {base_url} initial loaded={residual if residual else '(none)'}")
                    for round_i in range(1, max_rounds + 1):
                        residual = _ps_names(base_url)
                        if not residual and client is not None and hasattr(client, "ps"):
                            try:
                                residual = _model_names_from_ps(client.ps())
                            except Exception:
                                pass
                        if not residual:
                            print(f"Ollama: {base_url} EMPTY after round {round_i - 1} (verified /api/ps)")
                            return attempts, []
                        print(f"Ollama: {base_url} round {round_i}/{max_rounds} still loaded={residual}")
                        for name in list(residual):
                            attempts += 1
                            _unload_one_http(base_url, name)
                            if client is not None:
                                _unload_one_client(client, name)
                            _ollama_stop_cli(name)
                        time.sleep(0.35)
                    residual = _ps_names(base_url)
                    if residual:
                        print(f"Ollama: WARNING residual still loaded on {base_url}: {residual}")
                    return attempts, residual

                total_unloaded = 0
                residual_all = []
                for base in _ollama_urls():
                    print(f"Ollama: Purging server at {base}...")
                    n, residual = _purge_server_until_empty(base)
                    total_unloaded += n
                    for r in residual:
                        if r not in residual_all:
                            residual_all.append(r)

                # --- Comfy-side comfyui-ollama state: leave nothing ---
                sessions_cleared = 0
                context_cleared = 0
                files_wiped = 0
                instances_cleared = 0

                for mod_name, mod in list(sys.modules.items()):
                    if mod is None or not _is_comfyui_ollama_module(mod_name):
                        continue
                    try:
                        bag = getattr(mod, "CHAT_SESSIONS", None)
                        if isinstance(bag, dict):
                            for _sid, sess in list(bag.items()):
                                try:
                                    msgs = getattr(sess, "messages", None)
                                    if isinstance(msgs, list):
                                        msgs.clear()
                                    if hasattr(sess, "model"):
                                        setattr(sess, "model", "")
                                except Exception:
                                    pass
                            sessions_cleared += len(bag)
                            bag.clear()
                            print(f"Ollama: Cleared CHAT_SESSIONS via {mod_name} entries={sessions_cleared}")
                    except Exception as e:
                        print(f"Ollama: CHAT_SESSIONS clear failed ({mod_name}): {e}")

                    for attr in (
                        "OllamaGenerateAdvance",
                        "OllamaGenerate",
                        "OllamaVision",
                        "OllamaGenerateV2",
                        "OllamaChat",
                        "OllamaSaveContext",
                        "OllamaLoadContext",
                    ):
                        try:
                            cls = getattr(mod, attr, None)
                            if cls is None:
                                continue
                            if hasattr(cls, "saved_context"):
                                setattr(cls, "saved_context", None)
                                context_cleared += 1
                        except Exception as e:
                            print(f"Ollama: class {attr} clear failed ({mod_name}): {e}")

                    # Wipe on-disk saved_context artifacts under the custom node (keep .keep only)
                    try:
                        mod_file = getattr(mod, "__file__", None)
                        if mod_file:
                            base_dir = os.path.dirname(os.path.realpath(mod_file))
                            ctx_dir = os.path.join(base_dir, "saved_context")
                            if os.path.isdir(ctx_dir):
                                for fn in os.listdir(ctx_dir):
                                    if fn == ".keep":
                                        continue
                                    path = os.path.join(ctx_dir, fn)
                                    try:
                                        if os.path.isfile(path) or os.path.islink(path):
                                            os.remove(path)
                                            files_wiped += 1
                                        elif os.path.isdir(path):
                                            shutil.rmtree(path)
                                            files_wiped += 1
                                    except Exception as e:
                                        print(f"Ollama: wipe file failed {path}: {e}")
                                print(f"Ollama: wiped saved_context dir files via {mod_name} count={files_wiped}")
                    except Exception as e:
                        print(f"Ollama: saved_context dir wipe failed ({mod_name}): {e}")

                # Instance-level saved_context / message buffers on live Ollama node objects
                try:
                    for obj in gc.get_objects():
                        try:
                            if not hasattr(obj, "__class__"):
                                continue
                            cname = type(obj).__name__
                            if "Ollama" not in cname:
                                continue
                            changed = False
                            if hasattr(obj, "saved_context"):
                                setattr(obj, "saved_context", None)
                                changed = True
                            for attr in ("messages", "context", "history", "chat_history"):
                                if hasattr(obj, attr):
                                    val = getattr(obj, attr)
                                    if isinstance(val, list):
                                        val.clear()
                                        changed = True
                                    elif val is not None and attr != "messages":
                                        try:
                                            setattr(obj, attr, None)
                                            changed = True
                                        except Exception:
                                            pass
                            if changed:
                                instances_cleared += 1
                        except Exception:
                            pass
                    if instances_cleared:
                        print(f"Ollama: Cleared instance state on {instances_cleared} Ollama objects")
                except Exception as e:
                    print(f"Ollama: gc instance sweep failed: {e}")

                try:
                    gc.collect()
                except Exception:
                    pass

                # Final verify every known URL
                for base in _ollama_urls():
                    left = _ps_names(base)
                    if left:
                        print(f"Ollama: FINAL VERIFY FAIL {base} still loaded={left}")
                        for x in left:
                            if x not in residual_all:
                                residual_all.append(x)
                    else:
                        print(f"Ollama: FINAL VERIFY OK {base} loaded=(none)")

                print(
                    f"Ollama: Done — unload_attempts={total_unloaded}, "
                    f"sessions_cleared={sessions_cleared}, "
                    f"context_attrs_cleared={context_cleared}, "
                    f"files_wiped={files_wiped}, "
                    f"instances_cleared={instances_cleared}, "
                    f"residual={residual_all if residual_all else '(none)'}"
                )
            except Exception as e:
                print(f"Ollama: Error purging: {e}")
                import traceback
                print(f"Ollama: Traceback: {traceback.format_exc()}")

        return (anything,)

