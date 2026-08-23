import sys
import os
import torch
import gc
import logging

__version__ = "2.4.5"

# Ensure ComfyUI root is on sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

_SAGE160_LOGGED = set()


def _install_sage_attention_noise_guard():
    """
    External runtime patch:
    suppress repeated unsupported head_dim=160 sage-attention errors
    without editing ComfyUI core files.
    """
    global _SAGE160_LOGGED

    try:
        from comfy.ldm.modules import attention as comfy_attention
    except Exception as e:
        print(f"[ComfyUI-VRAM-Manager] WARNING: failed to import attention module for patch: {e}")
        return

    original_wrapped = getattr(comfy_attention, "attention_sage", None)
    if original_wrapped is None:
        print("[ComfyUI-VRAM-Manager] WARNING: attention_sage not found; skip patch")
        return

    if getattr(original_wrapped, "_dm_sage160_guard", False):
        return

    @comfy_attention.wrap_attn
    def attention_sage_guarded(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return comfy_attention.attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

        exception_fallback = False
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head),
                (q, k, v),
            )
            tensor_layout = "NHD"

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        try:
            out = comfy_attention.sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
        except Exception as e:
            err = str(e)
            if "Unsupported head_dim: 160" in err:
                if "unsupported_head_dim_160" not in _SAGE160_LOGGED:
                    logging.info("Sage attention unsupported for head_dim=160; using pytorch attention fallback.")
                    _SAGE160_LOGGED.add("unsupported_head_dim_160")
            else:
                logging.error("Error running sage attention: {}, using pytorch attention instead.".format(e))
            exception_fallback = True

        if exception_fallback:
            if tensor_layout == "NHD":
                q, k, v = map(
                    lambda t: t.transpose(1, 2),
                    (q, k, v),
                )
            return comfy_attention.attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)

        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = (
                    out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                )
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out

    attention_sage_guarded._dm_sage160_guard = True
    comfy_attention.attention_sage = attention_sage_guarded

    if getattr(comfy_attention, "optimized_attention", None) is original_wrapped:
        comfy_attention.optimized_attention = attention_sage_guarded
    if getattr(comfy_attention, "optimized_attention_masked", None) is original_wrapped:
        comfy_attention.optimized_attention_masked = attention_sage_guarded
    if hasattr(comfy_attention, "REGISTERED_ATTENTION_FUNCTIONS"):
        if comfy_attention.REGISTERED_ATTENTION_FUNCTIONS.get("sage") is original_wrapped:
            comfy_attention.REGISTERED_ATTENTION_FUNCTIONS["sage"] = attention_sage_guarded

    print("[ComfyUI-VRAM-Manager] Installed external sage-attention head_dim=160 noise guard")


_install_sage_attention_noise_guard()


def _ensure_nvidia_ml_py_latest():
    """
    Auto-upgrade nvidia-ml-py (import name: pynvml) to the latest PyPI release.
    Required for General Manage VRAM (v2.4.0); replaces manual `pip install -U nvidia-ml-py`.
    Also invoked from install.py when ComfyUI-Manager installs/updates this node.
    """
    import subprocess

    try:
        print("[ComfyUI-VRAM-Manager] Ensuring nvidia-ml-py is latest...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "nvidia-ml-py"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            print(f"[ComfyUI-VRAM-Manager] WARNING: nvidia-ml-py upgrade failed: {err}")
            return False
        print("[ComfyUI-VRAM-Manager] nvidia-ml-py is up to date (or was upgraded).")
        return True
    except Exception as e:
        print(f"[ComfyUI-VRAM-Manager] WARNING: nvidia-ml-py ensure error: {e}")
        return False


def _install_general_vram_management():
    """
    Startup patch: auto-detect non-PyTorch VRAM usage (browsers, other apps)
    via NVML and apply general manage VRAM patch to ComfyUI memory management at load time.
    """
    _ensure_nvidia_ml_py_latest()

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

# Import Memory Manager nodes (including any for ModelPatchMemoryCleaner)
try:
    from .nodes.memory_manager import MemoryManager, any
    print("[ComfyUI-VRAM-Manager] Successfully imported MemoryManager from .nodes.memory_manager")
except ImportError as e:
    try:
        from nodes.memory_manager import MemoryManager, any
        print("[ComfyUI-VRAM-Manager] Successfully imported MemoryManager from nodes.memory_manager")
    except ImportError as e2:
        print(f"[ComfyUI-VRAM-Manager] WARNING: Failed to import MemoryManager: {e2}")
        MemoryManager = None
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


# Import Purge VRAM V2 node.
# Prefer nodes.purge_vram (Method 2c clears HSWQ NVFP4 runtime pools + ZI
# ConvRot parity Hadamard after Distorch nuclear kill). Root purge_vram.py is
# legacy fallback only.
try:
    from .nodes.purge_vram import DisTorchPurgeVRAMV2
    print("[ComfyUI-VRAM-Manager] Successfully imported DisTorchPurgeVRAMV2 from .nodes.purge_vram")
except ImportError as e:
    try:
        from .purge_vram import DisTorchPurgeVRAMV2
        print("[ComfyUI-VRAM-Manager] Successfully imported DisTorchPurgeVRAMV2 from .purge_vram")
    except ImportError as e2:
        print(f"[ComfyUI-VRAM-Manager] WARNING: Failed to import DisTorchPurgeVRAMV2: {e2}")
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
    from .nodes.sa import PatchSageAttentionDM
    print("[ComfyUI-VRAM-Manager] Successfully imported PatchSageAttentionDM from .nodes.sa")
except ImportError as e:
    try:
        from nodes.sa import PatchSageAttentionDM
        print("[ComfyUI-VRAM-Manager] Successfully imported PatchSageAttentionDM from nodes.sa")
    except ImportError as e2:
        print(f"[ComfyUI-VRAM-Manager] WARNING: Failed to import PatchSageAttentionDM: {e2}")
        PatchSageAttentionDM = None


# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,
}

# Register Memory Manager nodes if available
if MemoryManager is not None:
    NODE_CLASS_MAPPINGS["MemoryManager"] = MemoryManager
    print("[ComfyUI-VRAM-Manager] Registered MemoryManager node")
else:
    print("[ComfyUI-VRAM-Manager] ERROR: MemoryManager is None, not registered")

# Register Purge VRAM V2 node if available
if DisTorchPurgeVRAMV2 is not None:
    NODE_CLASS_MAPPINGS["DisTorchPurgeVRAMV2"] = DisTorchPurgeVRAMV2
    print("[ComfyUI-VRAM-Manager] Registered DisTorchPurgeVRAMV2 node")
else:
    print("[ComfyUI-VRAM-Manager] ERROR: DisTorchPurgeVRAMV2 is None, not registered")

# Register SageAttention node if available
if PatchSageAttentionDM is not None:
    NODE_CLASS_MAPPINGS["PatchSageAttentionDM"] = PatchSageAttentionDM
    print("[ComfyUI-VRAM-Manager] Registered PatchSageAttentionDM node")
else:
    print("[ComfyUI-VRAM-Manager] ERROR: PatchSageAttentionDM is None, not registered")

print(f"[ComfyUI-VRAM-Manager] Total registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPatchMemoryCleaner": "Model Patch Memory Cleaner",
}

# Register Memory Manager node display names if available
if MemoryManager is not None:
    NODE_DISPLAY_NAME_MAPPINGS["MemoryManager"] = "Memory Manager"

# Register Purge VRAM V2 node display name if available
if DisTorchPurgeVRAMV2 is not None:
    NODE_DISPLAY_NAME_MAPPINGS["DisTorchPurgeVRAMV2"] = "General Purge VRAM V2"

# Register SageAttention node display name if available
if PatchSageAttentionDM is not None:
    NODE_DISPLAY_NAME_MAPPINGS["PatchSageAttentionDM"] = "Patch Sage Attention DM"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 
