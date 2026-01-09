"""
SageAttention patch node for DistorchMemoryManager
"""
import torch
import logging
from comfy.ldm.modules import attention as comfy_attention
import comfy.model_management as mm
from comfy.ldm.modules.attention import wrap_attn
from comfy.patcher_extension import CallbacksMP

# SageAttention modes
sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]

def get_sage_func_dm(sage_attention, allow_compile=False):
    # Detect SageAttention version
    try:
        import sageattention
        sage_version = None
        try:
            sage_version = sageattention.__version__
        except AttributeError:
            try:
                import importlib.metadata
                sage_version = importlib.metadata.version("sageattention")
            except Exception:
                sage_version = None
        
        if sage_version and sage_version != "unknown":
            try:
                import torch
                cuda_version = torch.version.cuda or "unknown"
                torch_version = torch.version.__version__ or "unknown"
                logging.info(f"Patching comfy attention to use SageAttention {sage_version}+cu{cuda_version}torch{torch_version}")
            except:
                logging.info(f"Patching comfy attention to use SageAttention {sage_version}")
        else:
            logging.info("Patching comfy attention to use sageattn")
    except:
        logging.info("Patching comfy attention to use sageattn")
    
    from sageattention import sageattn
    if sage_attention == "auto":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    elif "sageattn3" in sage_attention:
        from sageattn3 import sageattn3_blackwell
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out

    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout="HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head),
                (q, k, v),
            )
            tensor_layout="NHD"
        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
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
    return attention_sage


class PatchSageAttentionDM():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."}),
        },
        "optional": {
            "allow_compile": ("BOOLEAN", {"default": False, "tooltip": "Allow the use of torch.compile for the sage attention function, requires latest sageattn 2.2.0 or higher."})
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching attention mode. This doesn't use the model patching system and thus can't be disabled without running the node again with 'disabled' option."
    CATEGORY = "Memory"

    def patch(self, model, sage_attention, allow_compile=False):
        model_clone = model.clone()
        
        @torch.compiler.disable()
        def patch_attention_enable(model):
            if sage_attention != "disabled":
                new_attention = get_sage_func_dm(sage_attention, allow_compile=allow_compile)
                def attention_override_sage(func, *args, **kwargs):
                    return new_attention.__wrapped__(*args, **kwargs)
                
                if "transformer_options" not in model.model_options:
                    model.model_options["transformer_options"] = {}
                model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
            else:
                if "transformer_options" in model.model_options:
                    if "optimized_attention_override" in model.model_options["transformer_options"]:
                        del model.model_options["transformer_options"]["optimized_attention_override"]
                
                flash_attention_enabled = False
                try:
                    if hasattr(mm, 'FLASH_IS_AVAILABLE') and mm.FLASH_IS_AVAILABLE:
                        flash_attention_enabled = True
                    elif hasattr(mm, 'flash_attention_enabled'):
                        flash_attention_enabled = mm.flash_attention_enabled()
                    else:
                        try:
                            current_attn = comfy_attention.optimized_attention
                            attn_name = current_attn.__name__
                            if attn_name == "attention_flash":
                                flash_attention_enabled = True
                        except:
                            pass
                except:
                    pass
                
                if flash_attention_enabled:
                    logging.info("Restoring initial comfy attention")
                    if hasattr(mm, 'FLASH_ATTN_VERSION') and mm.FLASH_ATTN_VERSION and mm.FLASH_ATTN_VERSION != "unknown":
                        try:
                            version_parts = mm.FLASH_ATTN_VERSION.split('.')
                            major_version = int(version_parts[0])
                            if major_version >= 3:
                                flash_attn_type = "FA-3"
                            else:
                                flash_attn_type = "FA-2"
                            logging.info(f"[ComfyUI] Using {flash_attn_type} (Flash-Attention {mm.FLASH_ATTN_VERSION}) direct")
                        except Exception:
                            if hasattr(mm, 'FLASH_ATTN_TYPE') and mm.FLASH_ATTN_TYPE:
                                logging.info(f"[ComfyUI] Using {mm.FLASH_ATTN_TYPE} (Flash-Attention {mm.FLASH_ATTN_VERSION}) direct")
                            else:
                                logging.info(f"[ComfyUI] Using Flash-Attention {mm.FLASH_ATTN_VERSION} direct")
                    else:
                        logging.info("[ComfyUI] Using Flash-Attention direct")
                else:
                    logging.info("Restoring initial comfy attention")
        
        @torch.compiler.disable()
        def patch_attention_disable(model):
            if "transformer_options" in model.model_options:
                if "optimized_attention_override" in model.model_options["transformer_options"]:
                    del model.model_options["transformer_options"]["optimized_attention_override"]
            
            # Output FA log even when SA is enabled, during reset
            flash_attention_enabled = False
            try:
                if hasattr(mm, 'FLASH_IS_AVAILABLE') and mm.FLASH_IS_AVAILABLE:
                    flash_attention_enabled = True
                elif hasattr(mm, 'flash_attention_enabled'):
                    flash_attention_enabled = mm.flash_attention_enabled()
                else:
                    try:
                        current_attn = comfy_attention.optimized_attention
                        attn_name = current_attn.__name__
                        if attn_name == "attention_flash":
                            flash_attention_enabled = True
                    except:
                        pass
            except:
                pass
            
            if flash_attention_enabled:
                logging.info("Restoring initial comfy attention")
                if hasattr(mm, 'FLASH_ATTN_VERSION') and mm.FLASH_ATTN_VERSION and mm.FLASH_ATTN_VERSION != "unknown":
                    try:
                        version_parts = mm.FLASH_ATTN_VERSION.split('.')
                        major_version = int(version_parts[0])
                        if major_version >= 3:
                            flash_attn_type = "FA-3"
                        else:
                            flash_attn_type = "FA-2"
                        logging.info(f"[ComfyUI] Using {flash_attn_type} (Flash-Attention {mm.FLASH_ATTN_VERSION}) direct")
                    except Exception:
                        if hasattr(mm, 'FLASH_ATTN_TYPE') and mm.FLASH_ATTN_TYPE:
                            logging.info(f"[ComfyUI] Using {mm.FLASH_ATTN_TYPE} (Flash-Attention {mm.FLASH_ATTN_VERSION}) direct")
                        else:
                            logging.info(f"[ComfyUI] Using Flash-Attention {mm.FLASH_ATTN_VERSION} direct")
                else:
                    logging.info("[ComfyUI] Using Flash-Attention direct")
            else:
                logging.info("Restoring initial comfy attention")
        
        model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_attention_enable)
        model_clone.add_callback(CallbacksMP.ON_CLEANUP, patch_attention_disable)
        
        return model_clone,
