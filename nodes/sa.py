"""
SageAttention patch node for DistorchMemoryManager
"""
import torch
import logging
from comfy.ldm.modules import attention as comfy_attention
from comfy.ldm.modules.attention import wrap_attn
from comfy.patcher_extension import CallbacksMP
from comfy.cli_args import args

# Import model_management safely (may fail during module import)
try:
    import comfy.model_management as mm
except Exception:
    mm = None

# SageAttention modes
sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]
_logged_sage_fallback_messages = set()


# Flash-Attention version detection (independent from model_management)
def get_flash_attention_info():
    """
    Get Flash-Attention version and type information.
    Returns: (is_available, version, type)
    """
    flash_is_available = False
    flash_attn_version = None
    flash_attn_type = None
    
    # Check if Flash-Attention is available regardless of args.use_flash_attention
    try:
        import flash_attn
        flash_is_available = True
        try:
            flash_attn_version = flash_attn.__version__
            try:
                version_parts = flash_attn_version.split('.')
                major_version = int(version_parts[0])
                if major_version >= 3:
                    flash_attn_type = "FA-3"
                else:
                    flash_attn_type = "FA-2"
            except Exception:
                flash_attn_type = None
        except AttributeError:
            try:
                import importlib.metadata
                flash_attn_version = importlib.metadata.version("flash-attn")
                try:
                    version_parts = flash_attn_version.split('.')
                    major_version = int(version_parts[0])
                    if major_version >= 3:
                        flash_attn_type = "FA-3"
                    else:
                        flash_attn_type = "FA-2"
                except Exception:
                    flash_attn_type = None
            except Exception:
                flash_attn_version = "unknown"
                flash_attn_type = None
    except ImportError:
        flash_is_available = False
    except Exception:
        flash_is_available = False
    
    return flash_is_available, flash_attn_version, flash_attn_type


# SageAttention version detection (independent from model_management)
def get_sage_attention_info():
    """
    Get SageAttention version information.
    Returns: (version, cuda_version, torch_version)
    """
    sage_version = None
    cuda_version = "unknown"
    torch_version = "unknown"
    
    try:
        import sageattention
        try:
            sage_version = sageattention.__version__
        except AttributeError:
            try:
                import importlib.metadata
                sage_version = importlib.metadata.version("sageattention")
            except Exception:
                sage_version = None
        
        try:
            cuda_version = torch.version.cuda or "unknown"
            torch_version = torch.version.__version__ or "unknown"
        except:
            pass
    except:
        pass
    
    return sage_version, cuda_version, torch_version


# SageAttention3 version detection (independent from model_management)
def get_sage_attention3_info():
    """
    Get SageAttention3 version information.
    Returns: (version, is_available, supports_blackwell)
    """
    sage3_version = None
    is_available = False
    supports_blackwell = False
    
    try:
        from sageattn3.blackwell import __version__ as blackwell_version
        sage3_version = blackwell_version
        is_available = True
        supports_blackwell = True
    except ImportError:
        try:
            import sageattn3
            sage3_version = "unknown"
            is_available = True
        except ImportError:
            pass
    
    return sage3_version, is_available, supports_blackwell


# Check if Flash-Attention is enabled (independent from model_management)
def is_flash_attention_enabled():
    """
    Check if Flash-Attention is currently enabled.
    Returns: bool
    """
    # First check if Flash-Attention is actually being used
    try:
        current_attn = comfy_attention.optimized_attention
        attn_name = current_attn.__name__
        if attn_name == "attention_flash":
            return True
    except:
        pass
    
    # Check if Flash-Attention is available and args.use_flash_attention is set
    flash_is_available, _, _ = get_flash_attention_info()
    if flash_is_available and args.use_flash_attention:
        return True
    
    return False

def get_sage_func_dm(sage_attention, allow_compile=False):
    # SA3 uses separate logging, so only log SA2 info for non-SA3 modes
    if "sageattn3" not in sage_attention:
        # Detect SageAttention version using our own function
        sage_version, cuda_version, torch_version = get_sage_attention_info()
        
        if sage_version and sage_version != "unknown":
            if cuda_version != "unknown" and torch_version != "unknown":
                logging.info(f"Patching comfy attention to use SageAttention {sage_version}+cu{cuda_version}torch{torch_version}")
            else:
                logging.info(f"Patching comfy attention to use SageAttention {sage_version}")
        else:
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
        # SA3-specific version detection and logging
        sage3_version, sa3_available, supports_blackwell = get_sage_attention3_info()
        if sage3_version and sage3_version != "unknown":
            logging.info(f"Patching comfy attention to use SageAttention3 {sage3_version} (Blackwell FP4)")
        else:
            logging.info("Patching comfy attention to use SageAttention3 (Blackwell FP4)")
        
        from sageattn3 import sageattn3_blackwell
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            # Convert NHD -> HND layout (SA3 expects HND: [batch, heads, seq_len, dim])
            if tensor_layout == "NHD":
                q_s, k_s, v_s = [x.transpose(1, 2) for x in (q, k, v)]
            else:
                q_s, k_s, v_s = q, k, v
            
            # SA3 constraints check - fallback to SDPA if needed
            # 1. SA3 does not support attention mask
            # 2. SA3 does not support headdim >= 256
            use_fallback = False
            if attn_mask is not None:
                use_fallback = True
            if q_s.size(-1) >= 256:
                use_fallback = True
            
            if use_fallback:
                out = sdpa(q_s, k_s, v_s, attn_mask=attn_mask, is_causal=is_causal)
            else:
                out = sageattn3_blackwell(
                    q_s, k_s, v_s,
                    is_causal=is_causal,
                    per_block_mean=(sage_attention == "sageattn3_per_block_mean")
                )
            
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
        use_pytorch_fallback = False
        try:
            out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        except Exception as e:
            # Keep this suppression narrowly scoped so only the known SD1.5-style
            # unsupported head_dim=160 error is silenced.
            if "Unsupported head_dim: 160" in str(e):
                msg_key = "unsupported_head_dim_160"
                if msg_key not in _logged_sage_fallback_messages:
                    logging.info("SageAttention head_dim=160 is unsupported; using pytorch attention fallback.")
                    _logged_sage_fallback_messages.add(msg_key)
            else:
                logging.error("Error running sage attention: {}, using pytorch attention instead.".format(e))
            use_pytorch_fallback = True

        if use_pytorch_fallback:
            if tensor_layout == "NHD":
                q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
            return comfy_attention.attention_pytorch(
                q, k, v, heads,
                mask=mask,
                skip_reshape=True,
                skip_output_reshape=skip_output_reshape,
                **kwargs
            ).to(in_dtype)

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
                # disabled: Load Flash-Attention if available (without --use-flash-attention option)
                if "transformer_options" in model.model_options:
                    if "optimized_attention_override" in model.model_options["transformer_options"]:
                        del model.model_options["transformer_options"]["optimized_attention_override"]
                
                # Get Flash-Attention info using our own function (same as SA)
                flash_is_available, flash_attn_version, flash_attn_type = get_flash_attention_info()
                
                if flash_is_available and hasattr(comfy_attention, 'attention_flash'):
                    # Set Flash-Attention as override
                    if "transformer_options" not in model.model_options:
                        model.model_options["transformer_options"] = {}
                    
                    def attention_override_flash(func, *args, **kwargs):
                        return comfy_attention.attention_flash(*args, **kwargs)
                    model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_flash
                    
                    # Log Flash-Attention version (same format as SA)
                    logging.info("Restoring initial comfy attention")
                    if flash_attn_version and flash_attn_version != "unknown":
                        if flash_attn_type:
                            logging.info(f"[ComfyUI] Using {flash_attn_type} (Flash-Attention {flash_attn_version}) direct")
                        else:
                            logging.info(f"[ComfyUI] Using Flash-Attention {flash_attn_version} direct")
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
            # Get Flash-Attention info using our own function (same as SA)
            flash_is_available, flash_attn_version, flash_attn_type = get_flash_attention_info()
            
            if flash_is_available:
                logging.info("Restoring initial comfy attention")
                # Log Flash-Attention version (same format as SA)
                if flash_attn_version and flash_attn_version != "unknown":
                    if flash_attn_type:
                        logging.info(f"[ComfyUI] Using {flash_attn_type} (Flash-Attention {flash_attn_version}) direct")
                    else:
                        logging.info(f"[ComfyUI] Using Flash-Attention {flash_attn_version} direct")
                else:
                    logging.info("[ComfyUI] Using Flash-Attention direct")
            else:
                logging.info("Restoring initial comfy attention")
        
        model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_attention_enable)
        model_clone.add_callback(CallbacksMP.ON_CLEANUP, patch_attention_disable)
        
        return model_clone,
