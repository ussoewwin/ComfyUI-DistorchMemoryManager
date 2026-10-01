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
sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean", "spargeattn"]
_logged_sage_fallback_messages = set()
_logged_sparge_fallback_messages = set()


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


# SpargeAttn-hswq (spas_sage_hswq_attn) version detection (independent from model_management)
def get_sparge_attn_info():
    """
    Get SpargeAttn-hswq package (spas_sage_hswq_attn) version information.
    Returns: (version, is_available)
    """
    sparge_version = None
    is_available = False
    try:
        import spas_sage_hswq_attn  # noqa: F401
        is_available = True
    except Exception:
        return sparge_version, is_available
    try:
        from importlib.metadata import version as _pkg_version
        sparge_version = _pkg_version("spas_sage_hswq_attn")
    except Exception:
        sparge_version = "unknown"
    return sparge_version, is_available


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

    tail = _build_sage_attention_tail(sage_func, allow_compile)
    return tail


def get_sparge_func_dm(sparge_topk=0.5):
    """
    Build the SpargeAttn-hswq attention function (completely separate from the
    SageAttention path above).

    Uses spas_sage_hswq_attn.spas_sage2_attn_meansim_topk_cuda (the fork's
    recommended plug-and-play API, based on SageAttention2++ quantized
    kernels with two-stage block-sparse filtering).

    Constraints (handled by explicit fallback to PyTorch SDPA, never silent
    quality loss without a log line):
    - No attention mask support (the API accepts attn_mask but ignores it;
      pass-through would silently drop masking)
    - headdim must be 64 or 128 (kernel assert)
    - seq_len must be >= 128 (kernel assert)
    """
    sparge_version, sparge_available = get_sparge_attn_info()
    # Sanitize widget inputs:
    # - None: workflow saved before v2.4.7 (input did not exist yet)
    # - 0.0 / negative / non-numeric: widget-shifted legacy save (UI shows "0")
    # - NaN / >1.0: out of range
    # All map to the default 0.5; values in (0, 1] are kept as-is.
    try:
        _k = float(sparge_topk)
    except Exception:
        _k = 0.5
    if not (_k > 0.0) or _k > 1.0:
        logging.info(f"SpargeAttn: invalid sparge_topk={sparge_topk!r} (kernel requires (0, 1]); using default 0.5")
        _k = 0.5
    sparge_topk = _k
    if sparge_available:
        logging.info(f"Patching comfy attention to use SpargeAttn-hswq {sparge_version or 'unknown'} (spas_sage_hswq_attn, topk={sparge_topk})")
    else:
        logging.warning("spas_sage_hswq_attn not installed; SpargeAttn mode will use pytorch attention fallback for every call.")

    from torch.nn.functional import scaled_dot_product_attention as sdpa

    def sparge_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
        # Explicit constraint checks -> SDPA fallback (SpargeAttn kernels do not
        # consume attn_mask, and only headdim 64/128 with seq_len>=128 work).
        headdim = q.size(-1)
        seq_len = q.size(-2) if tensor_layout == "HND" else q.size(-3)
        use_fallback = False
        if attn_mask is not None:
            use_fallback = True
        if headdim not in (64, 128):
            use_fallback = True
        if seq_len < 128:
            use_fallback = True

        if use_fallback:
            if tensor_layout == "NHD":
                out = sdpa(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
            else:
                q_s, k_s, v_s = [x.transpose(1, 2) for x in (q, k, v)]
                out = sdpa(q_s, k_s, v_s, attn_mask=attn_mask, is_causal=is_causal)
                out = out.transpose(1, 2)
            return out

        if not sparge_available:
            raise RuntimeError("spas_sage_hswq_attn is not installed")

        from spas_sage_hswq_attn import spas_sage2_attn_meansim_topk_cuda

        # API expects HND layout.
        if tensor_layout == "NHD":
            q_s, k_s, v_s = [x.transpose(1, 2) for x in (q, k, v)]
        else:
            q_s, k_s, v_s = q, k, v
        out = spas_sage2_attn_meansim_topk_cuda(
            q_s, k_s, v_s,
            topk=float(sparge_topk),
            is_causal=is_causal,
            tensor_layout="HND",
        )
        return out.transpose(1, 2) if tensor_layout == "NHD" else out

    sparge_func = torch.compiler.disable()(sparge_func)

    @wrap_attn
    def attention_sparge(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
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
        use_pytorch_fallback = False
        try:
            out = sparge_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        except Exception as e:
            err = str(e)
            if "not installed" in err:
                msg_key = "spas_not_installed"
                if msg_key not in _logged_sparge_fallback_messages:
                    logging.warning("spas_sage_hswq_attn is not installed; using pytorch attention fallback (SpargeAttn mode).")
                    _logged_sparge_fallback_messages.add(msg_key)
            elif "headdim should be in" in err or "seq_len should be not less than" in err:
                msg_key = "sparge_shape_" + err.split(" ")[0][:24]
                if msg_key not in _logged_sparge_fallback_messages:
                    logging.info(f"SpargeAttn shape constraint hit ({err.strip()[:80]}); using pytorch attention fallback.")
                    _logged_sparge_fallback_messages.add(msg_key)
            else:
                logging.error("Error running SpargeAttn attention: {}, using pytorch attention instead.".format(e))
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
    return attention_sparge


# --- SageAttention attention wrapper (unchanged, dedicated path) ---
def _build_sage_attention_tail(sage_func, allow_compile):
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
            "allow_compile": ("BOOLEAN", {"default": False, "tooltip": "Allow the use of torch.compile for the sage attention function, requires latest sageattn 2.2.0 or higher."}),
            "sparge_topk": ("COMBO", {"options": ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"], "default": "0.5", "tooltip": "SpargeAttn mode only: KV block keep ratio. Higher = more accurate, lower = faster. 1.0 = compute all blocks (no skipping). Kept after allow_compile so workflows saved before v2.4.7 load unchanged. String-based COMBO so the UI can never hold a broken value like 0. "})
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching attention mode. This doesn't use the model patching system and thus can't be disabled without running the node again with 'disabled' option."
    CATEGORY = "Memory"

    def patch(self, model, sage_attention, sparge_topk=None, allow_compile=False):
        model_clone = model.clone()
        
        @torch.compiler.disable()
        def patch_attention_enable(model):
            if sage_attention != "disabled":
                if sage_attention == "spargeattn":
                    # sparge_topk may arrive as None (old workflows without the
                    # input) or 0.0 (widget-shifted legacy save shown as "0" in
                    # the UI). get_sparge_func_dm sanitizes both to 0.5.
                    _topk = sparge_topk
                    try:
                        # COMBO UI stores strings; legacy workflows may carry a
                        # float, or a shifted/garbage value. All funnel through
                        # get_sparge_func_dm's sanitizer.
                        _topk = float(_topk) if _topk is not None else None
                    except Exception:
                        _topk = None
                    new_attention = get_sparge_func_dm(sparge_topk=_topk)
                else:
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
