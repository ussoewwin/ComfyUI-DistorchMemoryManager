# Flash-Attention Auto-Load and Version Detection — Complete Guide

## Why this was added

ComfyUI's `model_management` module has no official API to read Flash-Attention or SageAttention version strings. Previously, every ComfyUI update required manually editing `model_management.py` to restore version reporting.

To remove that maintenance cost, `ComfyUI-DistorchMemoryManager` implements fully independent version detection plus Flash-Attention auto-load inside the custom node. Version reporting no longer depends on ComfyUI updates, and when SageAttention is set to `disabled`, Flash-Attention (FA-2 / FA-3) can load automatically.

## Files added / modified

**File:** `sa.py`

Only this file was changed for this feature.

## Core behavior

### Runtime flow

1. **Generation start (`ON_PRE_RUN` callback):**
   - SageAttention enabled: obtain the SageAttention function, set `optimized_attention_override`, apply the patch, and log the SageAttention version.
   - SageAttention `disabled`: if the Flash-Attention package is importable, set `comfy_attention.attention_flash` as `optimized_attention_override` and log the Flash-Attention version.

2. **Generation end (`ON_CLEANUP` callback):**
   - Remove `optimized_attention_override` so ComfyUI resets its attention kernel.
   - ComfyUI re-selects the preferred kernel (typically Flash-Attention when available).
   - Log Flash-Attention version at this point as well (even if SageAttention was active during the run).

### Important points

- **No CLI flag required:** Flash-Attention can load from the node's `disabled` setting alone; `--use-flash-attention` is not required.
- **Per-generation logging:** SageAttention / Flash-Attention version lines are emitted every generation.
- **FA log while SA is on:** After cleanup, ComfyUI's kernel reset often selects Flash-Attention as the baseline; that log still appears when SageAttention was used for the run.

## Added code and meaning

### 1. Flash-Attention version helper (lines 22–67)

```python
def get_flash_attention_info():
    """
    Get Flash-Attention version and type information.
    Returns: (is_available, version, type)
    """
```

**Meaning:** Reports whether Flash-Attention is importable, its version string, and FA-2 vs FA-3. Fully independent of `model_management`.

**Flow:**
1. Try `import flash_attn`; on success set `flash_is_available = True`.
2. Read `flash_attn.__version__` when present.
3. If missing, use `importlib.metadata.version("flash-attn")`.
4. Split the version on `.` and take the major component.
5. `major_version >= 3` → `flash_attn_type = "FA-3"`, else `"FA-2"`.
6. Return `(is_available, version, type)`.

**Note:** Availability is based on import success only, not `args.use_flash_attention`, so the reported state matches what the process can actually load.

### 2. SageAttention version helper (lines 70–99)

```python
def get_sage_attention_info():
    """
    Get SageAttention version information.
    Returns: (version, cuda_version, torch_version)
    """
```

**Meaning:** Returns SageAttention version plus CUDA / PyTorch versions. Independent of `model_management`.

**Flow:**
1. `import sageattention`.
2. Prefer `sageattention.__version__`.
3. Else `importlib.metadata.version("sageattention")`.
4. Read `torch.version.cuda`.
5. Read `torch.version.__version__`.
6. Return `(version, cuda_version, torch_version)`.

### 3. Flash-Attention enabled check (lines 102–122)

```python
def is_flash_attention_enabled():
    """
    Check if Flash-Attention is currently enabled.
    Returns: bool
    """
```

**Meaning:** Detects whether Flash-Attention is the active attention path. Used mainly from `patch_attention_disable`; load control for `disabled` always attempts Flash-Attention when the package exists.

**Flow:**
1. Check whether `comfy_attention.optimized_attention.__name__` is `"attention_flash"`.
2. If not, combine `get_flash_attention_info()` availability with `args.use_flash_attention`.
3. Return `bool`.

### 4. `get_sage_func_dm()` change (lines 124–201)

**Change:** Use `get_sage_attention_info()` for logging.

```python
def get_sage_func_dm(sage_attention, allow_compile=False):
    # Detect SageAttention version using our own function
    sage_version, cuda_version, torch_version = get_sage_attention_info()
    
    if sage_version and sage_version != "unknown":
        if cuda_version != "unknown" and torch_version != "unknown":
            logging.info(f"Patching comfy attention to use SageAttention {sage_version}+cu{cuda_version}torch{torch_version}")
        else:
            logging.info(f"Patching comfy attention to use SageAttention {sage_version}")
    else:
        logging.info("Patching comfy attention to use sageattn")
```

**Meaning:** Centralizes version lookup in `get_sage_attention_info()`, removes duplication, and improves maintainability.

### 5. `patch_attention_enable()` change (lines 224–262)

**Key change:** When set to `disabled`, actually load Flash-Attention.

```python
@torch.compiler.disable()
def patch_attention_enable(model):
    if sage_attention != "disabled":
        # SageAttention enabled (existing path)
        new_attention = get_sage_func_dm(sage_attention, allow_compile=allow_compile)
        def attention_override_sage(func, *args, **kwargs):
            return new_attention.__wrapped__(*args, **kwargs)
        
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
    else:
        # disabled: load Flash-Attention without requiring CLI flags
        if "transformer_options" in model.model_options:
            if "optimized_attention_override" in model.model_options["transformer_options"]:
                del model.model_options["transformer_options"]["optimized_attention_override"]
        
        # Fetch Flash-Attention info
        flash_is_available, flash_attn_version, flash_attn_type = get_flash_attention_info()
        
        if flash_is_available and hasattr(comfy_attention, 'attention_flash'):
            # Install Flash-Attention as override
            if "transformer_options" not in model.model_options:
                model.model_options["transformer_options"] = {}
            
            def attention_override_flash(func, *args, **kwargs):
                return comfy_attention.attention_flash(*args, **kwargs)
            model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_flash
            
            # Log Flash-Attention version (same style as SA)
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
```

**Meaning:** Runs on `ON_PRE_RUN`. If SageAttention is `disabled` and Flash-Attention is available, installs `comfy_attention.attention_flash` as `optimized_attention_override` and logs the version.

**Note:** Load decision ignores `args.use_flash_attention`; it only requires a successful package import and `attention_flash` on the module.

### 6. `patch_attention_disable()` change (lines 264–285)

**Key change:** Always log Flash-Attention on cleanup, including when SageAttention was active.

```python
@torch.compiler.disable()
def patch_attention_disable(model):
    # Remove override so ComfyUI can reset its attention kernel
    if "transformer_options" in model.model_options:
        if "optimized_attention_override" in model.model_options["transformer_options"]:
            del model.model_options["transformer_options"]["optimized_attention_override"]
    
    # Even when SA was on, log FA after reset (ComfyUI often selects FA as baseline)
    flash_is_available, flash_attn_version, flash_attn_type = get_flash_attention_info()
    
    if flash_is_available:
        logging.info("Restoring initial comfy attention")
        # Log Flash-Attention version (same style as SA)
        if flash_attn_version and flash_attn_version != "unknown":
            if flash_attn_type:
                logging.info(f"[ComfyUI] Using {flash_attn_type} (Flash-Attention {flash_attn_version}) direct")
            else:
                logging.info(f"[ComfyUI] Using Flash-Attention {flash_attn_version} direct")
        else:
            logging.info("[ComfyUI] Using Flash-Attention direct")
    else:
        logging.info("Restoring initial comfy attention")
```

**Meaning:** Runs on `ON_CLEANUP`. Clears the override, lets ComfyUI reset kernels, and logs Flash-Attention when available (including after SageAttention runs).

**Note:** ComfyUI's per-generation kernel reset often restores Flash-Attention as the baseline, so this cleanup log appears every generation.

## How Flash-Attention load works

### Control via `optimized_attention_override`

ComfyUI's model patching path honors `model.model_options["transformer_options"]["optimized_attention_override"]`. Setting that key bypasses default attention selection and calls the provided function.

When `disabled`, this node installs Flash-Attention as follows:

```python
def attention_override_flash(func, *args, **kwargs):
    return comfy_attention.attention_flash(*args, **kwargs)
model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_flash
```

Calls to `comfy.ldm.modules.attention.optimized_attention` then run `attention_flash` directly.

### `attention_flash` implementation

`attention_flash` is defined in ComfyUI's `comfy/ldm/modules/attention.py` (around line 680) and calls `flash_attn_wrapper` (`flash_attn_func`).

## How version strings are obtained

### Flash-Attention

1. **Direct import:** `import flash_attn`
2. **Version attribute:** `flash_attn.__version__`
3. **Fallback:** `importlib.metadata.version("flash-attn")`
4. **FA-2 / FA-3:** major from `split('.')`
   - `major_version >= 3` → `"FA-3"`
   - otherwise → `"FA-2"`

### SageAttention

1. **Direct import:** `import sageattention`
2. **Version attribute:** `sageattention.__version__`
3. **Fallback:** `importlib.metadata.version("sageattention")`
4. **CUDA / PyTorch:** `torch.version.cuda` and `torch.version.__version__`

## Design notes

### 1. Full independence

No dependency on `model_management` version helpers. ComfyUI updates do not break reporting.

### 2. Load without CLI flags

Flash-Attention can be selected from the node's `disabled` mode using package importability alone.

### 3. Per-generation logs

- **ON_PRE_RUN (before generate):** SageAttention log when SA is on; Flash-Attention log when `disabled`.
- **ON_CLEANUP (after generate):** Always re-check Flash-Attention and log (including after SA runs, because kernel reset restores the FA baseline when available).

### 4. Error handling

Important paths are wrapped in try/except so failures degrade safely.

### 5. Kernel reset behavior

ComfyUI resets attention kernels each generation. When Flash-Attention is available, that baseline is often FA, which is why FA logs still appear after SageAttention generations.

## Summary

`ComfyUI-DistorchMemoryManager` now owns independent Flash-Attention / SageAttention version detection and Flash-Attention auto-load when SageAttention is `disabled`. There is no reliance on `model_management` patches, CLI flags are optional, and ComfyUI updates no longer require editing core files for version logging.
