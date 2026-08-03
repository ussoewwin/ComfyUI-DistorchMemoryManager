# SageAttention3 (SA3) Integration — Complete Guide

## 1. Background and goals

### 1.1 Problem
- Target hardware: RTX 5060 Ti 16GB (Blackwell architecture)
- With `--use-sage-attention`, switch freely between SA2 (SageAttention 2.x) and SA3 (SageAttention 3.0) from the node UI
- The UI already exposes `sageattn3`, but the backend needed improvements

### 1.2 Goals
1. Add SA3 version detection and logging
2. Skip SA2 version logs when SA3 is selected
3. Implement fallbacks for SA3 constraints

---

## 2. Implementation plan

### 2.1 Survey results

| Component | File | SA3 support |
|---|---|---|
| ComfyUI core | `attention.py` | Implemented (`attention3_sage`) |
| DistorchMemoryManager | `nodes/sa.py` | Basic support; improvements needed |
| sageattn3 package | `sageattn3/api.py` | v3.0.0.b1 Blackwell FP4 |

### 2.2 Improvement points
1. No SA3-specific version detection helper
2. SA2 version logs still printed when SA3 is selected
3. No handling for SA3 constraints (`headdim >= 256`, no mask support)

---

## 3. Modified files

### Target
`nodes/sa.py`

---

## 4. Added / changed code

### 4.1 SA3 version detection

**Location:** lines 102–125

```python
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
```

**Meaning:**
- `sage3_version`: SA3 version string (e.g. `3.0.0.b1`)
- `is_available`: whether the SA3 package is installed
- `supports_blackwell`: whether Blackwell kernels are available

---

### 4.2 Conditional logging

**Location:** lines 150–163

```python
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
```

**Meaning:**
- When SA3 is selected (`sageattn3` or `sageattn3_per_block_mean`), skip SA2 logs
- Print SA2 version info only for SA2 modes

---

### 4.3 Improved SA3 path

**Location:** lines 184–220

```python
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
```

---

## 5. Technical details

### 5.1 Tensor layout conversion

```
NHD: [batch, seq_len, heads, dim]  <- ComfyUI default
HND: [batch, heads, seq_len, dim]  <- layout SA3 expects
```

Conversion:

```python
q_s, k_s, v_s = [x.transpose(1, 2) for x in (q, k, v)]
```

### 5.2 SA3 constraints and fallback

| Constraint | Reason | Fallback |
|---|---|---|
| `headdim >= 256` | SA3 FP4 kernel unsupported | PyTorch SDPA |
| `attn_mask != None` | SA3 has no mask path | PyTorch SDPA |

### 5.3 `per_block_mean` option

Inside `preprocess_qkv` in `sageattn3/api.py`:

```python
if per_block_mean:
    # Mean over 128-token blocks (Triton kernel; faster)
    q, qm = triton_group_mean(q)
else:
    # Mean over the full sequence (more accurate)
    qm = q.mean(dim=-2, keepdim=True)
    q = q - qm
```

**Selection guide:**
- `sageattn3`: standard mode, accuracy-first
- `sageattn3_per_block_mean`: faster mode, Triton-optimized

---

## 6. How to use

### 6.1 Launch ComfyUI
```bash
python main.py --use-sage-attention
```

### 6.2 Node settings
1. Add the `Patch Sage Attention DM` node
2. Choose from the `sage_attention` dropdown:
   - SA2 family: `auto`, `sageattn_qk_int8_pv_fp16_cuda`, etc.
   - SA3 family: `sageattn3`, `sageattn3_per_block_mean`

### 6.3 Expected logs

**SA2 selected:**
```
Patching comfy attention to use SageAttention 2.1.1+cu128torch2.7
```

**SA3 selected:**
```
Patching comfy attention to use SageAttention3 3.0.0.b1 (Blackwell FP4)
```

---

## 7. Diff summary

| Lines | Change | Content |
|---|---|---|
| 102–125 | Added | `get_sage_attention3_info()` |
| 151–152 | Modified | Skip SA2 logs when SA3 is selected |
| 184–188 | Added | SA3-specific version detection / logging |
| 195–200 | Modified | Clarified NHD↔HND layout conversion |
| 202–210 | Added | Fallback decision logic |
| 211–218 | Modified | Fallback path vs SA3 call branch |
