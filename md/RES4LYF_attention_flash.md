# RES4LYF Flash Attention Fix — Full Technical Note

This document records **what failed, why, the underlying issue, what changed, and what each change means** for the fix applied to `sd/attention.py` shipped with the custom node **RES4LYF**.

---

## 1. Summary

- **Symptom**: Logs showed `Flash Attention failed, using default SDPA:` followed repeatedly by PyTorch internals such as  
  `schema_.has_value() INTERNAL ASSERT FAILED` (near `ATen/core/dispatch/OperatorEntry.h`).
- **Typical environment**: PyTorch **2.11** + CUDA **13.x** (e.g. `2.11.0+cu130`), ComfyUI with `--use-flash-attention`, workflows that use **attention code paths through RES4LYF**.
- **Fix**: In RES4LYF’s `attention_flash` implementation, **remove the extra `torch.library.custom_op` wrapper** and **remove the SDPA fallback**; **call `flash_attn_func` directly** instead.

---

## 2. What the error looked like (log output)

A typical sequence was:

1. **Warning**  
   `Flash Attention failed, using default SDPA: ...`

2. **PyTorch internal assert** (example)  
   `schema_.has_value() INTERNAL ASSERT FAILED at "...OperatorEntry.h":84`  
   `Tried to access the schema for  which doesn't have a schema registered yet`  
   `please report a bug to PyTorch.`

Exact wording can vary by build/version, but the nature is the same: **dispatcher / operator schema resolution failure**.

---

## 3. Cause (what the code was doing)

Before the fix, RES4LYF `attention.py` roughly had a **two-stage** structure.

### 3.1 Stage 1: Another custom op layered on top of `flash_attn_func`

Following the same pattern as ComfyUI core, it wrapped flash-attn’s `flash_attn_func` like this:

```python
@torch.library.custom_op("flash_attention::flash_attn", mutates_args=())
def flash_attn_wrapper(q, k, v, dropout_p=0.0, causal=False):
    return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
```

That registers a **new** PyTorch custom op name: **`flash_attention::flash_attn`**.

Meanwhile, **the flash-attn package itself** (see `flash_attn_interface.py`) uses `torch.library.custom_op` and related APIs for PyTorch 2.4+ integration.

So the same computation was routed through **two custom-op layers**: **flash-attn’s internal registration** plus **the RES4/ComfyUI-style alias `flash_attention::flash_attn`**.

### 3.2 Stage 2: Fallback to `torch.nn.functional.scaled_dot_product_attention` on exception

```python
try:
    out = flash_attn_wrapper(...)
except Exception as e:
    logging.warning(f"Flash Attention failed, using default SDPA: {e}")
    out = torch.nn.functional.scaled_dot_product_attention(...)
```

If stage 1 raised or hit dispatcher inconsistency, execution **fell through to SDPA** after the warning. That path could hit **dispatcher issues of the same family**, which is why logs looked like **warning + internal assert** in succession.

---

## 4. The underlying issue (design level)

| Aspect | Issue |
|--------|--------|
| **Duplicate abstraction** | `flash_attn_func` is already the supported entry point for integrating flash-attn with PyTorch. **Stacking another ComfyUI-style custom op on top** is prone to dispatcher clashes on some versions. |
| **Fallback target** | SDPA also goes through PyTorch’s attention backends and can **trigger another failure on the same dispatcher**, not “a safe escape hatch.” |
| **Confusion with “bad build”** | Internal PyTorch paths in the log push people toward **ABI / wheel** theories; in this incident the main driver was **how this node’s bundled attention code called flash-attn**. |

The core problem: **deviation from the intended API (`flash_attn_func` directly)** plus **routing failures into SDPA**.

---

## 5. Why it didn’t show on PyTorch 2.10 but did on 2.11

- With **the same RES4 code**, differences in **`OperatorEntry` / schema resolution** mean 2.10 often **did not reach** the same internal asserts (or behaved differently on the same path).
- **2.11 changed dispatcher behavior**, so the **double wrapper + fallback** combination **surfaced** there.

This does **not** mean “2.10 was correct”—it means **latent risk that didn’t surface on 2.10**.

---

## 6. File changed

| Field | Value |
|-------|--------|
| **Path** | `ComfyUI/custom_nodes/RES4LYF/sd/attention.py` |
| **Function** | `attention_flash` |
| **Assumption** | `from flash_attn import flash_attn_func` succeeds near the top of the file (when using `--use-flash-attention`). |

---

## 7. What changed (code details)

### 7.1 Removed

1. The entire **`flash_attn_wrapper` block**  
   - `@torch.library.custom_op("flash_attention::flash_attn", ...)`  
   - `@flash_attn_wrapper.register_fake`  
   - The `AttributeError` stub `flash_attn_wrapper`

2. The **`try` / `except` inside `attention_flash`**  
   - Success path: `flash_attn_wrapper(q.transpose(1, 2), ...)`  
   - Failure path: `logging.warning(...)` + `torch.nn.functional.scaled_dot_product_attention(...)`

### 7.2 Core of `attention_flash` after the fix (excerpt)

```python
def attention_flash(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    assert mask is None
    # Match ComfyUI core: call flash_attn_func directly (avoid duplicate custom_op + broken SDPA fallback on torch 2.11+).
    out = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        causal=False,
    ).transpose(1, 2)
    if not skip_output_reshape:
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
    return out
```

(The real file continues with `optimized_attention` wiring as before.)

---

## 8. Meaning of each change

| Change | Meaning |
|--------|---------|
| **Remove `flash_attn_wrapper`** | Stops registering **`flash_attention::flash_attn`** as a second custom op, avoiding dispatcher mismatches common on PyTorch 2.11.x. |
| **Call `flash_attn_func` directly** | Restores the **supported API path**; leave custom-op registration to the flash-attn package. |
| **Remove `try` / SDPA** | Avoids falling into the **broken SDPA path** on failure; exceptions propagate, breaking the “FA fails → SDPA hits a second failure” chain. |
| **`assert mask is None`** | Makes explicit that this path does not support masks (aligned with the old `RuntimeError` inside `try`). |
| **Comment** | Explains why we **do not** keep the ComfyUI-style wrapper. |

---

## 9. Relationship to ComfyUI core

- **ComfyUI core** `comfy/ldm/modules/attention.py` may still contain the **same pattern** (`flash_attn_wrapper` + SDPA fallback) historically.
- In this incident, **logs pointed to RES4LYF’s `sd/attention.py`**, and **symptoms cleared with only the RES4 fix**—meaning **that workflow actually used RES4’s attention implementation**.
- If you avoid touching core, **other workflows that hit core’s same pattern** could still show issues; mitigations include **runtime patching from another custom node** so updates don’t overwrite edits.

---

## 10. Operational notes

1. **Updating RES4LYF** may **overwrite** `sd/attention.py` and **remove** this fix. Re-check diffs after upgrades using this note.
2. **`--use-flash-attention`** and an installed **`flash-attn`** remain prerequisites.
3. After the fix, if Flash Attention **truly fails** (e.g. OOM), it **no longer auto-falls back to SDPA**; you’ll see an **exception** instead. That trades “silent double failure” for easier debugging.

---

## 11. One-sentence summary

**RES4LYF wrapped `flash_attn_func` in an extra `custom_op` and fell back to SDPA on failure, which conflicted with PyTorch 2.11’s dispatcher and produced internal asserts in logs; calling `flash_attn_func` directly and removing the SDPA fallback removes that underlying risk.**

---

*Document note: Based on facts at investigation/fix time. Behavior may vary with PyTorch / ComfyUI / RES4LYF versions.*
