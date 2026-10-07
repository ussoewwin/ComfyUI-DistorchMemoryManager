# HSWQ Purge Gate Reconciliation — Complete Fix Guide (ComfyUI-VRAM-Manager v2.4.8)

**Date:** 2026-10-07
**Repos / commits:**
- ComfyUI-DistorchMemoryManager: `b266a48` (Method 2d), `f716917` (v2.4.8 version + changelog + root/nodes sync)
- ComfyUI-HSWQ-Loader-and-Tools: `409385f` (purge-rearm API), `a5134b7` (revert of `bdfcfcb`)

**Symptom fixed:** alternating corrupt/correct images across consecutive prompts in the
same ComfyUI process when a Distorch purge (HSWQ toggle) runs between generations —
including the FIRST prompt being corrupt (`bad→good→bad`).

---

## 1. What was the problem

### 1.1 Observed behavior (real logs, one process, same workflow, same loader code)

| Session | run1 | run2 | run3 | run4 |
|---|---|---|---|---|
| 2026-10-06 21:37 (single run) | good | — | — | — |
| 2026-10-07 00:11 | good | good | — | — |
| 2026-10-07 12:11 (V7b) | good | **BAD** | — | — |
| 2026-10-07 12:34 (pure 9e72c30 loader) | **BAD** | good | **BAD** | — |
| 2026-10-07 13:51 (after this fix) | good | good | good | good |

"BAD" = malformed composition (multiple heads, deformed hands) — ControlNet-signal
corruption, not noise. All runs logged identical loader diagnostics, identical
`bake hook ON` counts (12 installs / 4 peels per session), identical LoRA EVIDENCE
lines; only run-position differed. That proved the corruption was **process state
carried across prompts, not loader code**.

### 1.2 The state that crosses runs: the HSWQ purge peel

Inside run 1 (mid-sampling), the workflow's DisTorchPurgeVRAMV2 node (HSWQ toggle on)
executed, in this order:

```
HSWQ INT8/NVFP4: Starting purge process...
... unload_all_models() / gc nuclear / PINNED sweep ...
HSWQ INT8/NVFP4: HSWQ stack peel uninstall_zimage_nvfp4_lora_bake -> True
[WARNING] restore TC stack: no saved product refs (SDXL needs apply_comfy_quant_nvfp4_patches first)
HSWQ INT8/NVFP4: HSWQ stack peel restore_nvfp4_tc_product_stack -> False
HSWQ INT8/NVFP4: Peeled NVFP4/ZI Linear LoRA bake wrap gc:comfy.ops.Linear.convert_weight/set_weight (was ver=8)
HSWQ INT8/NVFP4: Done — cleared 0 HSWQ INT8/NVFP4 ref(s), ... approx 0.0 MB tracked
```

The purge **peels HSWQ overlays** (Z Image NVFP4 bake/parity wraps, Linear LoRA-bake
class wraps, INT8 protect load arm, Krea2 stack, SDXL product TC) — a mechanism built
for the **Z Image → SDXL handoff** ("so the next model, SDXL, never inherits Z Image
state"). For every other case — the next prompt reloading **Z Image / Qwen INT8 /
Krea2 again** — the peel is a pure side effect: the wrappers are gone, and nothing
restores them.

---

## 2. The essential root cause: stale module-level apply gates

HSWQ's `apply_*` functions early-return when a module-level boolean says "already
applied". That boolean is **invisible to the purge**, which removes the actual
wrappers. After a peel, the next load path skips re-application and sampling runs on
a half-peeled stack.

Concretely (`nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py`):

```python
if (
    _PATCHES_APPLIED            # still True after the purge peel
    and _load_wrap_ok           # NEW (fdc60bc, 2026-08-26): load wrap stamp check
    and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
    and stack_ver >= _NVFP4_STACK_VER
):
    return True                 # skipped re-application when all four look healthy
```

The 2026-08-26 fix (`fdc60bc` + permanent `Dynamic.load` guard `d97bb5b`) patched
exactly this class of bug **for the TC (W4A4) NVFP4 load wrap only**. The 2026-10
workflow runs the **hybrid parity stack + INT8 ControlNet**, and the peel surface is
larger than the 08-26 guard covered:

| Family | Gate (module boolean) | Wrappers the peel removes | 08-26 guard covered? |
|---|---|---|---|
| ZI ConvRot NVFP4 (zi_comfy_quant_nvfp4) | `_PATCHES_APPLIED` | `_load_quantized_module` wrap, `mixed_precision_ops` stack_ver wrap, `detect_unet_config` stamp | partially (load wrap only, TC path) |
| ZI comfy_parity | `_PARITY_APPLIED` | parity `mixed_precision_ops` + load wrap, Linear bake wraps (gc scan) | no |
| INT8 comfy_quant | `_PATCHES_APPLIED` | `_load_quantized_module` INT8 decode/protect arm | no |
| SDXL NVFP4 product | `_PATCHES_APPLIED` | product TC wrap | restore attempted but returns False in ZI-continuing graphs |
| Krea2 NVFP4 | `_PATCHES_APPLIED` / `_APPLIED` | krea2 stack + bake hooks | no |
| bake hooks | `_GPU_BAKE_INSTALLED` | `Dynamic.load` / `load_models_gpu` wraps | guard re-arms bake only |

Whether a run is corrupted depends on which layers each peel/refresh path happened to
leave behind at that moment — hence the **alternating good/bad pattern** and why even
run 1 could be bad (a peel from the *previous* session tail, RAM-pressure unload
ordering, etc.). No loader-code variant (warning suppression, probes, `named_parameters`
vs `state_dict`) correlated with the outcome once state was controlled; the loader was
fully exonerated (`V7bDIAG`: identical keep-sets A/B, 157/157, `onlyA=0 onlyB=0`).

### 2.1 Why "don't peel unless next is SDXL" was not implementable

At purge execution time the node cannot know which model the **next prompt** loads —
the graph is already finished; the toggle only means "wipe HSWQ residue now". So the
fix cannot be a conditional skip. The correct contract is the Owner's statement:

> "HSWQ clear is not only for SDXL: it must clear Z Image / Qwen / Krea2 **and**
> hand off cleanly to the next generation, for all conditions."

Implementation: keep the peel (SDXL-handoff semantics unchanged), and after ALL
peeling, **reconcile each family's gate against reality** so the next legitimate load
path rebuilds what it needs. That is a pure-state correction with zero side effects.

---

## 3. Files created / modified

| Repo | File | Change | Verification |
|---|---|---|---|
| HSWQ | `patches/hswq_purge_rearm.py` | **NEW** — `hswq_purge_rearm_state()` gate reconciliation API | sha256 `6b6b8fad7f6102a7…`; mock unit tests healthy/peel/idempotent |
| HSWQ | `__init__.py` | + registration block (pure function module import, installs nothing) | compile OK; boot log `HSWQ purge-rearm reconciliation API registered` |
| Distorch | `nodes/purge_vram.py` | + Method 2d call after peel/gc/kitchen-reset | sha256 `85a0d28afae47f35…`; live log `Method 2d reconcile -> {...RESET...}` |
| Distorch | `purge_vram.py` (root fallback) | synced byte-identical to `nodes/purge_vram.py` | previously stale by one 2026-08-26-era hunk + dead `_HSWQ_KEEP_LOADER_FLAGS` declaration (0 references) |
| Distorch | `__init__.py`, `pyproject.toml` | version 2.4.7 → **2.4.8** | — |
| Distorch | `changelog/changelog.md`, `zhmd/changelog/changelog.md` | + v2.4.8 entries (EN/ZH) with release-notes links | — |
| HSWQ | `nodes/hswq_model_patch_loader.py` | **revert `bdfcfcb`** (root-logger filter warning suppression removed; back to pure 9e72c30, blob `4d722b4043f10ef2`) | live == repo HEAD (hash-verified) |

Live copies (`custom_nodes/ComfyUI-DistorchMemoryManager`, `custom_nodes/ComfyUI-HSWQ-Loader-and-Tools`)
are hash-identical to the repos.

---

## 4. Full code of everything new/changed

### 4.1 `ComfyUI-HSWQ-Loader-and-Tools/patches/hswq_purge_rearm.py` (entire new file)

```python
"""Reconcile HSWQ install gates after Distorch purge peels the quant stacks.

Background (2026-10-07):
``ComfyUI-DistorchMemoryManager`` ``DisTorchPurgeVRAMV2`` (HSWQ toggle) peels
the HSWQ overlays (ZI bake / parity / Linear wraps, Krea2 stack, INT8 decode
overlay, SDXL product TC wrap) so a *later SDXL* load never inherits Z Image
state. The peel side was written assuming the next load is SDXL; it has no
counterpart for the current primary use case: the NEXT prompt loads Z Image
ConvRot NVFP4 + INT8 ControlNet / Qwen INT8 / Krea2 again in the same process.

The HSWQ ``apply_*`` functions gate on module-level booleans
(``_PATCHES_APPLIED`` / ``_APPLIED`` / ``_GPU_BAKE_INSTALLED``) that the purge
does not know about. After a peel, gate=True makes the next load path skip
re-application (early ``return True``) while the actual wrappers are gone —
the run executes on a half-peeled stack and produces corrupt images
(observed as alternating good/bad prompts in one session).

``hswq_purge_rearm_state()`` never unwraps or wraps anything and never touches
tensors. It only inspects the live function/class stamps on
comfy.ops / comfy.model_detection / comfy.utils / comfy.model_patcher and
resets each family's gate boolean when its layer is ABSENT, so the next load
path re-applies via the legitimate install code.

Branch separation (Owner philosophy separate, branch, never mix): ZI /
SDXL-product-NVFP4 / Krea2 / INT8 / bake / parity each have an INDEPENDENT
predicate and reset its OWN global. A peeled ZI layer never resets the SDXL
gate, and vice versa.

Callers: Distorch purge calls this at the end of its peel (after all
unload_all_models / nuclear gc / peel APIs). It is safe to call any number of
times on a healthy stack (no-op).
"""
from __future__ import annotations

import sys


def _console(msg: str) -> None:
    print(f"[HSWQ purge-rearm] {msg}", flush=True)


def _mod_by_dotted_end(dotted: str):
    """Find an imported module by its dotted tail (independent of the
    custom_nodes package spelling): ``nodes.zimage_nvfp4.zi_comfy_quant_nvfp4``
    matches ``custom_nodes.ComfyUI-HSWQ-Loader-and-Tools.nodes.zimage_nvfp4.zi_comfy_quant_nvfp4``."""
    tail = "." + dotted.lstrip(".")
    for name, mod in list(sys.modules.items()):
        if mod is not None and name.endswith(tail):
            return mod
    return None


def _closure_named(fn, name: str):
    try:
        cells = getattr(fn, "__closure__", None) or ()
        code = getattr(fn, "__code__", None)
        if code is None:
            return None
        for n, cell in zip(code.co_freevars, cells):
            if n == name:
                return cell.contents
    except Exception:
        return None
    return None


_LOAD_PREV_NAMES = (
    "_orig_load",
    "original_load",
    "orig_load",
    "cur",
    "_prev",
    "_prev_load",
    "prev_load",
    "base",
)
_MP_PREV_NAMES = ("_orig_mp", "original_mp", "mp_fn", "_cur_mp", "prev_mp", "base")


def _walk_layers(fn, attr: str, closure_names: tuple, limit: int = 24):
    """Yield every layer of a wrapper chain (comfy.ops load / mixed_precision_ops).

    Follows the explicit ``attr`` back-pointer first, then known closure cells.
    Cycle-safe via id-seen set.
    """
    seen: set[int] = set()
    cur = fn
    depth = 0
    while cur is not None and callable(cur) and id(cur) not in seen and depth < limit:
        seen.add(id(cur))
        yield cur
        depth += 1
        try:
            nxt = getattr(cur, attr, None)
        except Exception:
            nxt = None
        if nxt is None or nxt is cur:
            for nm in closure_names:
                nxt = _closure_named(cur, nm)
                if nxt is not None and nxt is not cur:
                    break
        if nxt is cur:
            break
        cur = nxt


def _chain_has(fn, attr: str, closure_names: tuple, predicate) -> bool:
    try:
        for layer in _walk_layers(fn, attr, closure_names):
            try:
                if predicate(layer):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def hswq_purge_rearm_state() -> dict:
    """Peel-aware gate reconciliation. Returns {family: action} for logging.

    families: zimage_nvfp4 / sdxl_nvfp4_product / krea2_nvfp4 / int8_comfy_quant
    / zimage_parity / krea2_parity / zimage_bake_flag / krea2_bake_flag
    """
    report: dict = {}
    try:
        import comfy.ops as ops
        import comfy.model_detection as model_detection
        import comfy.utils as comfy_utils
    except Exception as e:  # comfy missing: nothing to reconcile
        _console(f"skipped: comfy import failed ({e})")
        return {"skipped": str(e)}

    load_fn = getattr(ops, "_load_quantized_module", None)
    mp_fn = getattr(ops, "mixed_precision_ops", None)
    detect = getattr(model_detection, "detect_unet_config", None)
    convert_old_quants = getattr(comfy_utils, "convert_old_quants", None)

    # --- independent presence predicates per family (never mixed) ----------
    # ZI ConvRot NVFP4: load wrap with full_load but NOT product_tc and NOT
    # krea2_full_load; detect packed_dims without krea2 txtlayers; mp carrying
    # _hswq_nvfp4_stack_ver (ZI stamps it; SDXL product stamps product_tc too,
    # so exclude the product chain by absence of product_tc on THAT layer).
    zi_load_ok = _chain_has(
        load_fn,
        "_hswq_nvfp4_orig_load",
        _LOAD_PREV_NAMES,
        lambda f: bool(getattr(f, "_hswq_nvfp4_full_load", False))
        and not bool(getattr(f, "_hswq_nvfp4_product_tc", False))
        and not bool(getattr(f, "_hswq_krea2_full_load", False)),
    )
    zi_mp_ok = _chain_has(
        mp_fn,
        "_hswq_nvfp4_orig_mp",
        _MP_PREV_NAMES,
        lambda f: int(getattr(f, "_hswq_nvfp4_stack_ver", 0) or 0) > 0
        and not bool(getattr(f, "_hswq_nvfp4_product_tc", False))
        and not bool(getattr(f, "_hswq_krea2_stack", False)),
    )
    zi_detect_ok = bool(
        getattr(detect, "_hswq_nvfp4_packed_dims", False)
        and not getattr(detect, "_hswq_krea2_txtlayers_fix", False)
    )

    # SDXL ConvRot NVFP4 product: identified ONLY by _hswq_nvfp4_product_tc.
    sdxl_ok = _chain_has(
        load_fn, "_hswq_nvfp4_orig_load", _LOAD_PREV_NAMES,
        lambda f: bool(getattr(f, "_hswq_nvfp4_product_tc", False)),
    ) or _chain_has(
        mp_fn, "_hswq_nvfp4_orig_mp", _MP_PREV_NAMES,
        lambda f: bool(getattr(f, "_hswq_nvfp4_product_tc", False)),
    )

    # Krea2 ConvRot NVFP4: krea2-specific stamps only.
    krea2_ok = (
        _chain_has(
            load_fn, "_hswq_nvfp4_orig_load", _LOAD_PREV_NAMES,
            lambda f: bool(getattr(f, "_hswq_krea2_full_load", False)),
        )
        or _chain_has(
            mp_fn, "_hswq_nvfp4_orig_mp", _MP_PREV_NAMES,
            lambda f: bool(getattr(f, "_hswq_krea2_stack", False)),
        )
        or _chain_has(
            convert_old_quants, "_hswq_krea2_prev_oldquants", ("_prev", "prev", "original"),
            lambda f: bool(getattr(f, "_hswq_krea2_oldquants", False)),
        )
        or bool(getattr(detect, "_hswq_krea2_txtlayers_fix", False))
    )

    # INT8 comfy_quant decode overlay (patches/comfy_quant_int8).
    int8_ok = _chain_has(
        load_fn, "_hswq_nvfp4_orig_load", _LOAD_PREV_NAMES,
        lambda f: bool(getattr(f, "_hswq_int8_decode_patched", False))
        or bool(getattr(f, "_hswq_int8_protect_in_load", False))
        or bool(getattr(f, "_hswq_int8_protect_arm_v2", False)),
    )

    # --- gate resets: one branch per family, own global only ----------------
    def _reset_gate(mod, attr: str, family: str, missing: bool) -> None:
        if mod is None:
            report[family] = "module-not-imported"
            return
        if not getattr(mod, attr, False):
            report[family] = "gate-already-open"
            return
        if not missing:
            report[family] = "layer-present(no-op)"
            return
        try:
            setattr(mod, attr, False)
            report[family] = "RESET"
            _console(f"{family}: peeled layer detected -> {attr}=False (next load re-applies)")
        except Exception as e:
            report[family] = f"reset-failed:{e}"

    _reset_gate(
        _mod_by_dotted_end("nodes.zimage_nvfp4.zi_comfy_quant_nvfp4"),
        "_PATCHES_APPLIED",
        "zimage_nvfp4",
        not (zi_load_ok and zi_mp_ok and zi_detect_ok),
    )
    _reset_gate(
        _mod_by_dotted_end("nodes.nvfp4.comfy_quant_nvfp4"),
        "_PATCHES_APPLIED",
        "sdxl_nvfp4_product",
        not sdxl_ok,
    )
    _reset_gate(
        _mod_by_dotted_end("nodes.krea2_convrot_nvfp4.comfy_quant_nvfp4"),
        "_PATCHES_APPLIED",
        "krea2_nvfp4",
        not krea2_ok,
    )
    _reset_gate(
        _mod_by_dotted_end("patches.comfy_quant_int8"),
        "_PATCHES_APPLIED",
        "int8_comfy_quant",
        not int8_ok,
    )

    # Parity gates: their apply() refreshes idempotently from live stamps
    # (no stale-gate early return), so force them open after a peel to make
    # the next apply_nvfp4_comfy_parity()/krea2 equivalent re-arm the parity
    # chain cleanly instead of trusting a stale bool. Independent globals.
    zi_par = _mod_by_dotted_end("nodes.zimage_nvfp4.nvfp4_comfy_parity")
    if zi_par is not None and getattr(zi_par, "_PARITY_APPLIED", False):
        try:
            zi_par._PARITY_APPLIED = False  # type: ignore[attr-defined]
            report["zimage_parity"] = "RESET(self-heal refresh)"
        except Exception as e:
            report["zimage_parity"] = f"reset-failed:{e}"
    else:
        report["zimage_parity"] = "gate-already-open"

    k2_par = _mod_by_dotted_end("nodes.krea2_convrot_nvfp4.nvfp4_comfy_parity")
    if k2_par is not None and getattr(k2_par, "_APPLIED", False):
        try:
            k2_par._APPLIED = False  # type: ignore[attr-defined]
            report["krea2_parity"] = "RESET(self-heal refresh)"
        except Exception as e:
            report["krea2_parity"] = f"reset-failed:{e}"
    else:
        report["krea2_parity"] = "gate-already-open"

    # Bake installed flags: bookkeeping only (install is stamp-gated per
    # Dynamic.load). Reset when the corresponding bake stamp is absent from
    # the Dynamic.load chain, so nothing believes the hook is still installed.
    try:
        import comfy.model_patcher as _mp
        Dynamic = getattr(_mp, "ModelPatcherDynamic", None)
        cur_load = getattr(Dynamic, "load", None) if Dynamic is not None else None
    except Exception:
        cur_load = None
    dyn_chain: list = []
    seen_ids: set[int] = set()
    c = cur_load
    while c is not None and id(c) not in seen_ids:
        seen_ids.add(id(c))
        dyn_chain.append(c)
        nxt = getattr(c, "_hswq_zi_rearm_guard_prev", None)
        if nxt is None:
            nxt = _closure_named(c, "cur") or _closure_named(c, "original")
        if nxt is c:
            break
        c = nxt

    def _dyn_has(stamp: str) -> bool:
        return any(bool(getattr(f, stamp, False)) for f in dyn_chain)

    for dotted, stamp, family in (
        ("nodes.zimage_nvfp4.nvfp4_lora_bake", "_hswq_zi_nvfp4_lora_bake", "zimage_bake_flag"),
        ("nodes.krea2_convrot_nvfp4.nvfp4_lora_bake", "_hswq_krea2_nvfp4_lora_bake", "krea2_bake_flag"),
    ):
        mod = _mod_by_dotted_end(dotted)
        if mod is None:
            report[family] = "module-not-imported"
            continue
        if getattr(mod, "_GPU_BAKE_INSTALLED", False) and not _dyn_has(stamp):
            try:
                mod._GPU_BAKE_INSTALLED = False  # type: ignore[attr-defined]
                report[family] = "RESET"
            except Exception as e:
                report[family] = f"reset-failed:{e}"
        else:
            report[family] = "consistent(no-op)"

    layer_state = {
        "zi_load": zi_load_ok, "zi_mp": zi_mp_ok, "zi_detect": zi_detect_ok,
        "sdxl_product": sdxl_ok, "krea2": krea2_ok, "int8": int8_ok,
        "zi_bake": _dyn_has("_hswq_zi_nvfp4_lora_bake"),
        "zi_guard": _dyn_has("_hswq_zi_rearm_guard"),
        "krea2_bake": _dyn_has("_hswq_krea2_nvfp4_lora_bake"),
    }
    report["layers"] = layer_state
    _console("reconciled " + ", ".join(f"{k}={v}" for k, v in sorted(report.items()) if k != "layers"))
    return report
```

### 4.2 `ComfyUI-HSWQ-Loader-and-Tools/__init__.py` (added block, verbatim)

```python
# Distorch purge gate reconciliation API (patches/hswq_purge_rearm.py).
# Pure function module - installs nothing, patches nothing. DisTorchPurgeVRAMV2
# (HSWQ toggle) calls hswq_purge_rearm_state() AFTER it peels the ZI / Krea2 /
# SDXL-product / INT8 overlays for an SDXL handoff. The peel side knows the
# wrappers, not the module-level apply gates (_PATCHES_APPLIED / _APPLIED /
# _GPU_BAKE_INSTALLED / _PARITY_APPLIED). If the NEXT prompt reloads Z Image
# ConvRot NVFP4, Qwen/ControlNet ConvRot INT8, Krea2 or SDXL in the same
# process, a stale gate could skip re-application and sample on a half-peeled
# stack (alternating good/bad prompts, 2026-10-07). hswq_purge_rearm_state()
# inspects live wrapper stamps and resets ONLY the gate whose family layer is
# actually missing (branch separation: each family has an independent
# predicate and its own global), so the next load re-applies via the
# legitimate install path. Safe to call repeatedly on a healthy stack.
try:
    from .patches import hswq_purge_rearm as _hswq_purge_rearm  # noqa: F401
    logger.info("HSWQ purge-rearm reconciliation API registered")
except Exception:
    logger.exception("HSWQ purge-rearm module import failed")
```

### 4.3 `ComfyUI-DistorchMemoryManager/nodes/purge_vram.py` — Method 2d (added block, verbatim; also synced to root `purge_vram.py`)

```python
                # Method 2d - HSWQ purge-aware gate reconciliation (2026-10-07).
                # This purge peels HSWQ overlays (ZI bake/parity, Linear ver=8
                # wraps, INT8 protect load arm, Krea2 stack, SDXL product TC)
                # for an SDXL handoff, but HSWQ's apply_* gates are module-level
                # booleans the peel does not know about. When the NEXT prompt
                # reloads Z Image ConvRot NVFP4 / Qwen+ControlNet ConvRot INT8 /
                # Krea2 / SDXL in the SAME process, a stale gate=True makes the
                # load path skip re-application and sampling runs on a
                # half-peeled stack (alternating good/corrupt prompts).
                # hswq_purge_rearm_state() inspects the live wrapper stamps and
                # resets ONLY each family gate whose layer is actually missing
                # (branch separation: ZI / SDXL-product / Krea2 / INT8 each has
                # an independent predicate and its own global). It never wraps
                # anything and never touches tensors. Safe no-op when HSWQ's
                # reconciliation module is absent (older HSWQ) — purge behavior
                # then equals the previous version exactly.
                print("HSWQ INT8/NVFP4: Method 2d - Reconcile HSWQ apply gates after peel...")
                def _hswq_find_rearm_fn():
                    # Plain getattr only: HSWQ reconciliation module is registered
                    # by its real dotted name; no dir(), no __getattr__ probing
                    # (kornia/basicsr LazyLoader hazard, same reason as above).
                    for _rn, _rmod in _sys_modules():
                        if _rmod is None:
                            continue
                        _rns = str(_rn)
                        if not _rns.endswith(".hswq_purge_rearm") and "hswq_purge_rearm" not in _rns:
                            continue
                        try:
                            fn = getattr(_rmod, "hswq_purge_rearm_state", None)
                        except Exception:
                            continue
                        if callable(fn):
                            return fn
                    return None

                _hswq_rearm_fn = _hswq_find_rearm_fn()
                if _hswq_rearm_fn is None:
                    print("HSWQ INT8/NVFP4: Method 2d skipped (no hswq_purge_rearm; HSWQ pre-2.6?)")
                else:
                    try:
                        _rearm_report = _hswq_rearm_fn()
                        print(f"HSWQ INT8/NVFP4: Method 2d reconcile -> {_rearm_report}")
                    except Exception as e_rearm:
                        print(f"HSWQ INT8/NVFP4: Method 2d reconcile failed: {e_rearm}")
```

### 4.4 changelog EN entry (v2.4.8)

```markdown
* **v2.4.8** – DisTorchPurgeVRAMV2: added **Method 2d** HSWQ gate reconciliation after the HSWQ purge peel. The peel (ZI NVFP4 bake / parity wraps, Linear LoRA-bake wraps, INT8 protect load arm, Krea2 stack, SDXL product TC) was written for the Z Image → SDXL handoff and left HSWQ module-level apply gates (`_PATCHES_APPLIED` / `_PARITY_APPLIED` / `_GPU_BAKE_INSTALLED`) untouched: when the next prompt reloaded Z Image ConvRot NVFP4 / Qwen + ControlNet ConvRot INT8 / Krea2 in the same process, the stale gates early-returned without re-applying the stack and sampling ran on a half-peeled stack (alternating corrupt/good prompts). Method 2d now calls `hswq_purge_rearm_state()` (`ComfyUI-HSWQ-Loader-and-Tools` `patches/hswq_purge_rearm.py`, commit `409385f`), which inspects live wrapper stamps and resets ONLY the gate of each family whose layer is actually missing (Z Image / SDXL product / Krea2 / INT8 — independent predicates, branch-separated); it never wraps, unwraps, or touches tensors. Older HSWQ without the module → single skip line, exact previous behavior. Verified 4 consecutive post-purge generations in one session, all correct with `NVFP4_LORA_BAKE_OK INT8_PROTECT_LORA_BAKE_OK`. Root and `nodes/purge_vram.py` synced. See [Release Notes v2.4.8](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.4.8) for details.
```

---

## 5. Design rules honored (Owner philosophy: separate, branch, never mix)

1. **Branch separation.** One independent predicate per family (ZI / SDXL product /
   Krea2 / INT8 / parity / bake), each resets **only its own global**. Identifying
   markers are explicit stamps (`_hswq_nvfp4_product_tc`, `_hswq_krea2_stack`,
   `_hswq_int8_decode_patched`, …) — never shape-based inference, never shared
   "common processing".
2. **No fallback escape.** fast-disk stays enabled; the purge stays nuclear; nothing
   is disabled. The peel is kept and completed by reconciliation.
3. **No side effects.** The API neither wraps nor unwraps anything, touches no
   tensors, and never attaches filters to the root logger (the aborted `bdfcfcb`
   route, reverted by `a5134b7`).
4. **Backward compatible.** Older HSWQ without the module → single skip log line,
   behavior byte-for-byte the old purge.
5. **Proven, not guessed.** Fix shipped only after: mock healthy/peel/idempotent
   unit tests, live 4-prompt verification with `Method 2d reconcile -> {'zimage_nvfp4':
   'RESET', 'int8_comfy_quant': 'RESET', ...}` after each purge, and full-run
   `NVFP4_LORA_BAKE_OK INT8_PROTECT_LORA_BAKE_OK` evidence lines on every generation.

## 6. Residual notes

- The benign 38-line `state dict on uninitialized op` warning storm is **back**
  (pure 9e72c30 loader after the revert). Removing it must use a zero-side-effect
  method (namespace-local shim, never the root logger) and be re-validated with the
  same multi-prompt protocol.
- The permanent `Dynamic.load` guard (`d97bb5b`) and `_load_wrap_ok` (``fdc60bc``)
  remain as-is; Method 2d supersedes them for gate correctness but they still
  protect the bake hook itself.
