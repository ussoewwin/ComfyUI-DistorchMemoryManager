# XPU (Intel GPU) and ROCm (AMD GPU) support — implementation plan

Status: plan only. Nothing in this document has been implemented; the code tree is unchanged.

Related request: issue #4, *feat: xpu support please*. The AMD/ROCm arm was added on the owner's
instruction; §1.2 states why the two arms are not the same work, and §4.6 is the AMD design.

---

## 1. Goal and hard constraints

**Goal.** Make the three nodes of this pack usable on a PyTorch XPU build: `Memory Manager`,
`General Purge VRAM V2`, and `Patch Accel Attention DM` must not fail on an Intel GPU, and the
parts that have an XPU equivalent must work there.

**Scope, two arms.** Arm 1 is Intel XPU, which needs a separate device namespace (§4.1-§4.5).
Arm 2 is AMD ROCm, which shares the `torch.cuda` namespace and therefore needs no device
abstraction — only dependency gating, an AMD-correct arch label, and explicit skip lines (§4.6).
Both arms are additive and gated, and neither may change the CUDA path by one byte.

**Hard constraints, in priority order:**

1. **The CUDA path changes by zero bytes.** Every existing `torch.cuda.*` line keeps its text,
   its indentation and its order. XPU support is added as a *guard in front of* the existing
   code, never by rewriting it.
2. **Absolute separation.** XPU logic lives in its own module and its own branch. No shared
   helper is modified in a way that a CUDA run can observe. A function whose behaviour a CUDA
   run can reach must return the exact value it returned before.
3. **Explicit capability gate.** A missing capability produces a named log line and a skip, not
   a silent fallback and not a `None` that fails later.
4. **No unverified claim.** Real XPU hardware is not available on this machine. The release must
   state the code path is unverified on hardware; the tests below only prove structure,
   separation and CUDA non-regression.

### 1.1 Reference precedents

Two existing implementations in this environment were measured. They sit on **different layers**
(see §1.2), so each is cited for the layer it actually covers. Neither is copied; both are cited
because they kept their working backend untouched while adding the other one.

| precedent | file | shape |
| :--- | :--- | :--- |
| `ComfyUI-NunchakuFluxLoraStacker` (AMD / ROCm) | that repository's `md/ROCM_NUNCHAKU_IMPORT_REGISTRATION_FIX_EXPLANATION.md` (verified present in the other repository, not in this one) | capability flag `_NUNCHAKU_AVAILABLE` set by a narrow guard, dependent nodes registered only when true, `RuntimeError`-raising stub instead of a `None` callable, `[ROCm]` log lines scoped to the dependent feature only |
| `comfyui-multigpu` 2.6.4 (Intel / XPU) | that package's `device_utils.py` (installed as a custom node) | `hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available()` guard, per-device strings (`cuda:N` / `xpu:N`), prefix dispatch in the cache-clear loop, delegation to ComfyUI's device-agnostic API, one-shot device-list cache |

**Measured scope of the MultiGPU XPU work:** `rg -c -i xpu` over that package returns 12 hits,
all in `device_utils.py` — device enumeration (`:47-50`), accelerator-availability probe
(`:106-107`), and cache clearing (`:193-199`). `__init__.py` has zero XPU references. So the
upstream "comprehensive xpu support" is, in code, **enumeration plus cache clear**; there is no
XPU host-pin release, no NVML equivalent and no XPU attention kernel there either. That matches
what is achievable in this pack, which is why the plan below claims no more than it.

### 1.2 AMD/ROCm and Intel/XPU are not the same layer

They are different problems, and the difference is measurable in torch and in ComfyUI themselves.

| | AMD / ROCm | Intel / XPU |
| :--- | :--- | :--- |
| torch namespace | **the same `torch.cuda`** | a separate `torch.xpu` |
| device string | `cuda:N` | `xpu:N` |
| discriminator | `torch.version.hip` is set (`torch.version.cuda` is `None`) | `torch.version.xpu` is set |
| device count | `_device_count_amdsmi()` on the ROCm path | `torch.xpu.device_count()` |
| `ipc_collect` | exists (`torch.cuda.ipc_collect`, `:1374-1384`) | absent |
| `cudart` | `_cudart` is hipified into the build (unverified here) | absent |

Measured evidence:

- `torch/cuda/__init__.py:246-249` — *"Check for ROCm, if true return true"*, gated on
  `torch.version.hip`. The same module carries the ROCm branches at `:278`, `:1016`, `:1200`,
  and `device_count()` at `:1318-1319` selects `_device_count_amdsmi()` when `torch.version.hip`.
- `comfy/model_management.py:199-204` — verbatim: *"NVIDIA + AMD/ROCm both expose their GPUs
  through torch.cuda.*"*, and both arms build `torch.device("cuda", i)`.
- `comfy/model_management.py:402-414` — `is_nvidia()` tests `torch.version.cuda`, `is_amd()` tests
  `torch.version.hip`, while `is_intel_xpu()` (→ `:205-207`) is a **separate** `elif` arm using
  `torch.xpu`.

Consequences for this pack:

| question | AMD / ROCm | Intel / XPU |
| :--- | :--- | :--- |
| does the existing `torch.cuda.*` code run? | yes, unchanged | no, those branches are skipped |
| is a new device abstraction needed? | **no** | **yes** (§4) |
| is a capability gate needed? | yes, but only for CUDA-only *dependencies* (nunchaku, sageattention kernels, `comfy_kitchen`, NVML) | yes, for both dependencies and device APIs |
| does this plan's XPU branch ever trigger on AMD? | no — `is_xpu()` is false there, so AMD keeps the CUDA path byte-for-byte | — |

The AMD precedent in `ComfyUI-NunchakuFluxLoraStacker` is therefore **not** a device-plumbing
model. What it actually did, per its own record, was: a narrow import guard so the pack still
loads when `nunchaku` cannot import on ROCm, a `_NUNCHAKU_AVAILABLE` registration gate, a
raising stub instead of a `None` callable, and `[ROCm]`-prefixed messages scoped to that feature.
That is the **degradation** layer. The MultiGPU XPU work is the **device** layer. This plan needs
both, and they are kept separate: §4.1 (device backend) and §4.5 (capability skips).

Two AMD-side facts this plan does **not** claim, because they cannot be measured on this machine:
whether `torch.cuda.cudart().cudaHostUnregister()` is functional on a ROCm build (the symbol is
hipified upstream, unverified here — the call already sits inside a per-pointer `try`, so a
failure is counted as skipped rather than raised), and what
`torch.cuda.get_device_capability()` returns on a ROCm build (it feeds only the `smXY` log string
and the sage kernel choice in `nodes/sa.py`).

One defect is shared by both targets and is fixed by this plan for both: the startup path runs
`pip install -U nvidia-ml-py` before any backend check (`__init__.py:150`), and NVML is
NVIDIA-only. On AMD and on XPU that upgrade is pointless network work.

---

## 2. Current coupling (measured)

### 2.1 Where the shipping code actually lives

This repository ships **two copies** of each node module, and the import order in
`__init__.py` prefers the package copy:

| file | bytes | sha256 (first 12) | imported by |
| :--- | ---: | :--- | :--- |
| `nodes/purge_vram.py` | 283,733 | `86393456BF07` | `__init__.py:230` (`from .nodes.purge_vram import …`) |
| `purge_vram.py` (repo root) | 283,733 | `86393456BF07` | `__init__.py:234` (`from .purge_vram import …`), fallback only |
| `nodes/memory_manager.py` | 7,034 | `C7BF13E8366C` | `__init__.py:204` |
| `memory_manager.py` (repo root) | 7,034 | `C7BF13E8366C` | `__init__.py:208`, fallback only |
| `nodes/sa.py` | 26,530 | — | imported by `nodes/purge_vram.py` |
| `__init__.py` | — | — | ComfyUI entry point |
| `install.py` | — | — | ComfyUI-Manager install hook |

**The root copies are byte-identical to the `nodes/` copies today** (measured sha256 equal) and
are both tracked by git. The v2.5.3 purge work already established the rule "edit both, keep
them byte-identical". This plan keeps that rule: **every edit to `nodes/purge_vram.py` and
`nodes/memory_manager.py` must be mirrored byte-for-byte into the root copies**, and the
acceptance check in §6 compares the two hashes.

`install.py` was not in the original scope of this plan and is listed here only so the file
inventory is complete: it runs `pip install -U nvidia-ml-py` (NVIDIA-only) and touches no torch
device API. It is **not modified** by this work. Consequence: the ComfyUI-Manager install hook
still upgrades `nvidia-ml-py` on an AMD or XPU machine. That is a harmless refresh of a binding
the startup path no longer uses; it is recorded here so the behaviour is not mistaken for a
bug.

### 2.2 `torch.cuda` occurrences

| file | occurrences | clusters |
| :--- | ---: | --- |
| `nodes/purge_vram.py` | 41 | 11 |
| `nodes/memory_manager.py` | 15 | 6 |
| `nodes/sa.py` | 2 | 1 |
| `__init__.py` | 2 | 1 |

(The root copies carry the same counts, since they are byte-identical.)

APIs used: `is_available`, `current_device`, `set_device`, `device`, `device_count`,
`empty_cache`, `synchronize`, `ipc_collect`, `mem_get_info`, `memory_allocated`,
`memory_reserved`, `get_device_capability`, `cudart`.

Non-torch CUDA dependencies:

| dependency | location | purpose |
| :--- | :--- | :--- |
| `torch.cuda.cudart().cudaHostUnregister` | `nodes/purge_vram.py:3098` | release ComfyUI-tracked host pins |
| `pynvml` / NVML | `__init__.py:153`, `:157`, `:163`, `:175`; `nodes/memory_manager.py:8-20` (module-level probe) | read system-wide VRAM usage for the startup patch |
| `torch._C._cuda_emptyCache` | `nodes/purge_vram.py:2266-2267` | extra cache flush inside the Nunchaku path (inside a `try`, so it already degrades) |

### 2.3 Cluster map — `nodes/purge_vram.py`

| cluster | lines | what it does |
| :--- | :--- | :--- |
| A | 190-201 | `purge_cache`: iterate devices, `empty_cache`, `ipc_collect`, restore device |
| B | 250-252 | `clear_model_patches` tail: `empty_cache` + `synchronize` |
| C | 501-507 | `purge_models` hard free (`unload_all_models` then `free_memory(1e30, torch.device(f"cuda:{di}"))` per device) |
| D | 986-1105 | Qwen3-VL: gc sweep for model instances and its nested handlers; the cache clear is cluster E |
| E | 1106-1114 | Qwen3-VL tail: per-device `empty_cache` / `ipc_collect` / `synchronize` |
| F | 2006 | Nunchaku Method 3: gc sweep for model instances |
| G | 2254-2270 | Nunchaku: per-device cache clear, run twice, plus `torch._C._cuda_emptyCache` |
| H | 2314-2325 | HSWQ memory diag: allocated / reserved / `mem_get_info` |
| I | 3096-3121 | host-pin unregister via `cudart` (per-pointer `try`, so a failure is already swallowed) |
| J | 3496-3499 | HSWQ: `free_memory(1e30, torch.device(f"cuda:{di}"))` per device |
| K | 3676-3691 | HSWQ tail: per-device cache clear, `else:` branch already prints "CUDA not available, skipped CUDA cache clear" |

### 2.4 Cluster map — `nodes/memory_manager.py`

| cluster | lines | what it does |
| :--- | :--- | :--- |
| L | 23-43 | `get_non_torch_vram_usage_bytes()` — NVML measurement; already returns `None` when `torch.cuda.is_available()` is false (line 32-33) |
| M | 65-67 | `clear_memory()` helper: `empty_cache` + `ipc_collect` |
| N | 91-93 | `clean_gpu` path of `MemoryManager` |
| O | 110-115 | `free_memory(1e30, torch.device(f"cuda:{di}"))` loop |
| P | 159-161 | `clean_gpu` path of `SafeMemoryManager` |
| Q | 173-178 | second `free_memory` loop |

### 2.5 Remaining files

| file | lines | what it does |
| :--- | :--- | :--- |
| `nodes/sa.py` | 33-38 | `_current_cuda_arch()` — `get_device_capability()` to build the `smXY` string |
| `nodes/sa.py` | 41-48 | `_cuda_version_at_least()` — `torch.version.cuda` comparison |
| `nodes/sa.py` | 18 | `sageattn_modes` list: the menu still advertises CUDA-only kernels |
| `nodes/sa.py` | 304, 311-348 | `from sageattention import …` imports; the package is CUDA-only |
| `__init__.py` | 163, 175 | startup patch: skip when CUDA is absent, then measure |

### 2.6 Two facts the original plan missed

1. **`nodes/purge_vram.py:2266-2267` calls `torch._C._cuda_emptyCache`.** Not a `torch.cuda.*`
   attribute, so it does not appear in the occurrence count, but it is CUDA-only and already sits
   inside a `try/except`, so it needs no change: on a build without that symbol the `hasattr`
   guard skips it, and on a build that has it the call is already inside `try`.
2. **The root copies of `purge_vram.py` and `memory_manager.py` are tracked in git and are
   byte-identical to the `nodes/` copies.** Any change to one without the other breaks the
   established invariant. See §2.1.

---

## 3. XPU capability matrix (measured on torch 2.15.0+cu134, this machine)

The first measurement in this session recorded `hasattr` results and was read as if they were
call results. The corrected table below is the **call result on a CUDA build**:

| symbol | `hasattr` | call on `cu134` build | consequence |
| :--- | :--- | :--- | :--- |
| `torch.xpu` module | yes | — | presence proves nothing |
| `torch.xpu.is_available()` | yes | **`False`** | usable as the gate |
| `torch.xpu.device_count()` | yes | **`0`** | usable as the gate |
| `torch.xpu.get_device_name(0)` | yes | `AssertionError: Torch not compiled with XPU enabled` | must not be probed |
| `torch.xpu.current_device()` | yes | same `AssertionError` | must not be probed |
| `torch.xpu.empty_cache()` | yes | **returns normally, no exception** | safe but does nothing |
| `torch.xpu.synchronize()` | yes | **`AssertionError: Torch not compiled with XPU enabled`** | **must never be reached without the gate** |
| `torch.xpu.device(idx)` | yes | — | context-manager form exists |
| `torch.xpu.memory_stats` / `get_device_properties` / `mem_get_info` | yes | — | used by ComfyUI's own XPU branch |
| `torch.xpu.ipc_collect` | **no** | — | capability absent |
| `torch.cuda.is_available()` | — | `True` | CUDA is the active backend here |
| `torch.cuda.cudart()` | yes | returns `torch._C._cudart` | CUDA-only |

**Consequence for the design:** `hasattr` alone is not a gate. `torch.xpu.synchronize()` on a
CUDA-only build raises `AssertionError`, so a `hasattr`-only branch would crash a CUDA machine
the moment it was entered. The gate must be the **call result** of `is_available()`, wrapped so
that an exception counts as "not available" — and the arbitration must be **ComfyUI's, not this
pack's**: `get_torch_device()` tests `is_intel_xpu()` *before* the CUDA fallback
(`comfy/model_management.py:186-193`), and `is_intel_xpu()` is true whenever `torch.xpu.is_available()`
is true and ComfyUI is in GPU state (`:110-114`, `:145-151`). So on a dual-backend build ComfyUI
itself selects XPU, and a gate that forced CUDA there would purge the device the run does not
use. The gate mirrors ComfyUI instead of inventing its own order.

On a genuine XPU build the same table inverts: `is_available()` is the authority, `ipc_collect`
and `cudart` do not exist on `torch.xpu`, and `get_device_capability` has no SM meaning.

### 3.1 What ComfyUI 0.46.0 already does

Measured in `comfy/model_management.py`:

| ComfyUI API | XPU behaviour |
| :--- | :--- |
| `soft_empty_cache()` | `torch.xpu.synchronize()` + `torch.xpu.empty_cache()` when `is_intel_xpu()` |
| `get_torch_device()` | returns `torch.device("xpu", torch.xpu.current_device())` when `is_intel_xpu()` |
| `get_free_memory(dev)` | `torch.xpu.memory_stats(dev)` + `get_device_properties(dev).total_memory` |
| `unload_all_models()` | iterates `get_all_torch_devices()` |
| `is_intel_xpu()` | `cpu_state == GPU and xpu_available`, where `xpu_available = torch.xpu.is_available()` |
| `free_memory(memory_required, device)` | device-agnostic; unloads models whose `.device == device` |

Consequences for this pack:

- The device-agnostic API already exists upstream, so this pack should **delegate** instead of
  calling a device backend directly. `mm.free_memory(1e30, torch.device("xpu:0"))` is the
  correct XPU form of the existing `free_memory(1e30, torch.device("cuda:0"))`.
- **`is_intel_xpu()` is the authoritative signal, not `torch.xpu.is_available()`.** It also
  requires `cpu_state == CPUState.GPU`, i.e. ComfyUI is actually running on the GPU. A machine
  where torch has an XPU build but ComfyUI is in CPU mode must **not** take the XPU branch.
  The gate therefore reads ComfyUI's flag first and falls back to the torch-level probe only
  when ComfyUI cannot be imported.
- `is_intel_xpu()` uses `xpu_available` set at ComfyUI import time; it does not re-probe.
- The AMD side of this matrix is §4.6: on ROCm the same `torch.cuda` namespace applies, so the
  backend-specific facts there are the build flags (`torch.version.hip`) and two symbols
  (`_cudart`, `_cuda_ipc_collect`) whose presence on a ROCm build is **not verified here**.

---

## 4. Design

### 4.1 One new module, one gate

New file `nodes/device_backend.py` — stdlib + torch only, no ComfyUI import at module import
time.

**The gate (the single most important function in this change):**

```python
def is_xpu():
    # ComfyUI's own flag is authoritative: it also requires that ComfyUI is
    # actually running on the GPU (cpu_state == GPU), not merely that torch
    # has an XPU build.
    try:
        import comfy.model_management as mm
        if hasattr(mm, "is_intel_xpu"):
            return bool(mm.is_intel_xpu())
    except Exception:
        pass
    # Fallback for a standalone probe: the call result, never hasattr, and
    # the same priority order ComfyUI itself uses (XPU first, get_torch_device
    # :186-187) — no CUDA-wins rule of our own.
    if not hasattr(torch, "xpu"):
        return False
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False
```

`is_xpu()` returns `False` on this machine — provable by execution, which is the direct evidence
that the CUDA path is untouched (measured: `torch.xpu.is_available()` is `False` on this CUDA
build). The ComfyUI flag is consulted first and is authoritative; the torch-level probe is a
fallback for the case where ComfyUI cannot be imported at all (for example while a node module is
being imported during ComfyUI start-up). In that case the run state is unknown, so the call
result of `is_available()` is the best available signal — and on a CUDA-only build it is `False`,
so the CUDA path still wins there.

| function | returns |
| :--- | :--- |
| `is_nvidia_build()` | build flag `torch.version.cuda`; gates the NVML paths without changing a CUDA build in any run mode |
| `is_amd()` / `is_xpu()` | ComfyUI flag first (`mm.is_amd` / `mm.is_intel_xpu`, authoritative), else the torch probe on the call result; memoised only when the authoritative flag was used |
| `xpu_device_count()` | `torch.xpu.device_count()` inside `try`, `0` on exception |
| `xpu_current_device()` / `xpu_set_device(i)` | passthrough inside `try` |
| `xpu_empty_cache()` / `xpu_synchronize()` | passthrough inside `try`, `False` on exception |
| `xpu_mem_get_info(i)` | `torch.xpu.mem_get_info(i)` inside `try`, `(0, 0)` on exception |
| `xpu_memory_allocated(i)` / `xpu_memory_reserved(i)` | passthrough inside `try`, `0` on exception |
| `xpu_ipc_collect()` | returns `False` — capability absent on `torch.xpu` |
| `xpu_device_string(i)` | `"xpu:{i}"` — the XPU counterpart of the literal `f"cuda:{di}"` |
| `xpu_clear_cache_all_devices()` | synchronise + empty on every device; returns the number emptied and prints one line when a build cannot synchronise |
| `host_pin_supported()` | `torch.cuda.cudart()` present and exposing `cudaHostUnregister` |
| `unregister_host_pin(ptr)` | `True` when the unregister call succeeded, `False` when unsupported or refused |
| `capability_report()` | the exact booleans above, for one log line |

The module never references `torch.cuda` except inside `is_xpu()`'s arbitration, and it touches
`torch.xpu` only through `try`-wrapped calls. A CUDA run therefore cannot be affected by
importing it.

### 4.2 Guard-first, never rewrite

Every cluster keeps its current lines untouched. Two approved shapes:

| shape | use when | effect on existing bytes |
| :--- | :--- | :--- |
| **S1 early skip** | cluster is at the end of a function or branch | none — added lines sit before the existing block, no re-indent |
| **S2 dispatch** | cluster is mid-function | the block is moved verbatim into a new function; the call site is one line; `git diff -w` must show additions only |

S2 is allowed only with a diff that shows the moved block as unchanged text (same order, same
whitespace inside the block). Any hunk that deletes a `torch.cuda` line is a rejected diff.

**One shape is explicitly forbidden:** wrapping the existing CUDA block in `if not is_xpu():`.
That re-indents the CUDA statements, so `git diff` shows deletions. It is the shape a quick
implementation reaches for and it violates constraint 1.

### 4.3 Per-file plan

| file | clusters | shape | new code |
| :--- | :--- | :--- | :--- |
| `nodes/purge_vram.py` | A, B, E, G, K | S2 — each becomes `_clear_cache_all_devices()`, called once per site; the XPU branch clears `xpu` and returns before the CUDA statements; the AMD branch falls through to the unchanged CUDA statements | 1 helper + 5 one-line call sites |
| | C, J | S2 — the `free_memory` call moves into `_free_all_devices()`, which picks `f"cuda:{di}"` or `device_backend.device_string(di)`; on AMD the `cuda:` form is used | 1 helper |
| | D, F | **no change** — both are already inside `if torch.cuda.is_available():` (measured `:986`, `:2006`), so they are already skipped on a non-CUDA backend. A guard would be dead code | 0 lines |
| | H | S2 — `_mem_diag` prints an `xpu` line when `is_xpu()` | 1 branch inside `_mem_diag` |
| | I | S1 — the new `device_backend.unregister_host_pin()` returns `False` and the skipped count is reported, naming the backend | 1 branch + 1 counter |
| `nodes/memory_manager.py` | L | S2 — `get_non_torch_vram_usage_bytes()` (the real name; it is the only function at `:23-43`) keeps returning `None` and gains the log line only: on XPU with `[XPU]`, on AMD with `[AMD]`; the current code already returns `None` when CUDA is absent | 1 log branch |
| | M, N, P | S1/S2 | 3 guards |
| | O, Q | S2 — same `_free_all_devices()` helper as C/J | shared helper |
| `nodes/sa.py` | — | capability check only: a runtime package probe plus a call-site guard for the sage modes (§4.6 item 5); on XPU/AMD without the package the mode is logged unavailable and the configured backend kept; `sageattn_modes` is not changed (the menu text is documentation, and the call site is where the decision belongs) | 1 probe helper + 1 call-site guard, no kernel code touched |
| `__init__.py` | — | startup patch: one build-flag guard (`torch.version.cuda` falsy) placed before the pip upgrade and the NVML import, printing one `[AMD]`/`[XPU]` line and returning; a CUDA build runs today's code unchanged in every run mode | 2 guards |

**Mirror rule.** Every edit to `nodes/purge_vram.py` and `nodes/memory_manager.py` is applied
byte-for-byte to the root `purge_vram.py` and `memory_manager.py`, and the sha256 of each pair
is compared afterwards. `nodes/sa.py` has no root twin and is edited once. The shared module is
imported through a form that is identical in both copies — see §4.7.

No file outside this list is touched. `comfy/*` is never modified. `install.py` is not modified.

### 4.4 What MultiGPU does that this plan adopts, and what it does not

**Adopted (measured in `device_utils.py`):**

| pattern | MultiGPU line | use here |
| :--- | :--- | :--- |
| `hasattr` chain around the probe | `:47`, `:106`, `:194` | yes, extended with the call result and the ComfyUI-flag arbitration |
| per-device strings and prefix dispatch | `:171-199` | yes, in the cache-clear helper |
| one-shot device-list cache | `:10`, `:30-31`, `:95` | yes, for the XPU device count |
| delegate to ComfyUI's device-agnostic API | `model_management_mgpu.py:59` `mm.get_free_memory(...)` | yes — `free_memory`, `soft_empty_cache`, `unload_all_models` |

**Not adopted, with reasons:**

| pattern | reason |
| :--- | :--- |
| `torch.xpu.empty_cache()` only, no `synchronize()` | ComfyUI's own `soft_empty_cache` synchronises first on XPU; skipping it leaves queued work running against a "freed" allocator. This plan synchronises, inside `try`. |
| enumerate every backend into one list | MultiGPU needs all devices for placement. This pack purges the **active** backend; mixing backends would purge a device the run does not use. |
| XPU chosen on a dual-backend build | not a defect but the measured ComfyUI behaviour: `get_torch_device()` returns XPU first (`:186-187`), so the gate follows ComfyUI and purges the device the run actually uses. A gate that forced CUDA here would clear the wrong device. |
| `intel_extension_for_pytorch` import probe | That package is legacy for XPU in current torch; probing it adds a dependency-shaped branch for no capability gain. |

### 4.5 What is not implemented on XPU, and why

| feature | XPU status | reason | behaviour |
| :--- | :--- | :--- | :--- |
| host-pin unregister (`cudart`) | skipped | no XPU host-registration API in torch | one `[XPU]` line with the skipped pointer count |
| `ipc_collect` | skipped | API absent on `torch.xpu` | `empty_cache` + `synchronize` still run |
| `torch._C._cuda_emptyCache` | skipped | CUDA-only symbol | already inside a `try`, never reached |
| startup non-torch VRAM measurement | skipped | `pynvml` is NVIDIA-only; `torch.xpu.mem_get_info` reports this process only | explicit skip line; no silent wrong value |
| HSWQ nuclear (comfy_kitchen workspaces, aimdo host buffers, CUDA graphs) | out of scope | the subsystem is CUDA-only | the HSWQ toggle on XPU logs that the subsystem is unavailable and performs only the generic unload/cache path |
| `Patch Accel Attention DM` kernels (`fa2`, `sageattn*`, comfy kitchen) | probe-driven | `fa2` and comfy-kitchen already degrade with a reason (`:541-549`, `:253-259`); sage modes gain a runtime package probe (§4.6 item 5) | a kernel that imports but fails at call time falls back per-call (`nodes/sa.py:426-441`) |
| `get_device_capability` (SM version) | not applicable | no SM concept, and the call raises on a CUDA build | version string prints `xpu` instead of `smXY`, and the probe is guarded |

No node advertises an XPU-only mode. The nodes work where the backend allows and name what they
skipped.

### 4.6 AMD/ROCm arm

**Gate.** `is_amd()` mirrors `is_xpu()` in §4.1: ComfyUI's own `mm.is_amd()` is authoritative when
ComfyUI is importable, because it additionally requires `cpu_state == CPUState.GPU`
(`comfy/model_management.py:409-414`); otherwise the fallback probe is
`bool(torch.version.hip)` **and** `torch.cuda.is_available()` — the call result, never
`hasattr`. `is_amd()` and `is_xpu()` are mutually exclusive by construction (hip is set for ROCm,
xpu for Intel), so no site can take both arms, and both are false on a CUDA build. Measured on
this machine: `torch.version.hip` is `None`.

**What needs no change.** Every CUDA call this pack makes is in the namespace ROCm provides, so
the existing lines run as they are: `is_available`, `device_count`, `current_device`,
`set_device`, `device`, `empty_cache`, `synchronize`, `ipc_collect`, `mem_get_info`,
`memory_allocated`, `memory_reserved`. The device string stays `cuda:N`, so the
`free_memory(1e30, torch.device(f"cuda:{di}"))` sites are already correct on AMD. ComfyUI
confirms the same arrangement for its own code (`comfy/model_management.py:199-204`).

**The seven AMD items to implement.**

| # | site | today | change |
| :--- | :--- | :--- | :--- |
| 1 | `__init__.py:150` | `_ensure_nvidia_ml_py_latest()` runs `pip install -U nvidia-ml-py` before any backend check, on every startup | skip when the build is not an NVIDIA build, tested with the **build flag** `torch.version.cuda` (not `mm.is_nvidia()`, which also requires `cpu_state == GPU` and would therefore change the behaviour of a CUDA build running in CPU mode); print one `[AMD]`/`[XPU]` line instead |
| 2 | `__init__.py:153-164` | `import pynvml` then `nvmlInit()`; on AMD the init fails and prints a `WARNING` | short-circuit on the same **build-flag** condition as item 1 (`torch.version.cuda` falsy), so a CUDA build keeps today's behaviour in every run mode and AMD/XPU never reach the NVML import; one `[AMD]`/`[XPU]` line names NVML as NVIDIA-only |
| 3 | `nodes/memory_manager.py:8-20` and `:23-43` | the module-level probe calls `pynvml.nvmlInit()` at import and prints *"pynvml imported but NVML init failed"* on AMD; `get_non_torch_vram_usage_bytes()` then returns `None` silently (`:30-31`) | do not call `nvmlInit()` on a non-NVIDIA build, and print one `[AMD]`/`[XPU]` line; the function keeps returning `None` |
| 4 | `nodes/sa.py:33-38` | `_current_cuda_arch()` returns `sm{major}{minor}`; on ROCm `get_device_capability()` is `(prop.major, prop.minor)` (`torch/cuda/__init__.py:859-873`), so AMD would print a bogus `smXY` | AMD-only branch returns `torch.cuda.get_device_properties().gcnArchName`, the field torch itself reads under `if torch.version.hip` (`:278-279`); the CUDA line is untouched |
| 5 | `nodes/sa.py:551` | every sage mode funnels into `get_sage_func_dm()`, whose body imports `from sageattention import …` (`:304`, `:311-348`) without a guard; a machine without the package raises `ImportError` out of the callback and the node errors, while `fa2` already has an unavailable path (`:541-549`) | strictly additive: a new probe helper plus a guard **before** the existing call site, mirroring the fa2 unavailable branch (log the reason, `_restore_previous(...)`, early `return` — an early return is already used in that callback at `:536-537` and `:548-549`); `get_sage_func_dm()` and its import lines are untouched. Two measured limits are kept as they are: a kernel *import* failure inside the resolver body (`:317`, `:323`, `:327`, `:331`, `:335`, `:346`) still raises and errors the node, and a kernel that imports but fails at *call* time already falls back per-call (`:426-441`) |
| 6 | `nodes/purge_vram.py:3096-3121` | `torch.cuda.cudart()` is imported as `_cudart` with `except ImportError: _cudart = None` (`torch/cuda/__init__.py:47-50`), so on a build without it the call would raise into the per-pointer `try` and be silently swallowed | `unregister_host_pin()` returns `False` when `cudart()` is `None` or lacks the symbol, counts it as skipped, and names the backend; this is the same helper the XPU arm uses |
| 7 | `nodes/purge_vram.py:1112`, `:2260`; `nodes/memory_manager.py:67` | `torch.cuda.ipc_collect()` is called **without** a local guard at these three sites, unlike `:196-199` and `:3681-3684` which already wrap it in `try/except` | no change to the CUDA statements (wrapping them would re-indent them). If a ROCm build lacks `torch._C._cuda_ipc_collect`, the two vendor sites fall into their existing outer handler, print the error, and the remainder of that vendor block is skipped while the node itself continues; `clear_memory()` has no in-pack caller (measured: the only occurrence of the name in the pack is its own definition) |

**On item 6, what must not change.** The existing code pops every pointer from
`mm.PINNED_MEMORY` and sets `mm.TOTAL_PINNED_MEMORY = 0` (`:3108-3115`) regardless of whether the
unregister succeeded. That bookkeeping is not altered by this work. On a build without `cudart`
there is also nothing to release: ComfyUI registers those pins with the same symbol
(`comfy/pinned_memory.py:69`), so `PINNED_MEMORY` stays empty.

**Not implemented on AMD.** System-wide VRAM measurement (the NVML equivalent) is skipped. The
AMD counterpart would be `amdsmi` — torch uses it on the ROCm path (`torch/cuda/__init__.py:1084-1095`,
`_HAS_AMDSMI` gate) — but reading it would add a dependency and cannot be verified on AMD
hardware here, so it is not used. The consequence is stated in the release: on AMD the startup
VRAM patch does nothing, exactly as on XPU.

**Already safe, no change.** `torch._C._cuda_emptyCache` (`nodes/purge_vram.py:2266-2267`) is
guarded by `hasattr` and wrapped in `try`; `comfy_kitchen` INT8/cache resets and the HSWQ nuclear
subsystem are inside `try` blocks that print a skip; `get_comfy_kitchen_func_dm()` already falls
back to PyTorch attention with a warning when the backend is not registered
(`nodes/sa.py:253-259`); and the sage path in `__init__.py:15-115` needs no change, because
ComfyUI's own `attention_sage` already catches the missing-symbol error and falls back
(`comfy/ldm/modules/attention.py:744-748`), and this pack's guard mirrors that fallback
(`__init__.py:67-87`).

### 4.7 How the shared module is imported under the byte-identity rule

`nodes/device_backend.py` is new, and both copies of `purge_vram.py` must reference it while
staying byte-identical. A plain relative import cannot do that: the root copy would need
`from .nodes.device_backend import …` and the `nodes/` copy `from .device_backend import …`,
which are different bytes.

The import is therefore written once, in a form that resolves in both locations, and copied
verbatim into both files:

```python
try:
    from . import device_backend as _db
except ImportError:
    from .nodes import device_backend as _db
```

Why it resolves: `nodes/purge_vram.py` is imported as `<pkg>.nodes.purge_vram`, so `.device_backend`
is `<pkg>.nodes.device_backend` and the first branch succeeds. The root `purge_vram.py` is imported
as `<pkg>.purge_vram`, so `.device_backend` is `<pkg>.device_backend`, which does not exist, and
the `except` branch reaches `<pkg>.nodes.device_backend`. Both statements are identical text, so
the mirror rule survives.

**Proof, executed:** a two-level package was built in a temporary directory with the same shape
(root `purge_vram.py`, `nodes/purge_vram.py`, `nodes/device_backend.py`), both copies written from
the same bytes, and both imported. Measured result: the two files had equal sha256, the root copy
resolved through the `except` branch and the `nodes/` copy through the first branch, and both
returned the value defined in `nodes/device_backend.py`. The temporary tree was removed
afterwards; nothing was written into this repository.

`__init__.py` is at the package root only, so it uses `from .nodes.device_backend import …`
directly. `nodes/sa.py` lives inside `nodes/`, so it uses `from .device_backend import …`
directly. Neither has a twin, so no dual form is needed there.

---

## 5. Behaviour matrix

| toggle / node | CUDA | AMD / ROCm | XPU |
| :--- | :--- | :--- | :--- |
| `purge_cache` | unchanged | unchanged (same `torch.cuda` calls, `cuda:N`) | `xpu.synchronize()` + `xpu.empty_cache()` per device |
| `purge_models` / `clear_model_patches` | unchanged | unchanged | unchanged (ComfyUI handles devices) |
| SeedVR2 / Qwen3-VL / Nunchaku purge | unchanged | unchanged | unchanged; final cache clear via the XPU path |
| `HSWQ` | unchanged | generic unload + cache clear; nuclear parts log a skip when their CUDA-only imports are absent | same as AMD |
| `Ollama` | unchanged | unchanged | unchanged (no torch device use) |
| `Memory Manager` `clean_gpu` | unchanged | unchanged | `xpu.synchronize()` + `xpu.empty_cache()` |
| `reset_virtual_memory` | unchanged | unchanged (`free_memory(1e30, cuda:N)`) | `free_memory(1e30, xpu:N)` per device |
| `Patch Accel Attention DM` | unchanged | fa2 / comfy-kitchen already degrade with a reason; sage modes gated by the runtime package probe (§4.6 item 5) | same as AMD |
| startup VRAM patch | unchanged | skipped, `[AMD]` line, `amdsmi` not used | skipped, `[XPU]` line |
| host-pin unregister | unchanged | attempted; counted skipped when `cudart()` is `None` | counted skipped |
| arch label in logs | `smXY` | `gcnArchName` | `xpu` |

---

## 6. Verification without non-CUDA hardware

Accepted premise: neither ROCm nor XPU execution can be tested here. What follows is the full
extent of what the release may claim.

| check | method | pass criterion |
| :--- | :--- | :--- |
| XPU gate is closed on this machine | call `device_backend.is_xpu()` on the CUDA build | `False`, and `capability_report()` shows `is_available=False`, `device_count=0` |
| AMD gate is closed on this machine | call `device_backend.is_amd()` on the CUDA build | `False` — `torch.version.hip` is `None` (measured) |
| the two arms cannot both fire | evaluate `is_amd() and is_xpu()` under both monkeypatches | always `False`; one arm at most |
| gate logic executed, not just read | run the §4.1 code on this machine with (a) no ComfyUI module, (b) a ComfyUI flag returning `True`, (c) one returning `False`, (d) one raising | measured: `False`, `True`, `False`, `False` — the fallback never overrides a `False` flag, and an exception counts as not-available |
| shared-module import resolves from both copies | the §4.7 two-level package experiment, both files byte-identical | both copies import the single `nodes/device_backend.py`; equal sha256 |
| CUDA path untouched | `git diff` of the change | every hunk is an addition; no deletion of a `torch.cuda` line; `git diff -w` shows no re-indent |
| root copies still identical | sha256 of `purge_vram.py` vs `nodes/purge_vram.py`, same for `memory_manager.py` | equal pairs |
| import safety | import each node module in a CUDA process | registered node count unchanged, `INPUT_TYPES` unchanged |
| XPU branch self-consistent | monkeypatch `is_xpu` to `True` plus a fake device object with `empty_cache` / `synchronize` / `mem_get_info` / `device_count` | each cluster calls the fake once, prints one `[XPU]` line, returns the same tuple as the CUDA path |
| AMD branch self-consistent | monkeypatch `is_amd` to `True` while the CUDA path is live | the existing `torch.cuda` calls are unchanged and still run, no `[XPU]` line appears, one `[AMD]` line per skipped capability, identical return tuple |
| ROCm without `_cuda_ipc_collect` | simulate a build whose `torch._C` has no `_cuda_ipc_collect` | the already-guarded sites (`:196-199`, `:3681-3684`) skip silently; the two unguarded vendor sites (`:1112`, `:2260`) fall into their existing outer handler and print an error without aborting the node |
| absent sageattention package | fake `sageattention` import that raises `ImportError` | the call-site guard logs the reason, restores the previous backend, and the node does not raise |
| kernel imports but fails at call time | fake `sage_func` raising at invocation | the per-call fallback at `nodes/sa.py:426-441` logs the error and returns PyTorch attention; the node continues |
| absent host-pin symbol | force `torch.cuda.cudart()` to `None` | `unregister_host_pin()` returns `False`, skip counter increments, no exception |
| missing capability path | fake device without `ipc_collect` and without host-pin API | skip counters increment, no exception |
| hostile host: absent capability throws | fake `synchronize` raising `AssertionError` while `is_xpu` stays `False` | no call is attempted, no exception reaches the user |
| CUDA regression | existing v2.5.3 measurements for `purge_cache`, drain, lock, `('*',)` return, `INPUT_TYPES` | identical values |
| static checks | `py_compile`, import in a process with no GPU | no error |

Not proven, and to be stated in the release: XPU execution on real Intel hardware, ROCm
execution on real AMD hardware, driver-specific timing on either, whether `cudart` host-pin
release works on a ROCm build, whether a ROCm build exposes `torch._C._cuda_ipc_collect`, and
which `sageattention` / `flash-attn` / `comfy_kitchen` kernels a given ROCm installation
provides.

---

## 7. Release and communication

- `README.md` and `zhmd/README.md`: one line under requirements — XPU and ROCm support are
  structural and unverified on hardware; CUDA behaviour is unchanged.
- Changelog entry: state the skipped capabilities per arm and that the CUDA path is untouched.
- Issue reply: name what works (generic purge, memory manager, device enumeration, per-device
  cache clear), what is skipped per arm (host pins, `ipc_collect`, NVML startup patch, HSWQ
  nuclear, accelerator backends), and that hardware verification was not possible in this
  environment.

---

## 8. Risk and rollback

| risk | mitigation |
| :--- | :--- |
| a `hasattr`-only branch crashes a CUDA machine via `torch.xpu.synchronize()` | measured `AssertionError`; the gate uses the call result and ComfyUI's `is_intel_xpu()` |
| a dual-backend build takes the XPU branch | that is ComfyUI's own selection (`get_torch_device()` prefers XPU, `:186-187`); mirroring it is what keeps the purge on the device the run uses |
| only the `nodes/` copy is edited and the root copy drifts | §6 hash comparison; the mirror rule in §2.1 and §4.3 |
| an existing CUDA statement is re-indented by an `if not is_xpu():` wrapper | §4.2 forbids that shape; `git diff -w` must show additions only |
| a guard is placed where `return` is illegal | per-cluster shape chosen from S1/S2; `py_compile` + import test catches it |
| silent degradation on XPU | every skipped capability prints a line containing `[XPU]` and the exact name |
| silent degradation on ROCm | every skipped capability prints a line containing `[AMD]` and the exact name |
| an AMD machine takes the XPU branch, or the reverse | impossible by construction: `is_amd()` keys on `torch.version.hip`, `is_xpu()` on `torch.version.xpu`, and the harness asserts `is_amd() and is_xpu()` is always `False` |
| the AMD arm changes CUDA behaviour by reaching a shared helper | the AMD arm adds log lines and one arch-string branch only; every shared helper returns the CUDA value when neither gate fires |
| regression on CUDA | additions-only diff rule plus the CUDA regression table |

Rollback is a single revert of one commit: both arms are additive, so reverting restores the
exact v2.5.3 tree.

---

## 9. Work breakdown

| step | output | estimate |
| :--- | :--- | ---: |
| 1 | `nodes/device_backend.py` (`is_amd`, `is_xpu`, `device_string`, `unregister_host_pin`, `capability_report`) | 1.5 h |
| 2 | `purge_vram.py` clusters A-K | 3 h |
| 3 | `memory_manager.py` clusters L-Q | 1.5 h |
| 4 | `sa.py` resolver gate + arch label + `__init__.py` startup skip (shared by both arms) | 1 h |
| 5 | fake-backend harness + CUDA regression run | 1.5 h |
| 6 | mirror the two root copies, including the §4.7 dual import form, + hash check | 0.5 h |
| 7 | README / changelog / release notes (EN + ZH) | 1 h |
| 8 | AMD arm: `[AMD]` skip lines, `cudart()` guard, `gcnArchName` path, the three `ipc_collect` site notes | 1.5 h |
| | **total** | **11.5 h** |

Acceptance: the diff contains additions only for the CUDA path, every cluster has a fake-backend
test, `is_xpu()` and `is_amd()` are both measured `False` on this machine, both root/`nodes`
copies stay byte-identical, and the release states the hardware verification gap for each arm.

---

## 10. Implementation record (measured)

Implemented on 2026-11-21 in this worktree. Not committed and not pushed at the time of
writing; the live custom-node copy has been synced.

| file | change |
| :--- | :--- |
| `nodes/device_backend.py` | new, 7,534 B (gates, `xpu_*` helpers, capability report) |
| `__init__.py` | +13 lines (startup guard before the pip upgrade and the NVML import) |
| `nodes/purge_vram.py` | +57 lines, mirrored byte-for-byte to `purge_vram.py` |
| `nodes/memory_manager.py` | +42 lines, mirrored byte-for-byte to `memory_manager.py` |
| `nodes/sa.py` | +44 lines (`gcnArchName` label, sage package probe, call-site guard) |

Diff audit: `git diff -w --numstat` shows additions only and zero deletions. The raw diff has
20 removed lines, all of them the re-indented NVML probe in the two `memory_manager.py` copies;
zero removed lines contain `torch.cuda` or `torch.xpu`.

Verification executed on this machine (torch 2.15.0+cu134, ComfyUI 0.46.0):

- gates: `is_xpu()` False, `is_amd()` False, `is_nvidia_build()` True, `host_pin_supported()` True,
  `xpu_is_available` False, `xpu_device_count` 0, `hip` None, `cuda` "13.4".
- `INPUT_TYPES` unchanged: purge 9 keys with `RETURN_TYPES` `('*',)`, `MemoryManager` 6 keys,
  `SafeMemoryManager` 4 keys.
- node level, gate closed: `MemoryManager` and `SafeMemoryManager` print the unchanged CUDA line
  and return `('x',)`.
- node level, gate forced: the XPU block runs (synchronise + empty per device, "GPU memory cleared
  (XPU)") and the node still returns `('x',)`.
- fake XPU with 2 devices: `xpu_clear_cache_all_devices()` 2, two empty calls, two sync calls,
  `xpu_device_string(1)` "xpu:1", `xpu_mem_get_info(0)` (1024, 2048).
- hostile fake whose `synchronize` raises `AssertionError`: `xpu_synchronize()` False, one warning
  line printed, no exception escapes.
- `py_compile` on the worktree and on the live copies: exit 0.
- mirror: `purge_vram.py` == `nodes/purge_vram.py` (`E9BE668CA9F4FBBA`), `memory_manager.py` ==
  `nodes/memory_manager.py` (`787E0BE6AE6EB667`).
- live sync: content equal to the worktree for all seven files (EOL normalised, the live tree is a
  CRLF checkout); no live file had uncommitted edits before the sync (content equalled `HEAD`).

Deviations from the plan above, each measured rather than assumed:

1. Cluster C is the `purge_models` hard free, not the SeedVR2 path (§2.3 label corrected).
2. Clusters D and F need no change: both already sit inside `if torch.cuda.is_available():`
   (`:986`, `:2006`), so they are already skipped on a non-CUDA backend and a guard would be dead
   code.
3. The helpers carry an `xpu_` prefix, and `is_nvidia_build()` / `host_pin_supported()` were added.
4. The shared import is `from . import device_backend as _db` — the same dual form, one name.
5. On a dual CUDA+XPU build both arms print, because the CUDA block cannot be re-gated without
   editing its condition. The extra clear targets a device the run is not using.