# Issue #3 Fix Documentation: Node Import Path Correction

## Overview

This document provides a complete explanation of the bug fix for [Issue #3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/issues/3), which resolved the problem where only one node (Model Patch Memory Cleaner) was visible in ComfyUI after updating, despite the repository containing four nodes.

## Problem Description

### Symptom
After updating the ComfyUI-DistorchMemoryManager custom node, users reported that only **one node** (Model Patch Memory Cleaner) was visible in ComfyUI's node palette, even though the repository contains **four nodes**:
1. Memory Manager
2. Safe Memory Manager
3. Purge VRAM V2 (DisTorchPurgeVRAMV2)
4. Patch Sage Attention DM

### Root Cause

The issue was caused by **incorrect import paths** in `__init__.py` after the project structure was refactored to move node files into a `nodes/` subdirectory.

#### Project Structure Change

**Before refactoring:**
```
ComfyUI-DistorchMemoryManager/
├── __init__.py
├── memory_manager.py
├── purge_vram.py
└── sa.py
```

**After refactoring:**
```
ComfyUI-DistorchMemoryManager/
├── __init__.py
└── nodes/
    ├── memory_manager.py
    ├── purge_vram.py
    └── sa.py
```

#### The Bug

After moving the node files to the `nodes/` directory, the import statements in `__init__.py` were **not updated** to reflect the new directory structure. The code was still trying to import from the root directory:

```python
# ❌ INCORRECT (Before Fix)
from .memory_manager import MemoryManager, SafeMemoryManager, any
from .purge_vram import DisTorchPurgeVRAMV2
from .sa import PatchSageAttentionDM
```

These imports failed silently because:
1. Python's relative import (`.memory_manager`) looks for `memory_manager.py` in the same directory as `__init__.py`
2. After refactoring, `memory_manager.py` is now in `nodes/memory_manager.py`
3. The import fails, but the code has fallback logic that sets the variables to `None`
4. When variables are `None`, the conditional registration checks fail
5. Only `ModelPatchMemoryCleaner` (defined directly in `__init__.py`) gets registered

## Error Analysis

### Import Failure Chain

1. **First Import Attempt (Relative Import)**
   ```python
   try:
       from .memory_manager import MemoryManager, SafeMemoryManager, any
   except ImportError:
   ```
   - Fails because `memory_manager.py` is not in the same directory as `__init__.py`
   - Raises `ImportError`

2. **Second Import Attempt (Absolute Import)**
   ```python
   try:
       from memory_manager import MemoryManager, SafeMemoryManager, any
   except ImportError:
   ```
   - Also fails because `memory_manager.py` is not in `sys.path` as a top-level module
   - Raises `ImportError`

3. **Fallback Assignment**
   ```python
   MemoryManager = None
   SafeMemoryManager = None
   ```
   - Variables are set to `None` to prevent `NameError` later

4. **Registration Check**
   ```python
   if MemoryManager is not None:
       NODE_CLASS_MAPPINGS["MemoryManager"] = MemoryManager
   ```
   - Since `MemoryManager is None`, the condition is `False`
   - The node is **not registered** in `NODE_CLASS_MAPPINGS`
   - ComfyUI never sees the node

### Why ModelPatchMemoryCleaner Still Worked

`ModelPatchMemoryCleaner` is defined **directly in `__init__.py`** (lines 40-124), not imported from another file. Therefore, it was always available and got registered regardless of import failures:

```python
NODE_CLASS_MAPPINGS = {
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,  # ✅ Always works
}
```

## Solution

### Fix Applied

The import paths were corrected to point to the `nodes/` subdirectory:

```python
# ✅ CORRECT (After Fix)
from .nodes.memory_manager import MemoryManager, SafeMemoryManager, any
from .nodes.purge_vram import DisTorchPurgeVRAMV2
from .nodes.sa import PatchSageAttentionDM
```

### Complete Code Changes

#### Change 1: Memory Manager Import

**Before:**
```python
# Import Memory Manager nodes (including any for ModelPatchMemoryCleaner)
try:
    from .memory_manager import MemoryManager, SafeMemoryManager, any
except ImportError:
    try:
        from memory_manager import MemoryManager, SafeMemoryManager, any
    except ImportError:
        MemoryManager = None
        SafeMemoryManager = None
```

**After:**
```python
# Import Memory Manager nodes (including any for ModelPatchMemoryCleaner)
try:
    from .nodes.memory_manager import MemoryManager, SafeMemoryManager, any
except ImportError:
    try:
        from nodes.memory_manager import MemoryManager, SafeMemoryManager, any
    except ImportError:
        MemoryManager = None
        SafeMemoryManager = None
```

**Explanation:**
- `.nodes.memory_manager` is a relative import that looks for `nodes/memory_manager.py` relative to `__init__.py`
- The fallback `nodes.memory_manager` (without dot) is an absolute import that works if `nodes` is in `sys.path`

#### Change 2: Purge VRAM V2 Import

**Before:**
```python
# Import Purge VRAM V2 node
try:
    from .purge_vram import DisTorchPurgeVRAMV2
except ImportError:
    try:
        from purge_vram import DisTorchPurgeVRAMV2
    except ImportError:
        DisTorchPurgeVRAMV2 = None
```

**After:**
```python
# Import Purge VRAM V2 node
try:
    from .nodes.purge_vram import DisTorchPurgeVRAMV2
except ImportError:
    try:
        from nodes.purge_vram import DisTorchPurgeVRAMV2
    except ImportError:
        DisTorchPurgeVRAMV2 = None
```

**Explanation:**
- Same pattern as Change 1, but for the `purge_vram.py` file
- Now correctly imports from `nodes/purge_vram.py`

#### Change 3: SageAttention Import

**Before:**
```python
# Import SageAttention patch node
try:
    from .sa import PatchSageAttentionDM
except ImportError:
    try:
        from sa import PatchSageAttentionDM
    except ImportError:
        PatchSageAttentionDM = None
```

**After:**
```python
# Import SageAttention patch node
try:
    from .nodes.sa import PatchSageAttentionDM
except ImportError:
    try:
        from nodes.sa import PatchSageAttentionDM
    except ImportError:
        PatchSageAttentionDM = None
```

**Explanation:**
- Same pattern as Changes 1 and 2, but for the `sa.py` file
- Now correctly imports from `nodes/sa.py`

## Technical Details

### Python Import Mechanism

#### Relative Imports (`.nodes.memory_manager`)

When using a relative import like `from .nodes.memory_manager import ...`:
1. Python starts from the directory containing the current module (`__init__.py`)
2. The `.` means "current package"
3. `.nodes` means "look for a `nodes` subdirectory"
4. `.nodes.memory_manager` means "look for `memory_manager.py` in the `nodes` subdirectory"

**Path Resolution:**
```
__init__.py location: ComfyUI-DistorchMemoryManager/__init__.py
Relative path: .nodes.memory_manager
Resolved path: ComfyUI-DistorchMemoryManager/nodes/memory_manager.py
```

#### Absolute Imports (`nodes.memory_manager`)

When using an absolute import like `from nodes.memory_manager import ...`:
1. Python searches in `sys.path` for a module named `nodes`
2. If `nodes` is found, it looks for `memory_manager.py` inside it
3. This works if the parent directory of `nodes` is in `sys.path`

**Path Resolution:**
```
sys.path entry: ComfyUI-DistorchMemoryManager/ (parent directory)
Absolute path: nodes.memory_manager
Resolved path: ComfyUI-DistorchMemoryManager/nodes/memory_manager.py
```

### Why Two-Level Try-Except?

The code uses a two-level try-except pattern:

```python
try:
    from .nodes.memory_manager import ...  # Relative import (preferred)
except ImportError:
    try:
        from nodes.memory_manager import ...  # Absolute import (fallback)
    except ImportError:
        # Set to None if both fail
```

**Reason:**
1. **First attempt (relative)**: Works when the package is properly installed and imported as a module
2. **Second attempt (absolute)**: Works in edge cases where the relative import fails but the absolute path is in `sys.path`
3. **Fallback**: Prevents `NameError` if both imports fail, allowing the code to continue executing

### Node Registration Mechanism

ComfyUI discovers custom nodes by:
1. Scanning `custom_nodes/` directories
2. Loading `__init__.py` files
3. Looking for `NODE_CLASS_MAPPINGS` dictionary
4. Registering all entries in the dictionary

**Registration Code:**
```python
# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,  # Always registered
}

# Register Memory Manager nodes if available
if MemoryManager is not None:  # ✅ Now True after fix
    NODE_CLASS_MAPPINGS["MemoryManager"] = MemoryManager
if SafeMemoryManager is not None:  # ✅ Now True after fix
    NODE_CLASS_MAPPINGS["SafeMemoryManager"] = SafeMemoryManager

# Register Purge VRAM V2 node if available
if DisTorchPurgeVRAMV2 is not None:  # ✅ Now True after fix
    NODE_CLASS_MAPPINGS["DisTorchPurgeVRAMV2"] = DisTorchPurgeVRAMV2

# Register SageAttention node if available
if PatchSageAttentionDM is not None:  # ✅ Now True after fix
    NODE_CLASS_MAPPINGS["PatchSageAttentionDM"] = PatchSageAttentionDM
```

**Before Fix:**
- `MemoryManager is None` → Not registered
- `SafeMemoryManager is None` → Not registered
- `DisTorchPurgeVRAMV2 is None` → Not registered
- `PatchSageAttentionDM is None` → Not registered
- Result: Only 1 node visible

**After Fix:**
- `MemoryManager is not None` → Registered ✅
- `SafeMemoryManager is not None` → Registered ✅
- `DisTorchPurgeVRAMV2 is not None` → Registered ✅
- `PatchSageAttentionDM is not None` → Registered ✅
- Result: All 4 nodes visible ✅

## Verification

### How to Verify the Fix

1. **Check Import Success:**
   ```python
   # In __init__.py, add debug prints (temporary):
   print(f"MemoryManager: {MemoryManager}")
   print(f"SafeMemoryManager: {SafeMemoryManager}")
   print(f"DisTorchPurgeVRAMV2: {DisTorchPurgeVRAMV2}")
   print(f"PatchSageAttentionDM: {PatchSageAttentionDM}")
   ```
   - Before fix: All print `None`
   - After fix: All print class objects

2. **Check Node Registration:**
   ```python
   # In __init__.py, add debug print (temporary):
   print(f"NODE_CLASS_MAPPINGS: {list(NODE_CLASS_MAPPINGS.keys())}")
   ```
   - Before fix: `['ModelPatchMemoryCleaner']`
   - After fix: `['ModelPatchMemoryCleaner', 'MemoryManager', 'SafeMemoryManager', 'DisTorchPurgeVRAMV2', 'PatchSageAttentionDM']`

3. **Check ComfyUI Node Palette:**
   - Open ComfyUI
   - Search for "Memory" category
   - Should see all 4 nodes listed

### Expected Behavior After Fix

- **All 4 nodes** appear in ComfyUI's node palette under the "Memory" category
- No import errors in ComfyUI console
- All nodes function correctly when used in workflows

## Impact

### Before Fix
- ❌ Only 1 node visible (Model Patch Memory Cleaner)
- ❌ 3 nodes missing (Memory Manager, Safe Memory Manager, Purge VRAM V2, Patch Sage Attention DM)
- ❌ Users unable to use missing nodes
- ❌ Silent failure (no error messages)

### After Fix
- ✅ All 4 nodes visible
- ✅ All nodes functional
- ✅ Users can access all features
- ✅ Proper error handling maintained

## Lessons Learned

### Best Practices

1. **Always Update Imports After Refactoring**
   - When moving files to subdirectories, immediately update all import statements
   - Use find-and-replace to ensure consistency

2. **Test After Structural Changes**
   - After refactoring, verify all imports work
   - Test that all nodes are visible in ComfyUI
   - Check console for import errors

3. **Use Explicit Error Messages**
   - Consider adding logging when imports fail:
   ```python
   except ImportError as e:
       print(f"Warning: Failed to import MemoryManager: {e}")
       MemoryManager = None
   ```

4. **Maintain Fallback Logic**
   - Keep the two-level try-except pattern for robustness
   - Ensure graceful degradation if imports fail

## Related Files

- **Fixed File:** `__init__.py`
- **Node Files:**
  - `nodes/memory_manager.py` (MemoryManager, SafeMemoryManager)
  - `nodes/purge_vram.py` (DisTorchPurgeVRAMV2)
  - `nodes/sa.py` (PatchSageAttentionDM)
- **Issue:** [GitHub Issue #3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/issues/3)
- **Release:** [v2.3.3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v2.3.3)

## Summary

The bug was caused by **outdated import paths** after refactoring the project structure. The fix was simple but critical: updating three import statements to point to the new `nodes/` subdirectory. This fix ensures all four custom nodes are properly imported and registered with ComfyUI, resolving the issue where only one node was visible after updates.

**Key Takeaway:** When refactoring project structure, always update import paths immediately and verify that all modules can be imported correctly.
