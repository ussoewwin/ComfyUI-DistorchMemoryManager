# Release Notes v2.3.3 - Fix Node Import Paths (Issue #3)

## Overview

This release fixes [Issue #3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/issues/3), which caused only one node (Model Patch Memory Cleaner) to be visible in ComfyUI after updating, despite the repository containing four nodes.

## Problem Description

### Symptom
After updating the ComfyUI-DistorchMemoryManager custom node, users reported that only **one node** (Model Patch Memory Cleaner) was visible in ComfyUI's node palette, even though the repository contains **four nodes**:
1. Memory Manager
2. Safe Memory Manager
3. Purge VRAM V2 (DisTorchPurgeVRAMV2)
4. Patch Sage Attention DM

### Root Causes

The issue had **two root causes**:

#### 1. Incorrect Import Paths

After refactoring the project structure to move node files into a `nodes/` subdirectory, the import statements in `__init__.py` were not updated to reflect the new directory structure.

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

The code was still trying to import from the root directory:
```python
# ❌ INCORRECT (Before Fix)
from .memory_manager import MemoryManager, SafeMemoryManager, any
from .purge_vram import DisTorchPurgeVRAMV2
from .sa import PatchSageAttentionDM
```

#### 2. Missing `__init__.py` in `nodes/` Directory

Even after fixing the import paths, the `nodes/` directory was missing an `__init__.py` file, which is required for Python to recognize it as a package. Without this file, relative imports (like `from .nodes.memory_manager import ...`) fail silently.

## Solution

### Fix 1: Corrected Import Paths

Updated all import statements to point to the `nodes/` subdirectory:

```python
# ✅ CORRECT (After Fix)
from .nodes.memory_manager import MemoryManager, SafeMemoryManager, any
from .nodes.purge_vram import DisTorchPurgeVRAMV2
from .nodes.sa import PatchSageAttentionDM
```

### Fix 2: Added `nodes/__init__.py`

Created `nodes/__init__.py` to make the `nodes/` directory a proper Python package:

```python
# nodes/__init__.py
# Nodes package for ComfyUI-DistorchMemoryManager
# This file makes the nodes directory a Python package
```

### Fix 3: Added Debug Logging

Added comprehensive debug logging to help diagnose import issues:

- **Import Success Logging**: Logs when imports succeed
- **Import Failure Logging**: Logs detailed error messages when imports fail
- **Node Registration Logging**: Logs which nodes are successfully registered
- **Registration Summary**: Logs the complete list of registered nodes

Example debug output:
```
[ComfyUI-DistorchMemoryManager] Successfully imported MemoryManager and SafeMemoryManager from .nodes.memory_manager
[ComfyUI-DistorchMemoryManager] Successfully imported DisTorchPurgeVRAMV2 from .nodes.purge_vram
[ComfyUI-DistorchMemoryManager] Successfully imported PatchSageAttentionDM from .nodes.sa
[ComfyUI-DistorchMemoryManager] Registered MemoryManager node
[ComfyUI-DistorchMemoryManager] Registered SafeMemoryManager node
[ComfyUI-DistorchMemoryManager] Registered DisTorchPurgeVRAMV2 node
[ComfyUI-DistorchMemoryManager] Registered PatchSageAttentionDM node
[ComfyUI-DistorchMemoryManager] Total registered nodes: ['ModelPatchMemoryCleaner', 'MemoryManager', 'SafeMemoryManager', 'DisTorchPurgeVRAMV2', 'PatchSageAttentionDM']
```

## Technical Details

### Why `__init__.py` is Required

In Python, a directory must contain an `__init__.py` file to be recognized as a package. Without it:
- Relative imports (`.nodes.memory_manager`) fail
- The directory is treated as a namespace package (Python 3.3+) or not recognized at all (Python < 3.3)
- Import errors occur silently in some cases

### Import Mechanism

The code uses a two-level try-except pattern for robustness:

```python
try:
    from .nodes.memory_manager import ...  # Relative import (preferred)
except ImportError:
    try:
        from nodes.memory_manager import ...  # Absolute import (fallback)
    except ImportError:
        # Set to None if both fail
```

**Why two levels?**
1. **First attempt (relative)**: Works when the package is properly installed and imported as a module
2. **Second attempt (absolute)**: Works in edge cases where the relative import fails but the absolute path is in `sys.path`
3. **Fallback**: Prevents `NameError` if both imports fail, allowing the code to continue executing

## Changes Made

### Files Modified

1. **`__init__.py`**
   - Updated import paths from `.memory_manager` to `.nodes.memory_manager`
   - Updated import paths from `.purge_vram` to `.nodes.purge_vram`
   - Updated import paths from `.sa` to `.nodes.sa`
   - Added comprehensive debug logging for imports and node registration

2. **`nodes/__init__.py`** (NEW FILE)
   - Created to make `nodes/` directory a proper Python package
   - Required for relative imports to work correctly

## Verification

### How to Verify the Fix

1. **Check ComfyUI Console Output**
   - Start ComfyUI and check the console for debug messages
   - Should see "Successfully imported" messages for all four nodes
   - Should see "Registered" messages for all four nodes
   - Should see the complete list of registered nodes

2. **Check ComfyUI Node Palette**
   - Open ComfyUI
   - Search for "Memory" category
   - Should see all 4 nodes listed:
     - Memory Manager
     - Safe Memory Manager
     - LayerUtility: Purge VRAM V2
     - Patch Sage Attention DM
   - Model Patch Memory Cleaner should also be visible

3. **Test Node Functionality**
   - Add each node to a workflow
   - Verify that all nodes function correctly
   - Check that no import errors appear in the console

### Expected Behavior After Fix

- ✅ **All 4 nodes** appear in ComfyUI's node palette under the "Memory" category
- ✅ No import errors in ComfyUI console
- ✅ All nodes function correctly when used in workflows
- ✅ Debug logging provides clear information about import and registration status

## Impact

### Before Fix
- ❌ Only 1 node visible (Model Patch Memory Cleaner)
- ❌ 3 nodes missing (Memory Manager, Safe Memory Manager, Purge VRAM V2, Patch Sage Attention DM)
- ❌ Users unable to use missing nodes
- ❌ Silent failure (no error messages)
- ❌ Difficult to diagnose the problem

### After Fix
- ✅ All 4 nodes visible
- ✅ All nodes functional
- ✅ Users can access all features
- ✅ Proper error handling maintained
- ✅ Debug logging helps diagnose future issues

## Related Information

- **Issue**: [GitHub Issue #3](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/issues/3)
- **Fixed Files**: 
  - `__init__.py` (import paths and debug logging)
  - `nodes/__init__.py` (new file)
- **Documentation**: See `ISSUE_3_FIX_DOCUMENTATION.md` for complete technical details

## Summary

This release fixes a critical bug that prevented three of four custom nodes from being visible in ComfyUI. The fix involved:
1. Correcting import paths to reflect the new `nodes/` directory structure
2. Adding `nodes/__init__.py` to make the directory a proper Python package
3. Adding debug logging to help diagnose future import issues

**Key Takeaway**: When refactoring project structure, always:
- Update import paths immediately
- Ensure all subdirectories have `__init__.py` files if they contain Python modules
- Add debug logging to help diagnose issues
- Test thoroughly after structural changes
