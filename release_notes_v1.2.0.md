# Release Notes v1.2.0

## Overview

Version 1.2.0 introduces the **Model Patch Memory Cleaner** node, a dedicated memory management solution for ModelPatchLoader model patches. This release also includes significant enhancements to the **DisTorchPurgeVRAMV2** node with more aggressive model unloading capabilities and improved error handling.

## New Features

### Model Patch Memory Cleaner Node

A new dedicated node for clearing model patches loaded via ModelPatchLoader to prevent OOM (Out of Memory) errors during upscaling operations.

#### Purpose

The ModelPatchMemoryCleaner node is designed to explicitly clear model patches (such as Z-Image ControlNet, QwenImage BlockWise ControlNet, SigLIP MultiFeat Proj) loaded via ModelPatchLoader from VRAM. This prevents OOM errors during upscaling operations by freeing memory occupied by unused model patches.

#### Problem Background

Model patches loaded via ModelPatchLoader are managed differently from standard models in ComfyUI's memory system. These patches (stored in ModelPatcher's `additional_models` or `attachments`) can remain in VRAM even after use, causing OOM errors during subsequent operations like upscaling. Existing memory cleaning nodes cannot properly detect and clear these model patches, necessitating a dedicated solution.

#### Implementation Details

**File Created/Modified:**
- `ComfyUI/custom_nodes/ComfyUI-DistorchMemoryManager/__init__.py`

**Complete Code:**

```python
class ModelPatchMemoryCleaner:
    """
    Memory cleaner specifically for ModelPatcher loaded model patches.
    Clears model patches loaded via ModelPatchLoader to prevent OOM during upscaling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "clear_model_patches": ("BOOLEAN", {"default": True, "tooltip": "Clear model patches loaded via ModelPatchLoader"}),
                "clean_gpu": ("BOOLEAN", {"default": True}),
                "force_gc": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "clear_model_patches"
    CATEGORY = "Memory"

    def clear_model_patches(self, anything, clear_model_patches, clean_gpu, force_gc):
        try:
            if clear_model_patches:
                import comfy.model_management
                import comfy.model_patcher
                
                # Get current loaded models
                if hasattr(comfy.model_management, "current_loaded_models"):
                    current_loaded_models = comfy.model_management.current_loaded_models
                    
                    # Find and unload model patches
                    unloaded_count = 0
                    for i in range(len(current_loaded_models) - 1, -1, -1):
                        loaded_model = current_loaded_models[i]
                        if loaded_model is not None and hasattr(loaded_model, "model"):
                            model = loaded_model.model
                            # Check if this is a ModelPatcher with additional_models (model patches stored here)
                            if isinstance(model, comfy.model_patcher.ModelPatcher):
                                # Check for additional_models (model patches stored here)
                                if hasattr(model, "additional_models") and model.additional_models:
                                    # Mark as not currently used
                                    loaded_model.currently_used = False
                                    # Unload the model
                                    if hasattr(loaded_model, "model_unload"):
                                        loaded_model.model_unload()
                                    # Remove from current_loaded_models
                                    current_loaded_models.pop(i)
                                    unloaded_count += 1
                                    print(f"Unloaded model patch: {type(model.model).__name__ if hasattr(model, 'model') else 'ModelPatcher'}")
                                # Also check attachments for model patches
                                elif hasattr(model, "attachments") and model.attachments:
                                    # Mark as not currently used
                                    loaded_model.currently_used = False
                                    # Unload the model
                                    if hasattr(loaded_model, "model_unload"):
                                        loaded_model.model_unload()
                                    # Remove from current_loaded_models
                                    current_loaded_models.pop(i)
                                    unloaded_count += 1
                                    print(f"Unloaded model patch from attachments: {type(model.model).__name__ if hasattr(model, 'model') else 'ModelPatcher'}")
                    
                    if unloaded_count > 0:
                        print(f"Cleared {unloaded_count} model patch(es)")
                    
                    # Cleanup models GC
                    if hasattr(comfy.model_management, "cleanup_models_gc"):
                        comfy.model_management.cleanup_models_gc()
            
            if clean_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU memory cleared")
            
            if force_gc:
                gc.collect()
                print("Garbage collection completed")
            
            print("Model patch memory cleanup completed")
            
        except Exception as e:
            print(f"Model patch memory cleanup error: {e}")
        
        return (anything,)
```

#### Code Explanation

**1. Class Definition and Documentation**

The `ModelPatchMemoryCleaner` class is a dedicated memory cleaning node for model patches loaded via ModelPatcher. It was created to prevent OOM errors during upscaling operations.

**2. INPUT_TYPES Method (Node Input Definition)**

The `INPUT_TYPES` method defines the node's input parameters in ComfyUI:

- **anything**: AnyType input that accepts any data type and passes it through to the output. This is a passthrough input for data flow in ComfyUI workflows.
- **clear_model_patches**: Boolean value (default: True). Controls whether to clear model patches loaded via ModelPatchLoader. When True, detects and unloads model patches.
- **clean_gpu**: Boolean value (default: True). Controls whether to clear GPU memory. When True, executes `torch.cuda.empty_cache()` and `torch.cuda.synchronize()`.
- **force_gc**: Boolean value (default: True). Controls whether to force garbage collection. When True, executes `gc.collect()`.

**3. RETURN_TYPES and RETURN_NAMES (Node Output Definition)**

- **RETURN_TYPES**: Defines the node's output type. Returns one `any` type.
- **RETURN_NAMES**: Defines the output name. Output is named "any".
- **FUNCTION**: Specifies the method name to execute. The `clear_model_patches` method is called.
- **CATEGORY**: Node category. The node appears in the "Memory" category in ComfyUI's node menu.

**4. clear_model_patches Method (Main Processing)**

The main processing method that accepts four parameters.

**4.1. Model Patch Clearing Process**

When `clear_model_patches` is True, the model patch clearing process is executed:

- Imports `comfy.model_management` and `comfy.model_patcher`, which are ComfyUI core modules providing model management and ModelPatcher functionality.
- `current_loaded_models` is a list of models currently loaded in memory, managed by ComfyUI's model_management module.

**4.2. Model Patch Detection and Unloading**

The code iterates through the list from back to front to prevent index shifting when removing elements:

- Checks if each `loaded_model` is not None and has a `model` attribute.
- Verifies if the model is a `ModelPatcher` instance. Model patches loaded via ModelPatchLoader are wrapped in ModelPatcher.
- Checks the `additional_models` attribute. ModelPatcher stores additional models (model patches) in the `additional_models` dictionary. If this dictionary is not empty, it means model patches are loaded.
- For model patches found:
  - Sets `currently_used` to False, marking the model as "not in use" in ComfyUI's memory management system.
  - Calls `model_unload()` to unload the model, moving it from VRAM to CPU memory or disk.
  - Removes the model from the `current_loaded_models` list using `pop(i)`.
  - Increments `unloaded_count` to track the number of unloaded model patches.
  - Prints the type name of the unloaded model patch for debugging.
- Also checks the `attachments` attribute, as ModelPatcher may store model patches in `attachments` as well.

**4.3. Cleanup Process**

- Prints the number of unloaded model patches.
- Calls `cleanup_models_gc()` to perform garbage collection, cleaning up references to deleted models and preventing memory leaks.

**4.4. GPU Memory Clearing**

When `clean_gpu` is True and CUDA is available:

- `torch.cuda.empty_cache()`: Clears PyTorch's CUDA cache, freeing unused GPU memory.
- `torch.cuda.synchronize()`: Waits for CUDA operations to complete, ensuring memory clearing is fully completed.

**4.5. Garbage Collection**

When `force_gc` is True:

- `gc.collect()`: Executes Python's garbage collector, reclaiming unused objects including circular references.

**4.6. Error Handling**

All processing is wrapped in a try-except block to prevent node crashes even if errors occur:

- If an error occurs, an error message is printed.
- Finally, returns the `anything` input as-is to the output, allowing data to continue flowing through the workflow.

**5. Node Registration**

```python
NODE_CLASS_MAPPINGS = {
    ...
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ...
    "ModelPatchMemoryCleaner": "Model Patch Memory Cleaner",
}
```

- **NODE_CLASS_MAPPINGS**: Registers the node class with ComfyUI, allowing ComfyUI to recognize the node.
- **NODE_DISPLAY_NAME_MAPPINGS**: Defines the node's display name, shown in ComfyUI's node menu.

#### Usage Example

Place the ModelPatchMemoryCleaner node before upscaling operations to clear unused model patches and prevent OOM errors.

**Workflow Example:**
1. Load Z-Image ControlNet via ModelPatchLoader
2. Use ControlNet in QwenImageDiffsynthControlnet
3. Clear model patches with ModelPatchMemoryCleaner
4. Execute upscaling operation

Executing in this order ensures that model patches are not left in VRAM during upscaling, preventing memory shortages.

#### Technical Details

**ModelPatcher Structure:**
ModelPatcher is ComfyUI's model wrapper class. Model patches (such as ControlNet) are stored in `additional_models` or `attachments`.

- **additional_models**: Dictionary type. Keys are strings, values are lists of ModelPatcher instances. Can manage multiple model patches.
- **attachments**: Dictionary type. Stores additional model information.
- **current_loaded_models**: List of currently loaded models managed by ComfyUI's model_management module. A list of LoadedModel objects.
- **LoadedModel**: Class that wraps models. The `model` attribute stores a ModelPatcher, and the `currently_used` attribute manages whether it's in use.
- **model_unload()**: Method of LoadedModel. Unloads the model from VRAM and frees memory.

**Error Prevention:**
The implementation includes the following error prevention measures:

- **None checks**: Verifies that `loaded_model` and `model` are not None before processing.
- **hasattr checks**: Verifies that attributes exist before accessing them.
- **isinstance checks**: Verifies object types before processing.
- **try-except blocks**: Wraps all processing in try-except to prevent node crashes even if errors occur.

## Enhancements

### DisTorchPurgeVRAMV2 Improvements

The DisTorchPurgeVRAMV2 node has been significantly enhanced with more aggressive model unloading capabilities and improved error handling.

#### Enhanced Model Unloading

**Previous Behavior:**
- Only called `cleanup_models()` or `unload_model_to_cpu()` if available
- Basic error handling

**New Behavior (v1.2.0):**
- Calls `cleanup_models()` to remove dead models first
- Calls `cleanup_models_gc()` for garbage collection
- Marks all models as not currently used (`currently_used = False`)
- Aggressively unloads each model individually via `model_unload()`
- Displays the number of unloaded models
- Calls `cleanup_models()` again after unloading
- Calls `soft_empty_cache()` if available
- Comprehensive error handling with detailed error messages

#### Improved Error Handling

**None Checks:**
- Added `callable()` checks before calling `is_dead()`, `model_unload()`, and other methods
- Prevents "'NoneType' object is not callable" errors

**Error Messages:**
- Detailed error messages for each operation
- Separate error handling for `cleanup_models()`, `cleanup_models_gc()`, and `soft_empty_cache()`
- Error messages include context about which operation failed

#### Code Changes

```python
# Enhanced purge_models section
if purge_models:
    try:
        import comfy.model_management

        # Cleanup dead models first
        if hasattr(comfy.model_management, "cleanup_models") and callable(comfy.model_management.cleanup_models):
            try:
                comfy.model_management.cleanup_models()
            except Exception as e:
                print(f"Error in cleanup_models: {e}")
        
        # Cleanup models GC
        if hasattr(comfy.model_management, "cleanup_models_gc") and callable(comfy.model_management.cleanup_models_gc):
            try:
                comfy.model_management.cleanup_models_gc()
            except Exception as e:
                print(f"Error in cleanup_models_gc: {e}")
        
        # More aggressive model unloading
        if hasattr(comfy.model_management, "current_loaded_models"):
            current_loaded_models = comfy.model_management.current_loaded_models
            unloaded_count = 0
            
            # Mark all models as not currently used
            for loaded_model in current_loaded_models:
                if loaded_model is not None:
                    try:
                        if hasattr(loaded_model, "is_dead") and callable(loaded_model.is_dead):
                            if not loaded_model.is_dead():
                                loaded_model.currently_used = False
                        else:
                            loaded_model.currently_used = False
                    except Exception as e:
                        print(f"Error checking model status: {e}")
            
            # Try to unload models
            for i in range(len(current_loaded_models) - 1, -1, -1):
                loaded_model = current_loaded_models[i]
                if loaded_model is not None:
                    try:
                        if hasattr(loaded_model, "is_dead") and callable(loaded_model.is_dead):
                            if loaded_model.is_dead():
                                continue
                        if hasattr(loaded_model, "model_unload") and callable(loaded_model.model_unload):
                            if loaded_model.model_unload():
                                unloaded_count += 1
                    except Exception as e:
                        print(f"Error unloading model: {e}")
            
            if unloaded_count > 0:
                print(f"Unloaded {unloaded_count} model(s)")
            
            # Cleanup again after unloading
            if hasattr(comfy.model_management, "cleanup_models"):
                try:
                    comfy.model_management.cleanup_models()
                except Exception as e:
                    print(f"Error in cleanup_models: {e}")
        
        # Soft empty cache (if available)
        if hasattr(comfy.model_management, "soft_empty_cache") and callable(comfy.model_management.soft_empty_cache):
            try:
                comfy.model_management.soft_empty_cache()
            except Exception as e:
                print(f"Error in soft_empty_cache: {e}")
                
    except Exception as e:
        print(f"Error purging models: {e}")
```

## Core ComfyUI Improvements

### cleanup_models() Enhancement

Enhanced the `cleanup_models()` function in `ComfyUI/comfy/model_management.py` to safely handle cases where `real_model` is None:

```python
def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        try:
            loaded_model = current_loaded_models[i]
            if loaded_model is not None:
                if loaded_model.real_model is None:
                    to_delete = [i] + to_delete
                else:
                    try:
                        if loaded_model.real_model() is None:
                            to_delete = [i] + to_delete
                    except (TypeError, AttributeError):
                        # real_model might be None or not callable
                        to_delete = [i] + to_delete
        except Exception:
            # Skip if there's any error
            pass

    for i in to_delete:
        try:
            x = current_loaded_models.pop(i)
            del x
        except Exception:
            pass
```

### is_dead() Enhancement

Enhanced the `is_dead()` method in `LoadedModel` class to safely handle cases where `real_model` is None:

```python
def is_dead(self):
    if self.real_model is None:
        return False
    try:
        return self.real_model() is not None and self.model is None
    except (TypeError, AttributeError):
        # real_model might be None or not callable
        return False
```

These improvements prevent "'NoneType' object is not callable" errors that could occur when models are unloaded.

## Summary

Version 1.2.0 introduces a dedicated solution for managing model patches loaded via ModelPatchLoader, preventing OOM errors during upscaling operations. The DisTorchPurgeVRAMV2 node has been significantly enhanced with more aggressive model unloading and improved error handling. Core ComfyUI functions have also been improved to handle edge cases safely.

**Key Benefits:**
- Prevents OOM errors during upscaling after ModelPatchLoader usage
- More aggressive VRAM clearing with DisTorchPurgeVRAMV2
- Improved error handling and stability
- Safe handling of edge cases in core ComfyUI functions

