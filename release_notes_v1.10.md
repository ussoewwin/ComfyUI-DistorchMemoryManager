## Complete Explanation of the Copy Task

### 1. Purpose

Recreate the `LayerUtility: Purge VRAM V2` node—originally available in the LayerStyle package—with identical UI and behavior inside the `ComfyUI-DistorchMemoryManager` extension so it remains usable without missing functionality.

### 2. Target File

- `ComfyUI/custom_nodes/ComfyUI-DistorchMemoryManager/__init__.py`

This module defines the DistorchMemoryManager nodes; the Purge VRAM node was added here.

### 3. Added/Modified Class

```
class DisTorchPurgeVRAMV2:
    """
    Compatibility clone of the original LayerUtility PurgeVRAM V2 node,
    maintained within the Distortch Memory Manager package.
    """
```

- **INPUT_TYPES**

  ```python
  {
      "required": {
          "anything": (any, {}),
          "purge_cache": ("BOOLEAN", {"default": True}),
          "purge_models": ("BOOLEAN", {"default": True}),
      }
  }
  ```

  Uses the same input names as the LayerStyle version. `anything` leverages `AnyType` so the value passes through untouched.

- **RETURN_TYPES / RETURN_NAMES**

  `("any",)` – returns the incoming `anything` value directly to keep existing workflows intact.

- **FUNCTION**

  `"purge_vram"`

- **CATEGORY**

  `"DisTorch/Memory"` – keeps the node under the Distortch namespace while avoiding clashes with LayerStyle categories.

- **NODE_DISPLAY_NAME_MAPPINGS**

  ```python
  "DisTorchPurgeVRAMV2": "LayerUtility: Purge VRAM V2"
  ```

  Matches the display name used in workflows.

### 4. Behavior

When `purge_cache` is `True` (default), the node performs:

1. `gc.collect()` to clear Python-level garbage.
2. If CUDA is available:
   - Save the current device index.
   - Iterate over every GPU with `torch.cuda.set_device(idx)`.
   - Execute `torch.cuda.empty_cache()` to flush CUDA caches.
   - Attempt `torch.cuda.ipc_collect()` to release shared memory (ignoring failures).
   - Restore the original device.

When `purge_models` is `True` (default):

- Attempt to call `comfy.model_management.cleanup_models()`.
- If unavailable, fall back to `unload_model_to_cpu()`.
- Suppress exceptions to avoid breaking workflows if either function is missing.

Finally, the node returns `(anything,)`, so downstream nodes receive exactly what came in.

### 5. Node Registration

```python
NODE_CLASS_MAPPINGS = {
    "DisTorchPurgeVRAMV2": DisTorchPurgeVRAMV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DisTorchPurgeVRAMV2": "LayerUtility: Purge VRAM V2",
}
```

This ensures ComfyUI loads the node under the original display name at startup or when reloading custom nodes.

### 6. Verification

- Restarted ComfyUI and ran `Manager > Reload Custom Nodes`; both paths loaded the node.
- Confirmed that searching for “Purge VRAM V2” surfaces the node under `DisTorch/Memory`.
- UI matches the LayerStyle version: `anything` input, `purge_cache` and `purge_models` booleans, and `any` output.
- Runtime behavior: cache clearing and model unloading trigger as expected, while the input payload passes through unchanged.
