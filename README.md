# ComfyUI-DistorchMemoryManager

An independent memory management custom node for ComfyUI. Provides Distorch memory management functionality for efficient GPU/CPU memory handling.

## Overview

This custom node was created to address OOM (Out Of Memory) issues in video generation workflows like Upscaling with WAN2.2. The key point is that these OOM errors are caused by **system RAM shortage, not VRAM shortage** (can occur even on 64GB RAM systems depending on resolution and video length).

This is a completely original implementation designed specifically for Distorch memory management. Simply place it in the `custom_nodes` folder for easy installation and removal.

## Features

### Four Node Types

#### Purge VRAM V2 Compatibility (New in v1.10)
- **Description**: Restored LayerStyle’s **LayerUtility: Purge VRAM V2** inside the Distortch suite ([original node](https://github.com/chflame163/ComfyUI_LayerStyle))
- **Features**: Identical UI/behavior; keeps legacy workflows working without LayerStyle
- **Input**: Any data type (ANY) passthrough
- **Options**:
  - `purge_cache`: Run `gc.collect()`, flush CUDA caches, call `torch.cuda.ipc_collect()`
  - `purge_models`: Call `comfy.model_management.cleanup_models()` (fallback to `unload_model_to_cpu`)
  - **Reason**: The original LayerStyle node disappeared upstream, so we duplicated it here to keep older workflows alive.

#### Memory Cleaner
- **Description**: Basic memory cleaning node
- **Features**: Simple and safe memory management
- **Input**: Any data type (ANY)
- **Output**: Any data type (ANY)
- **Functions**:
  - GPU cache clearing
  - Python garbage collection
  - Distorch virtual memory release

#### Safe Memory Manager (Recommended)
- **Description**: Safe memory management node
- **Features**: Completely prevents UI corruption with safe memory management
- **Input**: Any data type (ANY)
- **Output**: Any data type (ANY)
- **Options**:
  - `clean_gpu`: Clear GPU memory
  - `force_gc`: Force garbage collection
  - `reset_virtual_memory`: Reset virtual memory

#### Memory Manager (Advanced)
- **Description**: Comprehensive memory management node (for advanced users)
- **Features**: Detailed memory management with UI corruption protection
- **Input**: Any data type (ANY)
- **Output**: Any data type (ANY)
- **Options**:
  - `clean_gpu`: Clear GPU memory
  - `clean_cpu`: Clear CPU memory (use with caution)
  - `force_gc`: Force garbage collection
  - `reset_virtual_memory`: Reset virtual memory
  - `restore_original_functions`: Restore original functions

## Installation

1. Clone or download to `ComfyUI/custom_nodes/` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager.git
```

2. Install dependencies:
```bash
cd ComfyUI-DistorchMemoryManager
pip install -r requirements.txt
```

3. Restart ComfyUI
4. Nodes will appear in the "Memory" category in the node palette

## Usage

### Basic Usage
1. Add any memory management node to your workflow
2. Connect any data to the input
3. Configure options as needed
4. Connect output to the next node

### Recommended Workflow Placement
```
[Previous Node] → [Memory Cleaner] → [Next Node]
```

or

```
[Previous Node] → [Safe Memory Manager] → [Next Node]
```

### Recommended Settings

**For video generation (WAN2.2, etc.)**:
- Use **Safe Memory Manager**
- `clean_gpu: True`
- `force_gc: True`
- `reset_virtual_memory: True`

**For maximum memory release**:
- Use **Memory Manager**
- `clean_cpu: True` (Warning: possible UI corruption)

## Comparison with Purge VRAM

### Distorch Memory Manager Advantages
1. **Distorch-specific functionality**:
   - `comfy.model_management.free_memory(0, 'cuda:0')` - Direct virtual memory release
   - `comfy.model_management.free_memory(0, 'cpu')` - CPU virtual memory release
2. **More detailed memory management**:
   - `torch.cuda.synchronize()` - Complete GPU synchronization
   - Memory usage measurement before/after
   - Detailed logging for effect verification
3. **Safe design**:
   - Safe memory clearing to prevent UI corruption
   - Error protection with exception handling

### Purge VRAM Advantages
1. **Model unloading functionality**:
   - `comfy.model_management.unload_all_models()` - Force unload all models
   - `comfy.model_management.soft_empty_cache()` - Soft cache clearing

**Recommendation**: Use Distorch Memory Manager together with Purge VRAM when model unloading is needed.

## Troubleshooting

### Out of Memory Errors
**Solution**:
1. Use **Safe Memory Manager**
2. Enable `clean_gpu` and `reset_virtual_memory`
3. Enable `force_gc` if needed

### UI Corruption
**Solution**:
1. Use **Safe Memory Manager**
2. Keep `clean_cpu` disabled
3. Enable only essential options

## Technical Details

### Implemented Features
- GPU memory clearing (`torch.cuda.empty_cache()`)
- GPU synchronization (`torch.cuda.synchronize()`)
- CPU memory clearing (`gc.collect()`)
- Virtual memory reset (`comfy.model_management.free_memory()`)

### Safety Features
- Safe implementation to prevent UI corruption
- Error handling with exception processing
- Gradual memory clearing

### Performance Characteristics
**Distorch Memory Manager is superior** in performance due to:
1. **Distorch-specific functions**: Direct virtual memory release with `free_memory()`
2. **More detailed management**: Memory usage measurement, detailed logging
3. **Safe design**: Considerations to prevent UI corruption
4. **Flexibility**: Three different levels of nodes

## Additional Tips

- Expanding paging file size can also reduce OOM occurrences during upscaling
- Note: For OOM during video generation inference (where VRAM is critical), paging file expansion won't help

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details

## Contributing

Bug reports and feature requests are welcome on the GitHub Issues page.

## Release History

- [v1.10](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager/releases/tag/v1.10) – Added the LayerUtility: Purge VRAM V2 compatibility node within DisTorch Memory Manager.

## Changelog

### v1.1.0
- ANY type input/output support
- Simplified node names
- Changed category to "Memory"

### v1.0.0
- Initial release
- Basic memory management functionality implemented
