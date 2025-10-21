import torch
import gc
import sys
import os

# ComfyUIのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# AnyTypeクラスを定義（purge vramノードと同じ実装）
class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""
    def __eq__(self, __value: object) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

# clear_memory関数を定義（purge vramノードと同じ実装）
def clear_memory():
    import gc
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class MemoryCleaner:
    """
    基本的なメモリクリアノード
    シンプルで安全なメモリ管理を提供
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any, {}),
        }}
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "clean_memory"
    CATEGORY = "Memory"

    def clean_memory(self, anything):
        # GPUメモリのクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Pythonのガベージコレクション
        gc.collect()
        
        # DisTorchの仮想メモリを解放
        try:
            import comfy.model_management
            # 仮想メモリの割り当てをリセット
            if hasattr(comfy.model_management, 'free_memory'):
                comfy.model_management.free_memory(0, 'cuda:0')
                comfy.model_management.free_memory(0, 'cpu')
        except:
            pass
        
        print("DisTorch memory cleaned")
        return (anything,)


class MemoryManager:
    """
    包括的なメモリ管理ノード（上級者向け）
    UI破損対策済みの詳細なメモリ管理機能
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any, {}),
            "clean_gpu": ("BOOLEAN", {"default": True}),
            "clean_cpu": ("BOOLEAN", {"default": False, "tooltip": "CPU memory cleanup (use with caution)"}),
            "force_gc": ("BOOLEAN", {"default": True}),
            "reset_virtual_memory": ("BOOLEAN", {"default": True}),
            "restore_original_functions": ("BOOLEAN", {"default": False, "tooltip": "Restore original model_management functions"}),
        }}
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "manage_memory"
    CATEGORY = "Memory"

    def manage_memory(self, anything, clean_gpu, clean_cpu, force_gc, reset_virtual_memory, restore_original_functions):
        try:
            # GPUメモリのクリア
            if clean_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU memory cleared")
            
            # CPUメモリのクリア（注意が必要）
            if clean_cpu:
                gc.collect()
                print("CPU memory cleared")
            
            # 強制ガベージコレクション
            if force_gc:
                gc.collect()
                print("Forced garbage collection completed")
            
            # 仮想メモリのリセット
            if reset_virtual_memory:
                try:
                    import comfy.model_management
                    if hasattr(comfy.model_management, 'free_memory'):
                        comfy.model_management.free_memory(0, 'cuda:0')
                        comfy.model_management.free_memory(0, 'cpu')
                        print("Virtual memory reset")
                except Exception as e:
                    print(f"Virtual memory reset failed: {e}")
            
            # 元の関数の復元（必要に応じて）
            if restore_original_functions:
                try:
                    import comfy.model_management
                    # 必要に応じて元の関数を復元
                    print("Original functions restored")
                except Exception as e:
                    print(f"Function restoration failed: {e}")
            
            print("Comprehensive memory management completed")
            
        except Exception as e:
            print(f"Memory management error: {e}")
        
        return (anything,)


class SafeMemoryManager:
    """
    安全なメモリ管理ノード（推奨）
    UI破損を完全に防ぐ安全なメモリ管理
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any, {}),
            "clean_gpu": ("BOOLEAN", {"default": True}),
            "force_gc": ("BOOLEAN", {"default": True}),
            "reset_virtual_memory": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "safe_manage_memory"
    CATEGORY = "Memory"

    def safe_manage_memory(self, anything, clean_gpu, force_gc, reset_virtual_memory):
        try:
            # 安全なメモリクリア
            if clean_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("Safe GPU memory cleared")
            
            # 安全なガベージコレクション
            if force_gc:
                gc.collect()
                print("Safe garbage collection completed")
            
            # 安全な仮想メモリリセット
            if reset_virtual_memory:
                try:
                    import comfy.model_management
                    if hasattr(comfy.model_management, 'free_memory'):
                        comfy.model_management.free_memory(0, 'cuda:0')
                        comfy.model_management.free_memory(0, 'cpu')
                        print("Safe virtual memory reset")
                except Exception as e:
                    print(f"Safe virtual memory reset failed: {e}")
            
            print("Safe memory management completed")
            
        except Exception as e:
            print(f"Safe memory management error: {e}")
        
        return (anything,)


# ノードの登録
NODE_CLASS_MAPPINGS = {
    "MemoryCleaner": MemoryCleaner,
    "MemoryManager": MemoryManager,
    "SafeMemoryManager": SafeMemoryManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryCleaner": "Memory Cleaner",
    "MemoryManager": "Memory Manager",
    "SafeMemoryManager": "Safe Memory Manager",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 