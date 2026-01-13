# SageAttentionパッチ機能追加の完全解説

## 概要

ComfyUI-DistorchMemoryManagerノードに、SageAttentionをComfyUIのattention機構にパッチする機能を追加しました。この機能により、ComfyUIの標準attention機構をSageAttentionに置き換えることができ、メモリ効率とパフォーマンスの向上が期待できます。

## 追加・修正したファイル

**ファイル**: `__init__.py`

このファイルのみを修正・追加しています。

## 追加したコード内容とその意味

### 1. インポート文の追加

```python
from comfy.ldm.modules import attention as comfy_attention
import comfy.model_management as mm
from comfy.cli_args import args
from comfy.ldm.modules.attention import wrap_attn
```

**意味**:
- `comfy_attention`: ComfyUIのattentionモジュールへのアクセス（元のattention関数を確認するため）
- `mm`: `model_management`モジュール（Flash-Attentionの状態やバージョン情報を取得するため）
- `args`: コマンドライン引数（Flash-Attentionが有効化されているか確認するため）
- `wrap_attn`: attention関数をラップするデコレータ（ComfyUIのattention形式に合わせるため）

### 2. SageAttentionモード定義（2179行目）

```python
sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]
```

**意味**: 利用可能なSageAttention実装モードのリスト

- `disabled`: SageAttentionを無効化（元のattention機構に戻す）
- `auto`: 自動選択のSageAttention実装
- `sageattn_qk_int8_pv_fp16_cuda`: CUDA実装（QK int8、PV FP16）
- `sageattn_qk_int8_pv_fp16_triton`: Triton実装（QK int8、PV FP16）
- `sageattn_qk_int8_pv_fp8_cuda`: CUDA実装（QK int8、PV FP8）
- `sageattn_qk_int8_pv_fp8_cuda++`: CUDA実装（QK int8、PV FP8、最適化版）
- `sageattn3`: SageAttention 3実装（Blackwell対応）
- `sageattn3_per_block_mean`: SageAttention 3実装（per-block mean版）

### 3. `get_sage_func_dm`関数（2181-2273行目）

この関数は、選択されたSageAttentionモードに応じて、適切なattention関数を返します。

#### バージョン検出とログ出力（2182-2206行目）

```python
def get_sage_func_dm(sage_attention, allow_compile=False):
    # Detect SageAttention version
    try:
        import sageattention
        sage_version = None
        try:
            sage_version = sageattention.__version__
        except AttributeError:
            try:
                import importlib.metadata
                sage_version = importlib.metadata.version("sageattention")
            except Exception:
                sage_version = None
        
        if sage_version and sage_version != "unknown":
            try:
                import torch
                cuda_version = torch.version.cuda or "unknown"
                torch_version = torch.version.__version__ or "unknown"
                logging.info(f"Patching comfy attention to use SageAttention {sage_version}+cu{cuda_version}torch{torch_version}")
            except:
                logging.info(f"Patching comfy attention to use SageAttention {sage_version}")
        else:
            logging.info("Patching comfy attention to use sageattn")
    except:
        logging.info("Patching comfy attention to use sageattn")
```

**意味**:
1. SageAttentionのバージョンを検出（`__version__`属性または`importlib.metadata`を使用）
2. CUDAバージョンとPyTorchバージョンも取得
3. ログに詳細なバージョン情報を出力（例: "Patching comfy attention to use SageAttention 2.2.0+cu121torch2.3.0"）
4. バージョン情報が取得できない場合は、フォールバックメッセージを出力

#### モード別の関数作成（2208-2233行目）

```python
    from sageattention import sageattn
    if sage_attention == "auto":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    # ... 他のモードも同様
```

**意味**: 選択されたモードに応じて、対応するSageAttention関数をインポートし、適切なパラメータで呼び出す内部関数`sage_func`を作成します。

#### torch.compile制御（2235-2236行目）

```python
    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)
```

**意味**: `allow_compile=False`の場合、`torch.compile`を無効化します。これにより、コンパイルによるオーバーヘッドを避けられます。

#### ComfyUI互換ラッパー（2238-2272行目）

```python
    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        # ... テンソル形状の変換処理
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        # ... 出力形状の変換処理
        return out
    return attention_sage
```

**意味**:
1. `@wrap_attn`デコレータでComfyUIのattention関数形式にラップ
2. ComfyUIから渡されるテンソル形式（`q, k, v, heads`）をSageAttentionが期待する形式（`q, k, v`の形状変換）に変換
3. FP32テンソルをFP16に変換（SageAttentionは主にFP16で動作）
4. マスクの次元を調整（batch次元やheads次元の追加）
5. 出力を元のデータ型と形状に戻す

### 4. `CallbacksMP`のインポート（2276行目）

```python
from comfy.patcher_extension import CallbacksMP
```

**意味**: ComfyUIのモデルパッチ拡張機能からコールバック機能をインポート。これにより、モデル実行の前後で処理を実行できます。

### 5. `PatchSageAttentionDM`クラス（2277-2398行目）

ComfyUIのカスタムノードクラス。モデルのattention機構をSageAttentionにパッチします。

#### クラス定義とINPUT_TYPES（2277-2292行目）

```python
class PatchSageAttentionDM():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."}),
        },
        "optional": {
            "allow_compile": ("BOOLEAN", {"default": False, "tooltip": "Allow the use of torch.compile for the sage attention function, requires latest sageattn 2.2.0 or higher."})
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching attention mode. This doesn't use the model patching system and thus can't be disabled without running the node again with 'disabled' option."
    CATEGORY = "Memory"
```

**意味**: ComfyUIノードの定義
- **入力**:
  - `model`: パッチする対象のモデル（MODEL型）
  - `sage_attention`: 使用するSageAttentionモード（`sageattn_modes`から選択）
  - `allow_compile`: torch.compileを有効にするか（オプション、デフォルトFalse）
- **出力**: パッチ済みのモデル（MODEL型）
- **カテゴリ**: "Memory"（UI上でこのカテゴリに表示される）

#### `patch`メソッド（2294-2398行目）

このメソッドがノード実行時に呼び出されます。

##### `patch_attention_enable`コールバック（2297-2348行目）

```python
        @torch.compiler.disable()
        def patch_attention_enable(model):
            if sage_attention != "disabled":
                new_attention = get_sage_func_dm(sage_attention, allow_compile=allow_compile)
                def attention_override_sage(func, *args, **kwargs):
                    return new_attention.__wrapped__(*args, **kwargs)
                
                if "transformer_options" not in model.model_options:
                    model.model_options["transformer_options"] = {}
                model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
            else:
                # disabled時の処理（Flash-Attentionの状態確認とログ出力）
                # ...
```

**意味**: `ON_PRE_RUN`コールバックとして登録され、各モデル実行の**前**に呼び出されます。

1. **SageAttention有効時（2299-2306行目）**:
   - `get_sage_func_dm`を呼び出してSageAttention関数を取得（この時点でログが出力される）
   - `attention_override_sage`というラッパー関数を作成
   - モデルの`model_options["transformer_options"]["optimized_attention_override"]`に設定することで、ComfyUIのattention機構をSageAttentionに置き換える

2. **`disabled`時（2307-2348行目）**:
   - 既存のオーバーライドを削除して元のattention機構に戻す
   - Flash-Attentionが有効かどうかを`model_management`から確認（優先順位: `FLASH_IS_AVAILABLE` → `flash_attention_enabled()` → 関数名チェック）
   - Flash-Attentionが有効な場合、バージョン情報を含めてログ出力（FA-2かFA-3かも判定）

##### `patch_attention_disable`コールバック（2350-2393行目）

```python
        @torch.compiler.disable()
        def patch_attention_disable(model):
            if "transformer_options" in model.model_options:
                if "optimized_attention_override" in model.model_options["transformer_options"]:
                    del model.model_options["transformer_options"]["optimized_attention_override"]
            
            # SAオンでもリセット時にはFAログを出力
            flash_attention_enabled = False
            try:
                if hasattr(mm, 'FLASH_IS_AVAILABLE') and mm.FLASH_IS_AVAILABLE:
                    flash_attention_enabled = True
                elif hasattr(mm, 'flash_attention_enabled'):
                    flash_attention_enabled = mm.flash_attention_enabled()
                # ... Flash-Attention状態の判定とログ出力
            except:
                pass
```

**意味**: `ON_CLEANUP`コールバックとして登録され、各モデル実行の**後**に呼び出されます。

1. **オーバーライド削除（2352-2354行目）**:
   - 実行後のクリーンアップとして、attentionオーバーライドを削除（次の実行に影響しないように）

2. **Flash-Attention状態の判定とログ出力（2356-2393行目）**:
   - `patch_attention_enable`と同様のロジックでFlash-Attentionの状態を確認
   - リセット時に元のFlash-Attention状態をログ出力
   - **重要な点**: SageAttentionが有効な状態でも、リセット時には元のFlash-Attentionの状態がログに出力される（これにより、実行のたびに現在のattention機構の状態が確認できる）

##### コールバック登録と戻り値（2395-2398行目）

```python
        model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_attention_enable)
        model_clone.add_callback(CallbacksMP.ON_CLEANUP, patch_attention_disable)
        
        return model_clone,
```

**意味**:
- `ON_PRE_RUN`: 各実行の前に`patch_attention_enable`が呼び出され、SageAttentionが適用される
- `ON_CLEANUP`: 各実行の後に`patch_attention_disable`が呼び出され、クリーンアップとログ出力が行われる
- コールバックを使用することで、**実行のたびに**ログが出力され、attention機構が正しく適用される

### 6. ノード登録（2402-2416行目）

```python
# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "MemoryManager": MemoryManager,
    "SafeMemoryManager": SafeMemoryManager,
    "DisTorchPurgeVRAMV2": DisTorchPurgeVRAMV2,
    "ModelPatchMemoryCleaner": ModelPatchMemoryCleaner,
    "PatchSageAttentionDM": PatchSageAttentionDM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryManager": "Memory Manager",
    "SafeMemoryManager": "Safe Memory Manager",
    "DisTorchPurgeVRAMV2": "LayerUtility: Purge VRAM V2",
    "ModelPatchMemoryCleaner": "Model Patch Memory Cleaner",
    "PatchSageAttentionDM": "Patch Sage Attention DM",
}
```

**意味**: ComfyUIに新しいノードを登録
- `NODE_CLASS_MAPPINGS`: ノードの内部クラス名とクラスのマッピング
- `NODE_DISPLAY_NAME_MAPPINGS`: UI上に表示される名前のマッピング

## 実装の重要なポイント

### 1. 動的ログ出力

コールバックを使用することで、モデル実行のたびにログが出力されます。これにより：
- SageAttentionが有効化された際に、バージョン情報がログに出力される
- 実行後に元のFlash-Attention状態がログに出力される
- 実行のたびに現在のattention機構の状態が確認できる

### 2. 元実装との互換性

`comfyui-kjnodes`の`PathchSageAttentionKJ`ノードと同じ動作を再現しています：
- 同じSageAttentionモードをサポート
- 同じログフォーマットで出力
- 同じコールバック機構を使用

### 3. Flash-Attentionの判定

`model_management`モジュールから直接状態を取得：
- `FLASH_IS_AVAILABLE`: Flash-Attentionが利用可能か
- `FLASH_ATTN_VERSION`: Flash-Attentionのバージョン
- `FLASH_ATTN_TYPE`: Flash-Attentionのタイプ（FA-2/FA-3）
- `flash_attention_enabled()`: Flash-Attentionが有効化されているか

### 4. クリーンアップ処理

`ON_CLEANUP`コールバックで：
- attentionオーバーライドを削除
- 次の実行に影響しないようにする
- 元のFlash-Attention状態をログ出力

### 5. エラーハンドリング

すべての重要な処理にtry-exceptブロックを配置：
- バージョン情報の取得失敗時も動作する
- Flash-Attention判定の失敗時も動作する
- 安全にフォールバック処理が行われる

## 使用方法

1. **ComfyUIでノードを追加**: "Memory"カテゴリから"Patch Sage Attention DM"ノードを追加
2. **モデルを接続**: CheckpointLoader等からモデルを入力
3. **SageAttentionモードを選択**: ドロップダウンから使用したいモードを選択
4. **オプション設定**: `allow_compile`を有効にする場合はチェック（sageattn 2.2.0以上が必要）
5. **実行**: モデルを実行すると、コンソールにログが出力される

## ログ出力例

### SageAttention有効時
```
Patching comfy attention to use SageAttention 2.2.0+cu121torch2.3.0
```

### SageAttention無効化時（Flash-Attention有効）
```
Restoring initial comfy attention
[ComfyUI] Using FA-3 (Flash-Attention 3.0.0) direct
```

### SageAttention無効化時（Flash-Attention無効）
```
Restoring initial comfy attention
```

## 注意事項

1. **実行のたびにパッチが適用される**: コールバックを使用しているため、実行のたびにSageAttentionが適用され、実行後にはクリーンアップされる
2. **`disabled`で元に戻す**: SageAttentionを無効化するには、ノードを再度実行し、`sage_attention`を`disabled`に設定する必要がある
3. **ログは実行のたびに出力される**: モデル実行のたびにログが出力されるため、コンソール出力が増える
4. **`allow_compile`の使用**: torch.compileを有効にする場合は、sageattn 2.2.0以上が必要

## まとめ

この実装により、ComfyUI-DistorchMemoryManagerノードにSageAttentionパッチ機能が追加され、元実装（comfyui-kjnodes）と同等の機能とログ出力が提供されます。コールバック機構を使用することで、実行のたびに適切にattention機構がパッチされ、詳細なログ情報が出力されます。
