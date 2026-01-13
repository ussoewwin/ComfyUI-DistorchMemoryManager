# Flash-Attention自動ロードとバージョン検出機能の完全解説

## 改造した理由

ComfyUIの`model_management`モジュールには、Flash-AttentionやSageAttentionのバージョン情報を取得する機能が公式には存在しません。過去は、ComfyUI更新のたびに`model_management.py`を手動修正してバージョン取得機能を追加していました。

この手間をなくすため、`ComfyUI-DistorchMemoryManager`ノード内に完全に独立したバージョン取得機能とFlash-Attention自動ロード機能を実装しました。これにより、ComfyUIの更新に影響されることなく、常にバージョン情報を取得でき、`disabled`時に自動的にFlash-Attention（FA-2/FA-3）をロードできるようになりました。

## 追加・修正したファイル

**ファイル**: `sa.py`

このファイルのみを修正・追加しています。

## 実装の核心的な仕組み

### 基本的な動作フロー

1. **生成開始時（ON_PRE_RUNコールバック）**:
   - SageAttentionが有効な場合: SageAttention関数を取得し、`optimized_attention_override`に設定してパッチ適用。この時点でSageAttentionのバージョンログを出力。
   - SageAttentionが`disabled`の場合: Flash-Attentionパッケージが利用可能なら、`comfy_attention.attention_flash`を`optimized_attention_override`に設定してFlash-Attentionを直接ロード。この時点でFlash-Attentionのバージョンログを出力。

2. **生成終了時（ON_CLEANUPコールバック）**:
   - `optimized_attention_override`を削除して、ComfyUIのカーネルリセットを実行。
   - ComfyUIは自動的に最適なカーネル（この場合はFlash-Attention）を初期状態として選択。
   - この時点でFlash-Attentionのバージョンログを出力（SageAttentionが有効な場合でも）。

### 重要なポイント

- **オプション不要**: `--use-flash-attention`などのCLIオプションを使用せずに、ノードの`disabled`設定だけでFlash-Attentionをロード可能。
- **生成ごとのログ出力**: 毎回の生成でSageAttention/Flash-Attentionのバージョン情報がログに出力される。
- **SA有効時でもFAログ**: SageAttentionが有効な状態でも、生成終了時のクリーンアップでFlash-Attentionが初期状態として自動選択され、そのログが出力される。

## 追加したコード内容とその意味

### 1. Flash-Attentionバージョン検出関数（22-67行目）

```python
def get_flash_attention_info():
    """
    Get Flash-Attention version and type information.
    Returns: (is_available, version, type)
    """
```

**意味**: Flash-Attentionパッケージが利用可能かどうか、バージョン番号、FA-2/FA-3の判定を取得する関数。`model_management`に完全に独立。

**処理の流れ**:
1. `import flash_attn`でパッケージのインポート可能性を確認。成功すれば`flash_is_available = True`。
2. `flash_attn.__version__`からバージョン文字列を取得を試みる。
3. `__version__`属性が存在しない場合、`importlib.metadata.version("flash-attn")`でバージョンを取得。
4. バージョン文字列を`split('.')`で分割し、先頭の数字（major version）を取得。
5. `major_version >= 3`なら`flash_attn_type = "FA-3"`、それ以外なら`"FA-2"`を設定。
6. 戻り値として`(is_available, version, type)`のタプルを返す。

**重要な点**: `args.use_flash_attention`の値に関係なく、パッケージのインポート可能性のみで判断します。これにより、実際にFlash-Attentionが利用可能な状態を正確に検出できます。

### 2. SageAttentionバージョン検出関数（70-99行目）

```python
def get_sage_attention_info():
    """
    Get SageAttention version information.
    Returns: (version, cuda_version, torch_version)
    """
```

**意味**: SageAttentionのバージョン情報、CUDAバージョン、PyTorchバージョンを取得する関数。`model_management`に完全に独立。

**処理の流れ**:
1. `import sageattention`でパッケージをインポート。
2. `sageattention.__version__`からバージョンを取得を試みる。
3. `__version__`が存在しない場合、`importlib.metadata.version("sageattention")`でバージョンを取得。
4. `torch.version.cuda`からCUDAバージョンを取得。
5. `torch.version.__version__`からPyTorchバージョンを取得。
6. 戻り値として`(version, cuda_version, torch_version)`のタプルを返す。

### 3. Flash-Attention有効判定関数（102-122行目）

```python
def is_flash_attention_enabled():
    """
    Check if Flash-Attention is currently enabled.
    Returns: bool
    """
```

**意味**: Flash-Attentionが現在実際に使用されているかを判定する関数。ただし、この実装では主に`patch_attention_disable`で使用されますが、実際のロード制御には使用されていません（`disabled`時は常にロードを試みるため）。

**処理の流れ**:
1. `comfy_attention.optimized_attention.__name__`が`"attention_flash"`かどうかを確認。
2. 実際に使用されていない場合、`get_flash_attention_info()`でFlash-Attentionが利用可能かを確認し、かつ`args.use_flash_attention`がTrueかどうかを確認。
3. 戻り値として`bool`を返す。

### 4. `get_sage_func_dm()`関数の修正（124-201行目）

**変更点**: `get_sage_attention_info()`を使用するように変更。

```python
def get_sage_func_dm(sage_attention, allow_compile=False):
    # Detect SageAttention version using our own function
    sage_version, cuda_version, torch_version = get_sage_attention_info()
    
    if sage_version and sage_version != "unknown":
        if cuda_version != "unknown" and torch_version != "unknown":
            logging.info(f"Patching comfy attention to use SageAttention {sage_version}+cu{cuda_version}torch{torch_version}")
        else:
            logging.info(f"Patching comfy attention to use SageAttention {sage_version}")
    else:
        logging.info("Patching comfy attention to use sageattn")
```

**意味**: SageAttentionのバージョン取得処理を、独立した`get_sage_attention_info()`関数を使用するように変更。コードの重複を排除し、保守性を向上。

### 5. `patch_attention_enable()`コールバックの修正（224-262行目）

**重要な変更**: `disabled`時に、Flash-Attentionを実際にロードするように変更。

```python
@torch.compiler.disable()
def patch_attention_enable(model):
    if sage_attention != "disabled":
        # SageAttention有効時の処理（従来通り）
        new_attention = get_sage_func_dm(sage_attention, allow_compile=allow_compile)
        def attention_override_sage(func, *args, **kwargs):
            return new_attention.__wrapped__(*args, **kwargs)
        
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
    else:
        # disabled: Flash-Attentionを実際にロード（オプション不要）
        if "transformer_options" in model.model_options:
            if "optimized_attention_override" in model.model_options["transformer_options"]:
                del model.model_options["transformer_options"]["optimized_attention_override"]
        
        # Flash-Attention情報を取得
        flash_is_available, flash_attn_version, flash_attn_type = get_flash_attention_info()
        
        if flash_is_available and hasattr(comfy_attention, 'attention_flash'):
            # Flash-Attentionをoverrideとして設定
            if "transformer_options" not in model.model_options:
                model.model_options["transformer_options"] = {}
            
            def attention_override_flash(func, *args, **kwargs):
                return comfy_attention.attention_flash(*args, **kwargs)
            model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_flash
            
            # Flash-Attentionバージョンログを出力（SAと同じ形式）
            logging.info("Restoring initial comfy attention")
            if flash_attn_version and flash_attn_version != "unknown":
                if flash_attn_type:
                    logging.info(f"[ComfyUI] Using {flash_attn_type} (Flash-Attention {flash_attn_version}) direct")
                else:
                    logging.info(f"[ComfyUI] Using Flash-Attention {flash_attn_version} direct")
            else:
                logging.info("[ComfyUI] Using Flash-Attention direct")
        else:
            logging.info("Restoring initial comfy attention")
```

**意味**: `ON_PRE_RUN`コールバックで実行される関数。SageAttentionが`disabled`の場合、Flash-Attentionパッケージが利用可能であれば、`comfy_attention.attention_flash`を`optimized_attention_override`に設定してFlash-Attentionを直接ロードします。この時点でFlash-Attentionのバージョンログを出力します。

**重要な点**: `args.use_flash_attention`の値に関係なく、パッケージのインポート可能性と`comfy_attention.attention_flash`の存在のみで判断します。これにより、CLIオプションを使用せずにFlash-Attentionをロードできます。

### 6. `patch_attention_disable()`コールバックの修正（264-285行目）

**重要な変更**: SageAttentionが有効な場合でも、Flash-Attentionのログを出力するように変更。

```python
@torch.compiler.disable()
def patch_attention_disable(model):
    # オーバーライドを削除してComfyUIのカーネルリセットを実行
    if "transformer_options" in model.model_options:
        if "optimized_attention_override" in model.model_options["transformer_options"]:
            del model.model_options["transformer_options"]["optimized_attention_override"]
    
    # SA有効時でも、リセット時にFAログを出力（ComfyUIのカーネルリセットによりFAが初期状態として選択される）
    flash_is_available, flash_attn_version, flash_attn_type = get_flash_attention_info()
    
    if flash_is_available:
        logging.info("Restoring initial comfy attention")
        # Flash-Attentionバージョンログを出力（SAと同じ形式）
        if flash_attn_version and flash_attn_version != "unknown":
            if flash_attn_type:
                logging.info(f"[ComfyUI] Using {flash_attn_type} (Flash-Attention {flash_attn_version}) direct")
            else:
                logging.info(f"[ComfyUI] Using Flash-Attention {flash_attn_version} direct")
        else:
            logging.info("[ComfyUI] Using Flash-Attention direct")
    else:
        logging.info("Restoring initial comfy attention")
```

**意味**: `ON_CLEANUP`コールバックで実行される関数。モデル実行後、`optimized_attention_override`を削除して、ComfyUIのカーネルリセットを実行します。ComfyUIは自動的に最適なカーネル（この場合はFlash-Attention）を初期状態として選択します。この時点でFlash-Attentionのバージョンログを出力します（SageAttentionが有効な場合でも）。

**重要な点**: SageAttentionが有効な状態でも、生成終了時のクリーンアップでComfyUIのカーネルリセットが発生し、その際にFlash-Attentionが初期状態として自動選択されます。そのため、このログは毎回の生成で出力されます。

## Flash-Attentionロードの仕組み

### `optimized_attention_override`によるロード制御

ComfyUIのモデルパッチングシステムでは、`model.model_options["transformer_options"]["optimized_attention_override"]`に関数を設定することで、デフォルトのattention選択ロジックをバイパスして、指定したattention関数を直接使用できます。

この実装では、`disabled`時に以下のようにFlash-Attentionをロードします:

```python
def attention_override_flash(func, *args, **kwargs):
    return comfy_attention.attention_flash(*args, **kwargs)
model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_flash
```

これにより、`comfy.ldm.modules.attention.optimized_attention`が呼び出された際に、`attention_flash`関数が直接実行されます。

### `attention_flash`関数の実装

`attention_flash`関数は`ComfyUI/comfy/ldm/modules/attention.py`の680行目で定義されており、内部で`flash_attn_wrapper`（`flash_attn_func`）を呼び出してFlash-Attentionを実行します。

## バージョン取得の仕組み

### Flash-Attentionバージョン取得

1. **直接インポート**: `import flash_attn`でパッケージをインポート。
2. **バージョン属性**: `flash_attn.__version__`からバージョン文字列を取得。
3. **フォールバック**: `__version__`が存在しない場合、`importlib.metadata.version("flash-attn")`で取得。
4. **FA-2/FA-3判定**: バージョン文字列を`split('.')`で分割し、先頭の数字（major version）で判定。
   - `major_version >= 3` → "FA-3"
   - それ以外 → "FA-2"

### SageAttentionバージョン取得

1. **直接インポート**: `import sageattention`でパッケージをインポート。
2. **バージョン属性**: `sageattention.__version__`からバージョン文字列を取得。
3. **フォールバック**: `__version__`が存在しない場合、`importlib.metadata.version("sageattention")`で取得。
4. **CUDA/PyTorchバージョン**: `torch.version.cuda`と`torch.version.__version__`から取得。

## 実装の重要なポイント

### 1. 完全な独立性

`model_management`モジュールへの依存を完全に排除しました。これにより、ComfyUIの更新に影響されることなく、常にバージョン情報を取得できます。

### 2. オプション不要のロード

`--use-flash-attention`などのCLIオプションを使用せずに、ノードの`disabled`設定だけでFlash-Attentionをロードできます。パッケージのインポート可能性のみで判断します。

### 3. 生成ごとのログ出力

- **ON_PRE_RUN（生成前）**: SageAttentionが有効な場合はSageAttentionのログ、`disabled`の場合はFlash-Attentionのログを出力。
- **ON_CLEANUP（生成後）**: 常にFlash-Attentionの状態を確認してログ出力（SageAttentionが有効な場合でも、ComfyUIのカーネルリセットによりFlash-Attentionが初期状態として選択されるため）。

### 4. エラーハンドリング

すべての重要な処理にtry-exceptブロックを配置し、エラーが発生しても安全に動作するようにしています。

### 5. カーネルリセットの理解

ComfyUIは生成ごとにカーネルリセットが発生し、初期状態（最適なカーネル）に戻ります。この初期状態は、Flash-Attentionが利用可能な場合は自動的にFlash-Attentionが選択されます。そのため、SageAttentionが有効な状態でも、生成終了時のクリーンアップでFlash-Attentionのログが出力されます。

## まとめ

この実装により、`ComfyUI-DistorchMemoryManager`ノード内に完全に独立したバージョン取得機能とFlash-Attention自動ロード機能が追加されました。`model_management`への依存を完全に排除し、ComfyUIの更新に影響されることなく、常にFlash-AttentionとSageAttentionのバージョン情報を取得できるようになりました。また、`disabled`時に自動的にFlash-Attention（FA-2/FA-3）をロードし、CLIオプションを使用せずに動作します。これにより、ComfyUI更新のたびに手動で`model_management.py`を修正する手間がなくなりました。
