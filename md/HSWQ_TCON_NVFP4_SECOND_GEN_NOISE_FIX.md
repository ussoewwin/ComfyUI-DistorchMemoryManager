# HSWQ tcon NVFP4 2回目ノイズ & SEG キャッシュ問題 修正ガイド

**日付:** 2026-08-26
**対象:** ComfyUI-DistorchMemoryManager + ComfyUI-HSWQ-Loader-and-Tools
**症状:** tcon (Z Image TC/W4A4) NVFP4 モデルで、パージ実行後の2回目生成がノイズ化する

---

## 1. 2回目ノイズ問題 (tcon NVFP4)

### 根本原因

パージで HSWQ スタックが剥がれた後、2回目の生成で NVFP4 レイヤーが正しく LoRA ベイクされずノイズになっていた。

ログでの確証:

| 世代 | ベイク結果 | 状態 |
|------|-----------|------|
| 1回目 | `nvfp4_baked=86 int8_baked=94` / `NVFP4_LORA_BAKE_OK` | 正常 |
| 2回目 | `nvfp4_baked=0 other_qt_baked=83` / `NVFP4_LORA_BAKE_N/A` | NVFP4 が `other_qt` に誤分類され ConvRot なしで誤ベイク → ノイズ |

### 原因の連鎖

1. パージが `ops._load_quantized_module` のラップを剥がす (`_hswq_nvfp4_full_load` スタンプ消失)
2. `apply_comfy_quant_nvfp4_patches()` は `_PATCHES_APPLIED` フラグと `stack_ver` のみで早期リターン
3. 再ロード時に `arm_nvfp4_module` が走らず、モジュールに `_hswq_nvfp4_convrot` フラグが設定されない
4. ベイク関数が NVFP4 レイヤーを検出できず `other_qt` として誤ベイク (ConvRot 回転が適用されない)
5. 重みが壊れてノイズ化

### 修正 (3件)

#### `fdc60bc` — HSWQ-Loader-and-Tools (本命修正)

`nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py`

早期リターンパス (2箇所) に `_load_wrap_ok` 条件を追加:

```python
_load_wrap_ok = bool(
    getattr(ops._load_quantized_module, "_hswq_nvfp4_full_load", False)
)
if (
    _PATCHES_APPLIED
    and _load_wrap_ok          # ← 追加: ラップが剥がれていたら早期リターンしない
    and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
    and stack_ver >= _NVFP4_STACK_VER
):
    return True
```

パージでラップが剥がれていれば両方の早期リターンをスキップし、完全な再適用に進んで `_load_quantized_module` を再ラップ、モジュールを再アームする。

#### `d97bb5b` — HSWQ-Loader-and-Tools

`nodes/zimage_nvfp4/load_unet.py`

`_install_permanent_dynamic_load_guard()` を追加。`ModelPatcherDynamic.load` の外側ガードで、`_hswq_zi_nvfp4_lora_bake` スタンプを**付けない**ため、パージの deep-clean が剥がしに来ても素通りする。毎回の `Dynamic.load` で `_ensure_dynamic_load_bake_wrap()` を呼び、ベイクフックが剥がれていれば自動再アーム (アーム済みなら no-op)。

#### `2936341` — DistorchMemoryManager

`purge_vram.py`

パージ完了後に `unload_models` + `free_memory` キューフラグを設定。ComfyUI の executor がローダーノードの出力キャッシュ (MODEL オブジェクト) を破棄し、次の生成でローダーノードが再実行されて `load_unet` 内の TC (W4A4) スタックパッチが再適用されるようにした。パージ自体の完全リセット動作は無変更。

---

## 2. SEG キャッシュ問題

### `2bc0075` — DistorchMemoryManager

**「Tried to unpin tensor not pinned by ComfyUI」警告の根本修正**

原因: `mm.unpin_memory()` は ComfyUI の `PINNED_MEMORY` 辞書に登録されたテンソルのみ処理可能。Detailer/SEGS キャッシュのテンソルは未登録のため警告を出して `False` を返す (例外は投げない)。旧コードの `try/except` フォールバックは例外が出ないため発動せず、警告が出るだけでアンピンもされていなかった。

修正: 呼び出し前に `PINNED_MEMORY` 登録を確認。

- 登録済み → `mm.unpin_memory()` (ComfyUI の帳簿と同期)
- 未登録 → 直接 `cudaHostUnregister()` (警告なしで確実に解放)

警告を消すのではなく、登録状態に応じた正しい解放経路を使う根本対応。

### `f59716a` — DistorchMemoryManager

**Detailer/SEGS キャッシュの完全削除**

不要になった Detailer/SEGS キャッシュパスを除去:

- `_drain_hswq_pin_cache()` 削除
- `_purge_detailer_segs_and_executor_cache()` 削除
- 4箇所の呼び出し (Method 0 / 0s / 0b / 0s2) 削除
- 計 265 行削減

HSWQ 完全リセット (モデルアンロード、PINNED_MEMORY アンレジスタ、kitchen キャッシュ、Hadamard、Linear ベイク peel、VRAM 解放) は無変更。

---

## コミット一覧

| リポジトリ | コミット | 内容 |
|-----------|---------|------|
| HSWQ-Loader-and-Tools | `fdc60bc` | パージで `_load_quantized_module` ラップが剥がれた場合に完全スタック再適用 |
| HSWQ-Loader-and-Tools | `d97bb5b` | パージで剥がれない Dynamic.load 恒久ガードでベイクフック自動再アーム |
| DistorchMemoryManager | `2936341` | パージ後にローダーノードキャッシュをリセット (TC スタック再構築) |
| DistorchMemoryManager | `2bc0075` | `PINNED_MEMORY` 登録確認で unpin 警告を根本修正 |
| DistorchMemoryManager | `f59716a` | Detailer/SEGS キャッシュスイープ削除 |

**全て dev / installed 同期・push 済み。**
