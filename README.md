# 🎵 CSIDE — Cover Song IDEntification

**メドレー音声を渡すだけで「どこで何の曲が流れているか」を自動検出するシステム**

MuQ（音楽 AI）による埋め込み + FAISS 近似最近傍検索を組み合わせた 2 段階パイプラインで、
DJ セット・カバーメドレー・ライブ録音などの曲区間を識別します。

---

## デモ（Web アプリ）

```bash
uv sync --extra web
uv run python launch_webapp.py
# → http://localhost:7860
```

**Algorithm Walkthrough タブ**でアルゴリズムの仕組みをインタラクティブに体験できます（音声・GPU 不要）。

---

## アーキテクチャ

```
音声ファイル
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 0: MuQExtractor              │
│  音声を 1 秒ごとに 1024 次元ベクトル  │
│  に変換（曲の「声紋」）              │
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │  Stage 1            │    Stage 2
    │  BoundaryDetector   │──▶ EmbeddingIndex (FAISS HNSW)
    │  ノベルティ曲線で    │    各セグメントを候補曲と照合
    │  曲の変わり目を検出  │
    └─────────────────────┘
               │
               ▼
    セグメントごとの曲 ID + スコア
    例: 0〜30s: song_A (0.95), 30〜55s: song_B (0.88), ...
```

### 主要モジュール

| モジュール | 役割 |
|---|---|
| `src/cside/features/muq.py` | MuQExtractor: 音声 → 1024 次元埋め込み |
| `src/cside/features/pipeline.py` | HDF5 キャッシュ付きパイプライン |
| `src/cside/retrieval/index.py` | EmbeddingIndex: FAISS HNSW インデックス |
| `src/cside/retrieval/cover_id.py` | CoverIdentifier: 特徴抽出 + 検索の統合 |
| `src/cside/segmentation/boundary.py` | BoundaryDetector: ノベルティ曲線で境界検出 |
| `src/cside/segmentation/medley.py` | MedleyDetector: 未知曲・マッシュアップ判定含む |
| `src/cside/evaluation/metrics.py` | F1 / Boundary Recall などの評価指標 |
| `webapp/` | Gradio Web アプリ |

---

## インストール

**Python 3.12 以上、[uv](https://docs.astral.sh/uv/) が必要です。**

```bash
git clone https://github.com/satory074/cside.git
cd cside

uv sync              # コア依存関係をインストール
uv sync --extra web  # Web アプリも使う場合
```

> **macOS の注意**: `torch` と `faiss-cpu` が同じ `libomp.dylib` を持つため、
> `KMP_DUPLICATE_LIB_OK=TRUE` が自動で設定されます。

---

## 使い方

### CLI

```bash
# 1. 候補曲のインデックスを作成
cside-index --input /path/to/songs/ --output data/indices/my_index

# 2. メドレーを解析
cside-medley --medley medley.mp3 --index data/indices/my_index --output results.json

# 3. 評価（アノテーションがある場合）
cside-evaluate --result results.json --ground-truth data/annotations/example_medley.json
```

#### 全 CLI コマンド

| コマンド | 説明 |
|---|---|
| `cside-extract` | 音声ディレクトリから MuQ 埋め込みを抽出して HDF5 にキャッシュ |
| `cside-index` | 音声 or HDF5 から FAISS インデックスを構築 |
| `cside-identify` | クエリ音声と最も似た曲を検索 |
| `cside-medley` | メドレー全体の区間を検出 |
| `cside-evaluate` | 検出結果をアノテーションと比較してスコアを出力 |

### Python API

```python
from cside.retrieval.cover_id import CoverIdentifier
from cside.segmentation.medley import MedleyDetector

# インデックスを読み込み
identifier = CoverIdentifier(index_prefix="data/indices/my_index")

# メドレーを解析
detector = MedleyDetector(identifier)
segments = detector.detect("medley.mp3")

for seg in segments:
    print(f"{seg['start']:.1f}s – {seg['end']:.1f}s : {seg['song_id']} (score={seg['score']:.3f})")
```

---

## セグメント結果のスキーマ

```python
{
    "start": float,          # 開始時刻（秒）
    "end": float,            # 終了時刻（秒）
    "song_id": str | None,   # 最も似た曲の ID（未知なら None）
    "score": float,          # コサイン類似度（0〜1）
    "is_mashup": bool,       # 複数曲の同時演奏かどうか
    "mashup_songs": list[str],
    "is_unknown": bool,      # データベースにない曲かどうか
    "score_ratio": float | None  # best / mean(top-k)、未知曲判定に使用
}
```

---

## 未知曲・マッシュアップの検出

FAISS は常に「最も近い曲」を返しますが、以下の 2 条件で **未知曲** を判定します（OR 条件）：

| 条件 | 意味 |
|---|---|
| `best_score < unknown_threshold` | スコア自体が低い → 類似する登録曲がない |
| `score_ratio < score_ratio_threshold` | 全候補のスコアが横並び → ダントツの 1 位がない |

**マッシュアップ**（2 曲同時演奏）は 2 位スコアが `best × mashup_threshold` 以上のとき検出。

---

## 開発

```bash
uv run pytest --no-cov          # テスト実行（32 テスト、約 2 秒）
uv run ruff check src/ tests/   # Lint
uv run black src/ tests/        # フォーマット
```

---

## 技術スタック

- **音楽 AI モデル**: [MuQ (OpenMuQ/MuQ-large-msd-iter)](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) — 1024 次元埋め込み
- **近似最近傍検索**: [FAISS](https://github.com/facebookresearch/faiss) — IndexHNSWFlat (O(log N))
- **音声処理**: [librosa](https://librosa.org/), [torchaudio](https://pytorch.org/audio/)
- **Web UI**: [Gradio](https://www.gradio.app/) 6.0+
- **パッケージ管理**: [uv](https://docs.astral.sh/uv/)

---

## デプロイ

Gradio アプリは **Hugging Face Spaces** へのデプロイを推奨します（無料・Gradio ネイティブサポート）。

```bash
# Spaces 用の設定ファイルを追加して push するだけ
# 詳細: https://huggingface.co/docs/hub/spaces-sdks-gradio
```

---

## ライセンス

MIT
