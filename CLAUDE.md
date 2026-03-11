# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install all dependencies
uv run pytest --no-cov           # Run all 29 tests
uv run pytest tests/test_foo.py::test_bar --no-cov  # Run a single test
uv run ruff check src/ tests/    # Lint
uv run black src/ tests/         # Format
```

### CLI entry points (after `uv sync`)

```bash
cside-extract  --input /audio/dir --output data/embeddings          # Extract MuQ embeddings → HDF5 cache
cside-index    --input /audio/dir --output data/indices/my_index    # Build FAISS index from audio
cside-index    --embeddings data/embeddings --output data/indices/my_index  # Build from pre-computed HDF5
cside-identify --query query.mp3 --index data/indices/my_index --k 5       # Cover song lookup
cside-medley   --medley medley.mp3 --index data/indices/my_index --output results.json  # Medley detection
cside-evaluate --result results.json --ground-truth data/annotations/example_medley.json  # Score results
```

## Architecture

The system is a two-stage pipeline for detecting which songs appear in a medley:

```
Audio → MuQExtractor (768-dim frame embeddings)
              ↓
        CoverIdentifier (EmbeddingIndex / FAISS)
              ↓
        MedleyDetector
          Stage 1: BoundaryDetector → segment timestamps
          Stage 2: per-segment FAISS search → song_id + mashup flag
```

### Key modules (`src/cside/`)

**`features/`**
- `muq.py` — `MuQExtractor`: wraps `m-a-p/MuQ-MuLan` from HuggingFace. Lazy-loads the model on first call to `.encode()`. Returns `(T, 768)` frame embeddings or `(768,)` mean-pooled.
- `pipeline.py` — `FeaturePipeline`: adds HDF5 disk caching on top of `MuQExtractor` (or legacy extractors). Cache key is `sha1(path + mtime + extractor_type)`.
- `legacy.py` — `LegacyExtractor`: chroma/CQT/MFCC/peak features from the original 2019 implementation, kept as comparison baseline.

**`retrieval/`**
- `index.py` — `EmbeddingIndex`: thin FAISS wrapper. Uses `IndexFlatIP` (exact cosine similarity via L2-normalised inner product). Saves as `<prefix>.faiss` + `<prefix>.json` (metadata).
- `cover_id.py` — `CoverIdentifier`: combines `FeaturePipeline` + `EmbeddingIndex`. Supports an optional trainable projection head (768 → 128 dim, InfoNCE contrastive loss). When `use_projection=False` or `_projection is None`, embeddings pass through unchanged.

**`segmentation/`**
- `boundary.py` — `BoundaryDetector`: cosine novelty curve (context window before/after each frame) → Gaussian-smoothed → `scipy.signal.find_peaks`. `MambaBoundaryDetector` is a stub for a future learned model.
- `medley.py` — `MedleyDetector`: orchestrates the two stages. `detect()` uses `BoundaryDetector`; `detect_sliding_window()` uses fixed windows. Mashup detection: if 2nd-best FAISS score ≥ `mashup_threshold × best_score`, the segment is flagged `is_mashup=True`.

**`evaluation/`**
- `metrics.py` — `frame_level_f1`, `boundary_recall`, `cover_id_recall_at_k`, `evaluate_medley_detection`.

**`data/`**
- `annotation.py` — Pydantic models: `MedleyGroundTruth` / `SegmentLabel`. JSON schema in module docstring. `to_frame_labels()` converts segment annotations to per-frame song ID lists.
- `datasets.py` — `load_covers80()` / `load_da_tacos()` via `mirdata`.

### macOS note
Both `torch` and `faiss-cpu` ship their own `libomp.dylib`, causing a duplicate-runtime fatal error. `KMP_DUPLICATE_LIB_OK=TRUE` is set in both `src/cside/__init__.py` and `tests/conftest.py` — this must be set before either library is imported.

### Testing without audio/GPU
Tests use 64-dim synthetic embeddings exclusively. To test `CoverIdentifier` without loading MuQ, instantiate with `use_projection=False` and inject embeddings directly via `identifier._index.add(track_id, emb)`.

### Segment result schema
```python
{
    "start": float,          # seconds
    "end": float,
    "song_id": str | None,   # top match (None if index is empty)
    "score": float,          # cosine similarity
    "is_mashup": bool,
    "mashup_songs": list[str]
}
```
