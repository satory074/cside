# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install all dependencies (core)
uv sync --extra web              # Install with Gradio webapp extras
uv run pytest --no-cov           # Run all 29 tests
uv run pytest tests/test_foo.py::test_bar --no-cov  # Run a single test
uv run ruff check src/ tests/    # Lint
uv run black src/ tests/         # Format
```

### Web app

```bash
uv run python launch_webapp.py   # → http://localhost:7860
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

Two-stage pipeline for detecting which songs appear in a medley:

```
Audio → MuQExtractor (1024-dim frame embeddings)
              ↓
        CoverIdentifier (EmbeddingIndex / FAISS)
              ↓
        MedleyDetector
          Stage 1: BoundaryDetector → segment timestamps
          Stage 2: per-segment FAISS search → song_id + mashup flag
```

### Key modules (`src/cside/`)

**`features/`**
- `muq.py` — `MuQExtractor`: wraps `OpenMuQ/MuQ-large-msd-iter` via the `muq` package (`from muq import MuQ`). Lazy-loads on first `.encode()` call. Returns `(T, 1024)` frame embeddings or `(1024,)` mean-pooled. Input is raw 24 kHz audio passed directly as a tensor — **no `AutoFeatureExtractor`** (the model has no `preprocessor_config.json`).
- `pipeline.py` — `FeaturePipeline`: HDF5 disk caching on top of `MuQExtractor`. Cache key = `sha1(path + mtime + extractor_type)`. No compression (omitted for write speed).
- `legacy.py` — `LegacyExtractor`: chroma/CQT/MFCC/peak features from the 2019 implementation, kept as a baseline.

**`retrieval/`**
- `index.py` — `EmbeddingIndex`: FAISS `IndexHNSWFlat` (O(log N) search, `hnsw_m=32`, `efSearch=64`) with L2-normalised inner product = cosine similarity. Saves as `<prefix>.faiss` + `<prefix>.json`. `add_batch()` bulk-normalizes and bulk-adds for efficiency. **Existing `.faiss` files built with `IndexFlatIP` are incompatible — rebuild with `cside-index`.**
- `cover_id.py` — `CoverIdentifier`: combines `FeaturePipeline` + `EmbeddingIndex`. Optional trainable projection head (1024 → 128 dim, InfoNCE). When `use_projection=False` or `_projection is None`, embeddings pass through unchanged.

**`segmentation/`**
- `boundary.py` — `BoundaryDetector`: cosine novelty curve (vectorized via cumulative sums, not a per-frame loop) → Gaussian-smoothed → `scipy.signal.find_peaks`. `MambaBoundaryDetector` is an unimplemented stub.
- `medley.py` — `MedleyDetector`: `detect()` uses `BoundaryDetector`; `detect_sliding_window()` uses fixed windows. Mashup flag: 2nd-best FAISS score ≥ `mashup_threshold × best_score`.

**`evaluation/`**
- `metrics.py` — `frame_level_f1`, `boundary_recall`, `cover_id_recall_at_k`, `evaluate_medley_detection`.

**`data/`**
- `annotation.py` — Pydantic models `MedleyGroundTruth` / `SegmentLabel`. `to_frame_labels()` converts to per-frame song ID lists.

### Web app (`webapp/`)

Not an installable package — lives at the project root, not inside `src/cside/`. `launch_webapp.py` adds the project root to `sys.path`.

- `app.py` — Gradio UI and event callbacks (`on_detect`, `on_mode_change`, `on_evaluate`)
- `index_manager.py` — `get_demo_identifier()` (loads pre-built index) / `build_from_uploads()` (indexes user-uploaded audio on-the-fly)
- `job_runner.py` — `run_detection()`: loads audio → extracts MuQ embeddings → runs `MedleyDetector`. Uses a single-worker `ThreadPoolExecutor` to prevent concurrent OOM.
- `visualizer.py` — `draw_timeline()`: renders `matplotlib` segment timeline as a PIL image.

### macOS note

Both `torch` and `faiss-cpu` ship their own `libomp.dylib`, causing a fatal duplicate-runtime error. `KMP_DUPLICATE_LIB_OK=TRUE` is set in both `src/cside/__init__.py` and `tests/conftest.py` — must be set **before** either library is imported.

### Testing without audio/GPU

Tests use 64-dim synthetic embeddings exclusively. To test `CoverIdentifier` without loading MuQ, instantiate with `use_projection=False` and inject embeddings directly via `identifier._index.add(track_id, emb)`.

## What's NOT Yet Implemented (Stubs)

- `MambaBoundaryDetector` (Phase 4 — requires `mamba-ssm`)
- SAM Audio source separation (Phase 2 optional)
- CLaMP 3 multimodal search (Phase 2 optional)
- Contrastive fine-tuning end-to-end (requires DA-TACOS audio)

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
