"""Integration tests for MedleyDetector (no audio, no GPU required)."""

import numpy as np
import pytest

from cside.retrieval.cover_id import CoverIdentifier
from cside.retrieval.index import EmbeddingIndex
from cside.segmentation.boundary import BoundaryDetector
from cside.segmentation.medley import MedleyDetector

DIM = 64  # small dim for fast tests


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_identifier(song_embeddings: dict[str, np.ndarray]) -> CoverIdentifier:
    """Build a CoverIdentifier backed by synthetic embeddings (no model loading)."""
    identifier = CoverIdentifier(index_dim=DIM, use_projection=False)
    # Bypass FeaturePipeline: inject embeddings directly into the index
    for tid, emb in song_embeddings.items():
        identifier._index.add(tid, emb)
    return identifier


def _make_medley_embeddings(
    song_embeddings: dict[str, np.ndarray],
    segment_song_ids: list[str],
    frames_per_segment: int = 10,
    noise: float = 0.05,
    rng_seed: int = 0,
) -> np.ndarray:
    """Create synthetic medley frame embeddings by concatenating noisy song vectors."""
    rng = np.random.default_rng(rng_seed)
    frames = []
    for song_id in segment_song_ids:
        base = song_embeddings[song_id]
        for _ in range(frames_per_segment):
            noisy = base + rng.standard_normal(DIM).astype(np.float32) * noise
            frames.append(_unit(noisy))
    return np.stack(frames)  # (T, DIM)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def song_embs():
    """Four orthogonal-ish song embeddings."""
    rng = np.random.default_rng(42)
    return {
        "song_A": _unit(rng.standard_normal(DIM).astype(np.float32)),
        "song_B": _unit(rng.standard_normal(DIM).astype(np.float32)),
        "song_C": _unit(rng.standard_normal(DIM).astype(np.float32)),
        "song_D": _unit(rng.standard_normal(DIM).astype(np.float32)),
    }


@pytest.fixture
def identifier(song_embs):
    return _make_identifier(song_embs)


# ---------------------------------------------------------------------------
# Sliding-window detection tests
# ---------------------------------------------------------------------------


def test_sliding_window_single_song(song_embs, identifier):
    """All frames from one song → single segment assigned to that song."""
    medley = _make_medley_embeddings(song_embs, ["song_A"] * 3, frames_per_segment=10)

    detector = MedleyDetector(identifier, hop_sec=1.0, mashup_threshold=0.85, top_k=4)
    segments = detector.detect_sliding_window(medley, window_sec=5.0, stride_sec=5.0)

    assert all(s["song_id"] == "song_A" for s in segments), (
        f"Expected all segments = song_A, got: {[s['song_id'] for s in segments]}"
    )


def test_sliding_window_two_songs(song_embs, identifier):
    """Two distinct song blocks → two merged segments."""
    medley = _make_medley_embeddings(
        song_embs, ["song_A"] * 3 + ["song_B"] * 3, frames_per_segment=10
    )

    detector = MedleyDetector(identifier, hop_sec=1.0, mashup_threshold=0.85, top_k=4)
    segments = detector.detect_sliding_window(medley, window_sec=10.0, stride_sec=10.0)

    song_ids = [s["song_id"] for s in segments]
    assert "song_A" in song_ids
    assert "song_B" in song_ids


def test_sliding_window_merge_consecutive(song_embs, identifier):
    """Consecutive windows with same song should be merged."""
    medley = _make_medley_embeddings(song_embs, ["song_C"] * 6, frames_per_segment=5)

    detector = MedleyDetector(identifier, hop_sec=1.0, top_k=4)
    segments = detector.detect_sliding_window(medley, window_sec=5.0, stride_sec=5.0)

    # After merging, should be a single segment
    assert len(segments) == 1
    assert segments[0]["song_id"] == "song_C"


def test_sliding_window_segment_scores_positive(song_embs, identifier):
    medley = _make_medley_embeddings(song_embs, ["song_D"] * 2, frames_per_segment=10)
    detector = MedleyDetector(identifier, hop_sec=1.0, top_k=4)
    segments = detector.detect_sliding_window(medley, window_sec=10.0, stride_sec=10.0)
    for seg in segments:
        assert seg["score"] > 0.0


# ---------------------------------------------------------------------------
# Boundary-based detection tests
# ---------------------------------------------------------------------------


def test_boundary_detect_returns_segments(song_embs, identifier):
    """Boundary-based detection should return non-empty list for multi-song medley."""
    medley = _make_medley_embeddings(
        song_embs, ["song_A"] * 2 + ["song_B"] * 2, frames_per_segment=20
    )
    boundary_detector = BoundaryDetector(
        context_frames=5,
        min_interval_sec=4.0,
        hop_sec=1.0,
        threshold_percentile=60.0,
    )
    detector = MedleyDetector(
        identifier,
        boundary_detector=boundary_detector,
        hop_sec=1.0,
        top_k=4,
    )
    segments = detector.detect(medley)

    assert len(segments) >= 1
    for seg in segments:
        assert seg["start"] < seg["end"]
        assert "song_id" in seg
        assert "score" in seg
        assert "is_mashup" in seg


def test_segment_schema(song_embs, identifier):
    """Every segment must have the full expected schema."""
    medley = _make_medley_embeddings(song_embs, ["song_A"] * 2, frames_per_segment=10)
    detector = MedleyDetector(identifier, hop_sec=1.0, top_k=4)
    segments = detector.detect_sliding_window(medley, window_sec=10.0, stride_sec=10.0)

    required_keys = {"start", "end", "song_id", "score", "is_mashup", "mashup_songs"}
    for seg in segments:
        assert required_keys.issubset(seg.keys())
        assert isinstance(seg["is_mashup"], bool)
        assert isinstance(seg["mashup_songs"], list)


def test_empty_index_returns_none_song_id():
    """When the index is empty, song_id should be None."""
    empty_identifier = CoverIdentifier(index_dim=DIM, use_projection=False)
    detector = MedleyDetector(empty_identifier, hop_sec=1.0)
    medley = np.random.randn(10, DIM).astype(np.float32)
    segments = detector.detect_sliding_window(medley, window_sec=5.0, stride_sec=5.0)
    assert all(s["song_id"] is None for s in segments)
