"""Tests for boundary detection and medley segmentation."""

import numpy as np
import pytest

from cside.segmentation.boundary import BoundaryDetector


def _make_embeddings_with_shift(n_frames: int = 60, dim: int = 32, shift_at: int = 30) -> np.ndarray:
    """Create synthetic frame embeddings with an abrupt change at `shift_at`."""
    rng = np.random.default_rng(42)
    emb = np.zeros((n_frames, dim), dtype=np.float32)
    emb[:shift_at] = rng.standard_normal((shift_at, dim)).astype(np.float32)
    emb[shift_at:] = rng.standard_normal((n_frames - shift_at, dim)).astype(np.float32) + 5.0
    return emb


def test_boundary_detector_finds_shift():
    emb = _make_embeddings_with_shift(n_frames=60, shift_at=30)
    detector = BoundaryDetector(
        context_frames=4,
        min_interval_sec=2.0,
        hop_sec=1.0,
        threshold_percentile=60.0,
    )
    boundaries = detector.detect(emb)

    # At least one boundary should be near frame 30 (±3s)
    assert any(abs(b - 30.0) <= 3.0 for b in boundaries), (
        f"Expected boundary near 30s, got: {boundaries}"
    )


def test_boundary_detector_too_few_frames():
    emb = np.random.randn(3, 32).astype(np.float32)
    detector = BoundaryDetector(context_frames=8, hop_sec=1.0)
    boundaries = detector.detect(emb)
    assert boundaries == []


def test_boundary_detector_returns_sorted():
    emb = _make_embeddings_with_shift(n_frames=120, shift_at=60)
    detector = BoundaryDetector(context_frames=4, hop_sec=1.0)
    boundaries = detector.detect(emb)
    assert boundaries == sorted(boundaries)
