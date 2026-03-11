"""Tests for EmbeddingIndex and CoverIdentifier (without GPU or real audio)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cside.retrieval.index import EmbeddingIndex


def _random_emb(dim: int = 768) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_index_add_and_search():
    index = EmbeddingIndex(dim=64)
    emb_a = _random_emb(64)
    emb_b = _random_emb(64)

    index.add("track_A", emb_a)
    index.add("track_B", emb_b)
    assert len(index) == 2

    results = index.search(emb_a, k=1)
    assert len(results) == 1
    assert results[0][0] == "track_A"
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_index_exclude():
    index = EmbeddingIndex(dim=64)
    emb = _random_emb(64)
    index.add("self", emb)
    index.add("other", _random_emb(64))

    results = index.search(emb, k=1, exclude={"self"})
    assert results[0][0] == "other"


def test_index_duplicate_add():
    index = EmbeddingIndex(dim=64)
    emb = _random_emb(64)
    index.add("track_A", emb)
    index.add("track_A", emb)  # should be silently ignored
    assert len(index) == 1


def test_index_empty_search():
    index = EmbeddingIndex(dim=64)
    results = index.search(_random_emb(64), k=5)
    assert results == []


def test_index_save_load(tmp_path):
    index = EmbeddingIndex(dim=64)
    emb_a = _random_emb(64)
    emb_b = _random_emb(64)
    index.add("A", emb_a)
    index.add("B", emb_b)

    prefix = tmp_path / "test_index"
    index.save(prefix)

    loaded = EmbeddingIndex.load(prefix)
    assert len(loaded) == 2

    results = loaded.search(emb_a, k=1)
    assert results[0][0] == "A"
