"""Tests for annotation schema and frame label conversion."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cside.data.annotation import MedleyGroundTruth, SegmentLabel


def _make_gt() -> MedleyGroundTruth:
    return MedleyGroundTruth(
        medley_id="test_001",
        audio_path="test.mp3",
        duration_sec=60.0,
        segments=[
            SegmentLabel(start=0.0, end=20.0, song_ids=["song_A"]),
            SegmentLabel(start=20.0, end=35.0, song_ids=["song_B", "song_C"], is_mashup=True),
            SegmentLabel(start=35.0, end=60.0, song_ids=["song_D"]),
        ],
    )


def test_segment_label_duration():
    seg = SegmentLabel(start=10.0, end=25.0, song_ids=["x"])
    assert seg.duration == pytest.approx(15.0)


def test_primary_song_id():
    seg = SegmentLabel(start=0.0, end=5.0, song_ids=["a", "b"])
    assert seg.primary_song_id == "a"


def test_to_frame_labels():
    gt = _make_gt()
    times, labels = gt.to_frame_labels(hop_sec=1.0)

    assert len(times) == 60
    # Frame 0 → song_A
    assert labels[0] == ["song_A"]
    # Frame 20 → song_B + song_C (mashup)
    assert set(labels[20]) == {"song_B", "song_C"}
    # Frame 50 → song_D
    assert labels[50] == ["song_D"]


def test_round_trip_json():
    gt = _make_gt()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        tmp = Path(f.name)
    try:
        gt.save(tmp)
        loaded = MedleyGroundTruth.from_json(tmp)
        assert loaded.medley_id == gt.medley_id
        assert len(loaded.segments) == len(gt.segments)
        assert loaded.segments[1].is_mashup is True
    finally:
        tmp.unlink(missing_ok=True)
