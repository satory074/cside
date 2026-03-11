"""Tests for evaluation metrics."""

import pytest

from cside.evaluation.metrics import (
    boundary_recall,
    cover_id_recall_at_k,
    evaluate_medley_detection,
    frame_level_f1,
)
from cside.data.annotation import MedleyGroundTruth, SegmentLabel


def test_frame_level_f1_perfect():
    labels = ["A", "A", "B", "B"]
    result = frame_level_f1(labels, labels)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_frame_level_f1_all_wrong():
    pred = ["A", "A"]
    true = ["B", "B"]
    result = frame_level_f1(pred, true)
    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_frame_level_f1_partial():
    pred = ["A", "B", "B", "A"]
    true = ["A", "A", "B", "B"]
    result = frame_level_f1(pred, true)
    # TP=2 (frame0=A, frame2=B), FP=2, FN=2
    assert 0.0 < result["f1"] < 1.0


def test_frame_level_f1_none_labels():
    pred = [None, "A", None]
    true = [None, "A", "B"]
    result = frame_level_f1(pred, true)
    # TP=1 (A), FP=0, FN=1 (B predicted as None)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)


def test_boundary_recall_perfect():
    pred = [10.0, 20.0, 30.0]
    true = [10.0, 20.0, 30.0]
    assert boundary_recall(pred, true) == pytest.approx(1.0)


def test_boundary_recall_within_tolerance():
    pred = [10.5]
    true = [10.0]
    assert boundary_recall(pred, true, tolerance=1.0) == pytest.approx(1.0)


def test_boundary_recall_outside_tolerance():
    pred = [12.0]
    true = [10.0]
    assert boundary_recall(pred, true, tolerance=1.0) == pytest.approx(0.0)


def test_boundary_recall_empty_true():
    assert boundary_recall([1.0, 2.0], []) == pytest.approx(1.0)


def test_cover_id_recall_at_1():
    # 2 queries; first has cover in rank-1, second does not
    ground_truth = {"q1": "work_A", "q2": "work_B", "c1": "work_A", "c2": "work_B"}
    ranked_lists = [
        ["c1", "c2"],   # query q1: cover c1 is rank 1 ✓
        ["c1", "c2"],   # query q2: cover c2 is rank 2, not rank 1 ✗
    ]
    recall = cover_id_recall_at_k(ranked_lists, ground_truth, k=1)
    assert recall == pytest.approx(0.5)


def test_evaluate_medley_detection():
    gt = MedleyGroundTruth(
        medley_id="t",
        audio_path="t.mp3",
        duration_sec=10.0,
        segments=[
            SegmentLabel(start=0.0, end=5.0, song_ids=["A"]),
            SegmentLabel(start=5.0, end=10.0, song_ids=["B"]),
        ],
    )
    pred = [
        {"start": 0.0, "end": 5.0, "song_id": "A"},
        {"start": 5.0, "end": 10.0, "song_id": "B"},
    ]
    metrics = evaluate_medley_detection(pred, gt, hop_sec=1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["boundary_recall"] == pytest.approx(1.0)
