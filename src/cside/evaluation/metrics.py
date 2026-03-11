"""Evaluation metrics for cover-song identification and medley detection.

Metrics
-------
cover_id_recall_at_k
    Standard Recall@K for cover-song retrieval (MIREX style).

frame_level_f1
    Frame-level Precision / Recall / F1 for medley detection.

boundary_recall
    Proportion of ground-truth boundaries detected within ±tolerance seconds.

evaluate_medley_detection
    End-to-end evaluation against a MedleyGroundTruth object.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Cover-song identification
# ---------------------------------------------------------------------------


def cover_id_recall_at_k(
    ranked_lists: list[list[str]],
    ground_truth: dict[str, str],
    k: int = 1,
) -> float:
    """Compute Recall@K for cover-song retrieval.

    Parameters
    ----------
    ranked_lists:
        For each query track, a list of retrieved track IDs ordered by score
        (most similar first).
    ground_truth:
        Mapping from query track_id → work_id (cluster).  All tracks sharing
        the same work_id are considered covers of each other.
    k:
        Cutoff rank.

    Returns
    -------
    float: proportion of queries where at least one cover appears in top-k.
    """
    if not ranked_lists:
        return 0.0

    # Build reverse mapping: work_id → set of track_ids
    work_to_tracks: dict[str, set[str]] = {}
    for track_id, work_id in ground_truth.items():
        work_to_tracks.setdefault(work_id, set()).add(track_id)

    hits = 0
    for query_id, ranked in zip(ground_truth.keys(), ranked_lists):
        work_id = ground_truth[query_id]
        cover_set = work_to_tracks[work_id] - {query_id}  # exclude query itself
        top_k = set(ranked[:k])
        if top_k & cover_set:
            hits += 1

    return hits / len(ranked_lists)


# ---------------------------------------------------------------------------
# Medley / segment detection
# ---------------------------------------------------------------------------


def frame_level_f1(
    pred_labels: list[str | None],
    true_labels: list[str | None],
) -> dict[str, float]:
    """Compute frame-level Precision, Recall, F1 across all frames.

    Parameters
    ----------
    pred_labels:
        Predicted song_id per frame (None = no assignment).
    true_labels:
        Ground-truth song_id per frame.

    Returns
    -------
    dict with keys ``"precision"``, ``"recall"``, ``"f1"``.
    """
    assert len(pred_labels) == len(true_labels), "Length mismatch"

    tp = sum(p == t and t is not None for p, t in zip(pred_labels, true_labels))
    fp = sum(p is not None and p != t for p, t in zip(pred_labels, true_labels))
    fn = sum(t is not None and p != t for p, t in zip(pred_labels, true_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def boundary_recall(
    pred_boundaries: list[float],
    true_boundaries: list[float],
    tolerance: float = 1.0,
) -> float:
    """Fraction of ground-truth boundaries detected within ±tolerance seconds.

    Parameters
    ----------
    pred_boundaries:
        Predicted boundary times (seconds).
    true_boundaries:
        Ground-truth boundary times (seconds).
    tolerance:
        Match window in seconds.

    Returns
    -------
    float: recall value in [0, 1].
    """
    if not true_boundaries:
        return 1.0  # nothing to recall

    hits = 0
    for tb in true_boundaries:
        for pb in pred_boundaries:
            if abs(pb - tb) <= tolerance:
                hits += 1
                break

    return hits / len(true_boundaries)


def evaluate_medley_detection(
    pred_segments: list[dict],
    ground_truth,  # MedleyGroundTruth
    hop_sec: float = 1.0,
    boundary_tolerance: float = 1.0,
) -> dict[str, float]:
    """Full medley detection evaluation.

    Parameters
    ----------
    pred_segments:
        List of dicts with keys ``start``, ``end``, ``song_id``.
    ground_truth:
        A ``MedleyGroundTruth`` instance.
    hop_sec:
        Frame resolution in seconds.
    boundary_tolerance:
        Tolerance for boundary recall (seconds).

    Returns
    -------
    dict with keys ``precision``, ``recall``, ``f1``, ``boundary_recall``.
    """
    n_frames = int(np.ceil(ground_truth.duration_sec / hop_sec))

    # Build ground-truth frame labels (first song_id or None)
    _, true_label_lists = ground_truth.to_frame_labels(hop_sec=hop_sec)
    true_labels: list[str | None] = [
        ls[0] if ls else None for ls in true_label_lists
    ]

    # Build predicted frame labels
    pred_labels: list[str | None] = [None] * n_frames
    pred_boundaries: list[float] = []
    for seg in pred_segments:
        start_idx = int(seg["start"] / hop_sec)
        end_idx = min(int(np.ceil(seg["end"] / hop_sec)), n_frames)
        for idx in range(start_idx, end_idx):
            pred_labels[idx] = seg["song_id"]
        pred_boundaries.append(seg["start"])

    frame_metrics = frame_level_f1(pred_labels, true_labels)

    # Extract true boundaries (start of each segment, skip t=0)
    true_boundaries = [seg.start for seg in ground_truth.segments if seg.start > 0]

    br = boundary_recall(pred_boundaries, true_boundaries, tolerance=boundary_tolerance)

    return {**frame_metrics, "boundary_recall": br}
