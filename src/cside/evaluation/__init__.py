"""Evaluation metrics for cover song identification and medley detection."""

from .metrics import (
    frame_level_f1,
    boundary_recall,
    cover_id_recall_at_k,
    evaluate_medley_detection,
)

__all__ = [
    "frame_level_f1",
    "boundary_recall",
    "cover_id_recall_at_k",
    "evaluate_medley_detection",
]
