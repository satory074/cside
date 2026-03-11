"""Medley fragment detection.

Two-stage pipeline
------------------
Stage 1 — Boundary detection (``BoundaryDetector``)
    Splits the medley audio into segments at points of high novelty.

Stage 2 — Per-segment cover identification (``CoverIdentifier`` / FAISS)
    For each segment, retrieve the most similar candidate track.
    If multiple candidates score above a threshold → mashup region.

The detector accepts frame-level MuQ embeddings (T × 768) of the medley
and a pre-built ``CoverIdentifier`` index of candidate tracks.

Result format
-------------
Each detected segment is a dict::

    {
        "start": float,           # segment start (seconds)
        "end": float,             # segment end (seconds)
        "song_id": str,           # top candidate track ID
        "score": float,           # similarity score
        "is_mashup": bool,        # True when ≥2 candidates score above threshold
        "mashup_songs": list[str] # all candidates above threshold (mashup case)
    }
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .boundary import BoundaryDetector
from ..retrieval.cover_id import CoverIdentifier

logger = logging.getLogger(__name__)


class MedleyDetector:
    """Detect which candidate tracks appear in which segments of a medley.

    Parameters
    ----------
    cover_identifier:
        A pre-built ``CoverIdentifier`` with candidate tracks already indexed.
    boundary_detector:
        Boundary detector instance.  Defaults to ``BoundaryDetector()``.
    hop_sec:
        Duration of one embedding frame (seconds).
    mashup_threshold:
        If the second-best candidate's score exceeds
        ``best_score * mashup_threshold``, the segment is flagged as a mashup.
        E.g. 0.85 means the 2nd candidate must be ≥85% as similar as the best.
    top_k:
        Number of candidates to retrieve per segment.
    """

    def __init__(
        self,
        cover_identifier: CoverIdentifier,
        boundary_detector: Optional[BoundaryDetector] = None,
        hop_sec: float = 1.0,
        mashup_threshold: float = 0.85,
        top_k: int = 5,
    ) -> None:
        self.cover_identifier = cover_identifier
        self.boundary_detector = boundary_detector or BoundaryDetector(hop_sec=hop_sec)
        self.hop_sec = hop_sec
        self.mashup_threshold = mashup_threshold
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, medley_embeddings: np.ndarray) -> list[dict]:
        """Run the full two-stage medley detection pipeline.

        Parameters
        ----------
        medley_embeddings:
            Frame-level embeddings of the medley, shape ``(T, D)``.

        Returns
        -------
        list of segment dicts (see module docstring for schema).
        """
        T = len(medley_embeddings)
        total_duration = T * self.hop_sec

        # Stage 1: boundary detection
        boundaries = self.boundary_detector.detect(medley_embeddings)
        # Include start (0) and end (duration) as implicit boundaries
        boundary_times = sorted(set([0.0] + boundaries + [total_duration]))
        logger.info("Medley split into %d segments", len(boundary_times) - 1)

        # Stage 2: per-segment identification
        segments = []
        for start_t, end_t in zip(boundary_times[:-1], boundary_times[1:]):
            seg_emb = self._segment_embedding(medley_embeddings, start_t, end_t)
            result = self._identify_segment(seg_emb, start_t, end_t)
            segments.append(result)

        return segments

    def detect_sliding_window(
        self,
        medley_embeddings: np.ndarray,
        window_sec: float = 16.0,
        stride_sec: float = 4.0,
    ) -> list[dict]:
        """Alternative: fixed sliding-window detection (no boundary detector).

        Useful when the medley structure is unknown and boundaries cannot
        be detected reliably.

        Parameters
        ----------
        window_sec:
            Window duration in seconds (≈ 4 bars at 120 BPM).
        stride_sec:
            Stride between consecutive windows.
        """
        T = len(medley_embeddings)
        window_frames = max(1, int(window_sec / self.hop_sec))
        stride_frames = max(1, int(stride_sec / self.hop_sec))

        segments = []
        frame = 0
        while frame < T:
            end_frame = min(frame + window_frames, T)
            start_t = frame * self.hop_sec
            end_t = end_frame * self.hop_sec

            seg_emb = medley_embeddings[frame:end_frame].mean(axis=0)
            result = self._identify_segment(seg_emb, start_t, end_t)
            segments.append(result)
            frame += stride_frames

        return self._merge_consecutive(segments)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _segment_embedding(
        self, embeddings: np.ndarray, start_t: float, end_t: float
    ) -> np.ndarray:
        """Mean-pool embeddings within a time range."""
        start_f = int(start_t / self.hop_sec)
        end_f = min(int(np.ceil(end_t / self.hop_sec)), len(embeddings))
        chunk = embeddings[start_f:end_f]
        if len(chunk) == 0:
            return np.zeros(embeddings.shape[1])
        return chunk.mean(axis=0)

    def _identify_segment(
        self, seg_emb: np.ndarray, start_t: float, end_t: float
    ) -> dict:
        """Retrieve candidates for a single segment embedding."""
        results = self.cover_identifier.identify_embedding(seg_emb, k=self.top_k)

        if not results:
            return {
                "start": start_t,
                "end": end_t,
                "song_id": None,
                "score": 0.0,
                "is_mashup": False,
                "mashup_songs": [],
            }

        best_id, best_score = results[0]

        # Mashup detection: flag if second candidate is close to best
        mashup_songs = [best_id]
        is_mashup = False
        if len(results) > 1:
            for other_id, other_score in results[1:]:
                if other_score >= best_score * self.mashup_threshold:
                    mashup_songs.append(other_id)
                    is_mashup = True

        return {
            "start": start_t,
            "end": end_t,
            "song_id": best_id,
            "score": float(best_score),
            "is_mashup": is_mashup,
            "mashup_songs": mashup_songs,
        }

    @staticmethod
    def _merge_consecutive(segments: list[dict]) -> list[dict]:
        """Merge adjacent windows assigned to the same song."""
        if not segments:
            return []

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            last = merged[-1]
            if seg["song_id"] == last["song_id"] and not seg["is_mashup"] and not last["is_mashup"]:
                # Extend the previous segment
                last["end"] = seg["end"]
                last["score"] = max(last["score"], seg["score"])
            else:
                merged.append(seg.copy())

        return merged
