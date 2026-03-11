"""Boundary detection for medley audio.

Strategy
--------
Given frame-level MuQ embeddings of a long medley audio:

1. Compute cosine novelty curve: at each frame, compare the local context
   window before vs. after the frame.  High novelty = likely boundary.
2. Pick local peaks above a threshold as boundary candidates.
3. (Optional) Mamba encoder can replace the novelty curve for learned
   boundary detection — see the ``MambaBoundaryDetector`` stub below.

The simple novelty-based approach (``NoveltyBoundaryDetector``) requires
no training and works well as a baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


class BoundaryDetector:
    """Cosine novelty–based structural boundary detector.

    Parameters
    ----------
    context_frames:
        Number of frames before/after each position used to build the
        local context window.  Larger values → smoother novelty curve.
    min_interval_sec:
        Minimum time between consecutive boundaries (seconds).
    hop_sec:
        Duration of one frame hop (seconds).  Used to convert frame
        indices to timestamps.
    threshold_percentile:
        Peaks whose novelty value is below this percentile of all
        novelty values are suppressed.
    """

    def __init__(
        self,
        context_frames: int = 8,
        min_interval_sec: float = 4.0,
        hop_sec: float = 1.0,
        threshold_percentile: float = 75.0,
    ) -> None:
        self.context_frames = context_frames
        self.min_interval_sec = min_interval_sec
        self.hop_sec = hop_sec
        self.threshold_percentile = threshold_percentile

    def detect(self, embeddings: np.ndarray) -> list[float]:
        """Detect structural boundaries from frame-level embeddings.

        Parameters
        ----------
        embeddings:
            Array of shape ``(T, D)`` — frame embeddings (e.g., MuQ output).

        Returns
        -------
        list of float: boundary timestamps in seconds, sorted ascending.
        """
        T = len(embeddings)
        if T < 2 * self.context_frames + 1:
            logger.warning("Too few frames (%d) to detect boundaries", T)
            return []

        novelty = self._novelty_curve(embeddings)
        boundaries = self._pick_peaks(novelty)

        # Convert frame indices to seconds
        times = [(b * self.hop_sec) for b in boundaries]
        logger.info("Detected %d boundaries", len(times))
        return sorted(times)

    # ------------------------------------------------------------------
    # Novelty curve
    # ------------------------------------------------------------------

    def _novelty_curve(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine novelty at each frame position.

        For frame i, the novelty is defined as:
            1 - cosine_similarity(mean(emb[i-c:i]), mean(emb[i:i+c]))
        where c = context_frames.
        """
        T, D = embeddings.shape
        novelty = np.zeros(T)

        # Normalise all frame vectors once
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = embeddings / norms

        c = self.context_frames
        # Build sliding window sums via cumulative sum (O(T×D) instead of O(T×c×D))
        cumsum = np.vstack([np.zeros((1, D)), np.cumsum(normed, axis=0)])  # (T+1, D)
        i_arr = np.arange(c, T - c)
        before = (cumsum[i_arr] - cumsum[i_arr - c]) / c          # (N, D)
        after  = (cumsum[i_arr + c] - cumsum[i_arr]) / c          # (N, D)

        dot = np.sum(before * after, axis=1)                      # (N,)
        denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1) + 1e-8
        novelty[c : T - c] = 1.0 - dot / denom

        # Smooth with a Gaussian kernel
        novelty = scipy.ndimage.gaussian_filter1d(novelty, sigma=2)
        return novelty

    def _pick_peaks(self, novelty: np.ndarray) -> list[int]:
        """Find peaks in the novelty curve above the threshold."""
        threshold = np.percentile(novelty, self.threshold_percentile)
        min_samples = max(1, int(self.min_interval_sec / self.hop_sec))

        peaks, _ = scipy.signal.find_peaks(
            novelty,
            height=threshold,
            distance=min_samples,
        )
        return peaks.tolist()


# ---------------------------------------------------------------------------
# Stub: Mamba-based boundary detector (Phase 4 extension)
# ---------------------------------------------------------------------------


class MambaBoundaryDetector:
    """Learned boundary detector using a Mamba (SSM) sequence model.

    This is a design stub for Phase 4.  The full implementation requires:
    - A Mamba encoder (``mamba-ssm`` package or equivalent)
    - A binary boundary classifier head
    - Training data: medley audio with annotated boundaries

    The interface is identical to ``BoundaryDetector.detect()``, so it can
    be swapped in without changing downstream code.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        hop_sec: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.hop_sec = hop_sec
        self.device = device
        self._model = None

    def detect(self, embeddings: np.ndarray) -> list[float]:
        raise NotImplementedError(
            "MambaBoundaryDetector is a Phase 4 stub. "
            "Use BoundaryDetector for the baseline novelty-based approach."
        )
