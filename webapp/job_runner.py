"""Background job runner for long-running medley detection tasks.

Uses a single-worker ThreadPoolExecutor to prevent OOM from concurrent MuQ loads.
KMP_DUPLICATE_LIB_OK must be set before torch/faiss are imported — done here at
module level (mirrors src/cside/__init__.py).
"""

from __future__ import annotations

import os

# Must be set before torch or faiss are imported (macOS libomp conflict)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Single-worker pool: medley detection is memory-heavy; serial execution avoids OOM
_executor = ThreadPoolExecutor(max_workers=1)


@dataclass
class RunnerState:
    status: str = "idle"          # idle | running | done | error
    progress: float = 0.0         # 0.0 – 1.0
    message: str = ""
    result: Optional[list[dict]] = None
    error: Optional[str] = None
    duration_sec: float = 0.0


def run_detection(
    medley_path: str,
    identifier,                    # CoverIdentifier
    detection_mode: str = "boundary",
    window_sec: float = 16.0,
    stride_sec: float = 4.0,
    hop_sec: float = 1.0,
    mashup_threshold: float = 0.85,
    top_k: int = 5,
    device: Optional[str] = None,
    progress_cb: Optional[Any] = None,  # callable(fraction, message) or gr.Progress
) -> list[dict]:
    """Run the full medley detection pipeline synchronously.

    Designed to be submitted to _executor via submit_job().

    Parameters
    ----------
    medley_path:
        Path to the medley audio file.
    identifier:
        A pre-loaded CoverIdentifier instance.
    detection_mode:
        "boundary" (default) or "sliding_window".
    window_sec, stride_sec:
        Used only when detection_mode == "sliding_window".
    hop_sec:
        Frame hop size in seconds.
    mashup_threshold:
        Ratio threshold for mashup detection.
    top_k:
        Number of FAISS candidates per segment.
    device:
        PyTorch device.
    progress_cb:
        Optional callable(fraction, message) for progress updates.

    Returns
    -------
    List of segment dicts (see MedleyDetector schema).
    """
    import librosa

    from cside.features.muq import MuQExtractor
    from cside.segmentation.boundary import BoundaryDetector
    from cside.segmentation.medley import MedleyDetector

    def _progress(frac: float, msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(frac, desc=msg)
            except Exception:
                pass

    # --- Stage 1: Load audio ---
    _progress(0.0, "Loading audio …")
    audio, sr = librosa.load(medley_path, sr=None, mono=True)
    duration = len(audio) / sr
    logger.info("Loaded medley: %.1f s @ %d Hz", duration, sr)

    # --- Stage 2: Extract MuQ embeddings ---
    _progress(0.1, "Extracting MuQ embeddings …")
    extractor = MuQExtractor(device=device)
    medley_embeddings = extractor.encode(audio, sr, aggregate=False)  # (T, 768)
    logger.info("Embeddings shape: %s", medley_embeddings.shape)
    _progress(0.5, f"Embeddings extracted ({medley_embeddings.shape[0]} frames)")

    # --- Stage 3: Detect segments ---
    boundary_detector = BoundaryDetector(hop_sec=hop_sec)
    detector = MedleyDetector(
        cover_identifier=identifier,
        boundary_detector=boundary_detector,
        hop_sec=hop_sec,
        mashup_threshold=mashup_threshold,
        top_k=top_k,
    )

    if detection_mode == "sliding_window":
        _progress(0.5, f"Sliding-window detection (window={window_sec}s, stride={stride_sec}s) …")
        segments = detector.detect_sliding_window(
            medley_embeddings, window_sec=window_sec, stride_sec=stride_sec
        )
    else:
        _progress(0.5, "Boundary-based detection …")
        segments = detector.detect(medley_embeddings)

    _progress(1.0, f"Done — {len(segments)} segment(s) detected")
    logger.info("Detection complete: %d segment(s)", len(segments))
    return segments


def submit_job(
    medley_path: str,
    identifier,
    **kwargs,
) -> Future:
    """Submit a detection job to the background executor.

    Returns a Future whose result() is the list of segment dicts.
    """
    return _executor.submit(run_detection, medley_path, identifier, **kwargs)
