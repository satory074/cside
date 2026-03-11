"""Legacy feature extractor — chroma CQT, CQT, MFCC, Peak.

These are the features from the original 2019 CSIDE implementation, kept
here as a baseline for comparison experiments (Phase 5).

All methods accept a waveform (numpy array + sample rate) and return a
2D numpy array of shape ``(n_features, n_frames)``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LegacyExtractor:
    """Thin wrapper around librosa for the original CSIDE feature set.

    Parameters
    ----------
    feature:
        One of ``"chroma"``, ``"cqt"``, ``"mfcc"``, ``"peak"``.
    n_fft:
        FFT window size (for ``peak`` mode).
    n_hop:
        Hop length in samples.
    """

    VALID_FEATURES = ("chroma", "cqt", "mfcc", "peak")

    def __init__(
        self,
        feature: str = "chroma",
        n_fft: int = 512,
        n_hop: int = 256,
    ) -> None:
        if feature not in self.VALID_FEATURES:
            raise ValueError(f"feature must be one of {self.VALID_FEATURES}, got {feature!r}")
        self.feature = feature
        self.n_fft = n_fft
        self.n_hop = n_hop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract features from a waveform.

        Returns
        -------
        np.ndarray of shape ``(n_bins, n_frames)``.
        """
        import librosa  # type: ignore

        if self.feature == "chroma":
            return librosa.feature.chroma_cqt(y=audio, sr=sr)

        if self.feature == "cqt":
            return np.abs(librosa.cqt(audio, sr=sr))

        if self.feature == "mfcc":
            return librosa.feature.mfcc(y=audio, sr=sr)

        if self.feature == "peak":
            return self._peak_features(audio, sr)

        raise ValueError(f"Unknown feature: {self.feature}")

    def extract_file(self, path: str) -> np.ndarray:
        """Load an audio file and extract legacy features."""
        import librosa  # type: ignore

        audio, sr = librosa.load(path, sr=None, mono=True)
        return self.extract(audio, sr)

    # ------------------------------------------------------------------
    # OTI (Optimal Transposition Index)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_oti(chroma_a: np.ndarray, chroma_b: np.ndarray) -> int:
        """Compute the OTI (key shift) that maximises the dot product.

        Parameters
        ----------
        chroma_a, chroma_b:
            Global chroma vectors (shape ``(12,)``).

        Returns
        -------
        int: transposition semitones (0–11).
        """
        dots = [np.dot(chroma_a, np.roll(chroma_b, -i)) for i in range(12)]
        return int(np.argmax(dots))

    # ------------------------------------------------------------------
    # Peak / audfprint-style landmark features
    # ------------------------------------------------------------------

    def _peak_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Audfprint-style peak / landmark features (chroma-CQT basis)."""
        import librosa  # type: ignore
        import scipy.signal  # type: ignore

        HPF_POLE = 0.98
        sgram = librosa.feature.chroma_cqt(y=audio, sr=sr)

        sgrammax = sgram.max()
        if sgrammax > 0:
            sgram = np.log(np.maximum(sgram, sgrammax / 1e6))
            sgram = sgram - sgram.mean()

        # High-pass filter
        sgram = np.array(
            [
                scipy.signal.lfilter([1, -1], [1, -HPF_POLE], row)
                for row in sgram
            ]
        )
        return sgram
