"""MuQ-based audio embedding extractor.

Uses the MuQ-large-msd-iter model (OpenMuQ/MuQ-large-msd-iter) — a music-domain
self-supervised model that outperforms MERT on audio fingerprinting tasks.

The extractor returns frame-level embeddings (T × 1024) suitable for:
- Cover song identification (aggregate → single 1024-dim vector)
- Medley segment detection (per-frame similarity queries)

Lazy loading
------------
The model is not downloaded until the first call to ``encode()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# MuQ model identifier on the Hugging Face Hub
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
SAMPLE_RATE = 24_000   # MuQ expected sample rate
EMBED_DIM = 1024       # output dimension of last_hidden_state


class MuQExtractor:
    """Extract MuQ embeddings from audio.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier. Defaults to ``OpenMuQ/MuQ-large-msd-iter``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Auto-detected when not specified.
    cache_dir:
        Directory for caching HF model weights.
    """

    def __init__(
        self,
        model_id: str = MUQ_MODEL_ID,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._device = device or self._auto_device()
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        audio: np.ndarray,
        sr: int,
        aggregate: bool = False,
    ) -> np.ndarray:
        """Encode an audio array into MuQ embeddings.

        Parameters
        ----------
        audio:
            1-D float32 waveform array.
        sr:
            Sample rate of ``audio``.  Resampled to 24 kHz automatically.
        aggregate:
            If True, return a single mean-pooled vector (shape ``(1024,)``).
            If False, return frame-level embeddings (shape ``(T, 1024)``).

        Returns
        -------
        np.ndarray of float32.
        """
        self._load_model()

        audio_24k = self._resample(audio, sr)
        wav_tensor = torch.tensor(audio_24k).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(wav_tensor, output_hidden_states=True)
            embeddings = outputs.last_hidden_state  # (1, T, 1024)

        emb = embeddings[0].cpu().numpy()  # (T, 1024)
        if aggregate:
            return emb.mean(axis=0)  # (1024,)
        return emb

    def encode_file(self, path: str | Path, aggregate: bool = False) -> np.ndarray:
        """Load an audio file and return MuQ embeddings.

        Parameters
        ----------
        path:
            Path to an audio file (any format supported by librosa).
        aggregate:
            See :meth:`encode`.
        """
        import librosa  # type: ignore

        audio, sr = librosa.load(str(path), sr=None, mono=True)
        return self.encode(audio, sr, aggregate=aggregate)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from muq import MuQ  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "muq is required for MuQ embeddings. "
                "Install with: uv add muq"
            ) from exc

        logger.info("Loading MuQ model from %s …", self.model_id)
        kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        self._model = MuQ.from_pretrained(self.model_id, **kwargs)
        self._model.eval()
        self._model.to(self._device)
        logger.info("MuQ model loaded on device: %s", self._device)

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == SAMPLE_RATE:
            return audio
        try:
            import torchaudio  # type: ignore

            waveform = torch.from_numpy(audio).unsqueeze(0)
            resampled = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            return resampled.squeeze(0).numpy()
        except ImportError:
            import librosa  # type: ignore

            return librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
