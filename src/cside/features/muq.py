"""MuQ-based audio embedding extractor.

Uses the MuQ-MuLan model (m-a-p/MuQ-MuLan, January 2025) — a music-domain
self-supervised model that outperforms MERT on audio fingerprinting tasks.

The extractor returns frame-level embeddings (T × 768) suitable for:
- Cover song identification (aggregate → single 768-dim vector)
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
MUQ_MODEL_ID = "m-a-p/MuQ-MuLan"
SAMPLE_RATE = 24_000   # MuQ expected sample rate
EMBED_DIM = 768        # output dimension of last_hidden_state


class MuQExtractor:
    """Extract MuQ embeddings from audio.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier. Defaults to ``m-a-p/MuQ-MuLan``.
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
        self._extractor = None

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
            If True, return a single mean-pooled vector (shape ``(768,)``).
            If False, return frame-level embeddings (shape ``(T, 768)``).

        Returns
        -------
        np.ndarray of float32.
        """
        self._load_model()

        audio_24k = self._resample(audio, sr)
        inputs = self._extractor(
            audio_24k,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = outputs.last_hidden_state  # (1, T, 768)

        emb = embeddings[0].cpu().numpy()  # (T, 768)
        if aggregate:
            return emb.mean(axis=0)  # (768,)
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
            from transformers import AutoFeatureExtractor, AutoModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers is required for MuQ embeddings. "
                "Install with: uv add transformers torch"
            ) from exc

        logger.info("Loading MuQ model from %s …", self.model_id)
        kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        self._extractor = AutoFeatureExtractor.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModel.from_pretrained(self.model_id, **kwargs)
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
