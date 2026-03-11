"""Unified feature extraction pipeline.

Supports both MuQ (default) and legacy (chroma/CQT/MFCC) extractors.
Embeddings are cached to disk as HDF5 files to avoid redundant computation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

ExtractorType = Literal["muq", "chroma", "cqt", "mfcc", "peak"]


class FeaturePipeline:
    """Extract and optionally cache audio features.

    Parameters
    ----------
    extractor:
        ``"muq"`` for MuQ embeddings (default), or a legacy feature name.
    cache_dir:
        If set, cache embeddings to ``<cache_dir>/<hash>.h5``.
    aggregate:
        For MuQ: if True, mean-pool frame embeddings into a single vector.
        Ignored for legacy features.
    device:
        PyTorch device for MuQ (auto-detected when None).
    """

    def __init__(
        self,
        extractor: ExtractorType = "muq",
        cache_dir: Optional[str | Path] = None,
        aggregate: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.extractor_type = extractor
        self.aggregate = aggregate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._muq: Optional[object] = None
        self._legacy: Optional[object] = None
        self._device = device

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, path: str | Path) -> np.ndarray:
        """Return embeddings / features for an audio file.

        Reads from cache if available; otherwise computes and caches.

        Returns
        -------
        np.ndarray:
            Shape ``(768,)`` or ``(T, 768)`` for MuQ,
            or ``(n_bins, T)`` for legacy features.
        """
        path = Path(path)
        cache_key = self._cache_key(path)

        if self.cache_dir:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        features = self._compute(path)

        if self.cache_dir:
            self._save_cache(cache_key, features)

        return features

    def extract_batch(self, paths: list[str | Path]) -> dict[str, np.ndarray]:
        """Extract features for a list of files.

        Returns
        -------
        dict mapping path string → embedding array.
        """
        results = {}
        for p in paths:
            try:
                results[str(p)] = self.extract(p)
            except Exception as exc:
                logger.warning("Failed to extract %s: %s", p, exc)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute(self, path: Path) -> np.ndarray:
        if self.extractor_type == "muq":
            return self._muq_extractor().encode_file(path, aggregate=self.aggregate)
        else:
            return self._legacy_extractor().extract_file(str(path))

    def _muq_extractor(self):
        if self._muq is None:
            from .muq import MuQExtractor  # type: ignore

            self._muq = MuQExtractor(device=self._device)
        return self._muq

    def _legacy_extractor(self):
        if self._legacy is None:
            from .legacy import LegacyExtractor  # type: ignore

            self._legacy = LegacyExtractor(feature=self.extractor_type)
        return self._legacy

    # ------------------------------------------------------------------
    # Cache I/O (HDF5)
    # ------------------------------------------------------------------

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        digest = hashlib.sha1(
            f"{path}:{stat.st_size}:{stat.st_mtime}:{self.extractor_type}:{self.aggregate}".encode()
        ).hexdigest()
        return digest

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.h5"

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        cp = self._cache_path(key)
        if not cp.exists():
            return None
        try:
            import h5py  # type: ignore

            with h5py.File(cp, "r") as f:
                return f["embedding"][:]
        except Exception as exc:
            logger.debug("Cache read failed for %s: %s", cp, exc)
            return None

    def _save_cache(self, key: str, embedding: np.ndarray) -> None:
        try:
            import h5py  # type: ignore

            with h5py.File(self._cache_path(key), "w") as f:
                f.create_dataset("embedding", data=embedding, compression="gzip")
        except Exception as exc:
            logger.debug("Cache write failed: %s", exc)
