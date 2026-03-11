"""FAISS-based embedding index for fast approximate nearest-neighbour search.

Architecture
------------
- Inner-product index (cosine similarity after L2-normalisation)
- HNSW graph for sub-linear query time
- Persisted to / loaded from disk as a single ``.faiss`` file alongside
  a ``.json`` metadata file that maps FAISS integer IDs → track IDs.

Usage
-----
    index = EmbeddingIndex(dim=128)
    index.add("song_a", embedding_a)
    index.add("song_b", embedding_b)
    index.save("my_index")

    index2 = EmbeddingIndex.load("my_index")
    results = index2.search(query_embedding, k=5)
    # [(track_id, score), ...]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """HNSW-based FAISS index for audio embeddings.

    Parameters
    ----------
    dim:
        Dimensionality of the embeddings (1024 for MuQ, 128 for projected).
    hnsw_m:
        Number of neighbours in the HNSW graph.  Higher → better recall,
        more memory.  Default 32 is a good balance.
    """

    def __init__(self, dim: int = 1024, hnsw_m: int = 32) -> None:
        self.dim = dim
        self.hnsw_m = hnsw_m
        self._index = None        # lazily created
        self._id_to_track: dict[int, str] = {}
        self._track_to_id: dict[str, int] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add(self, track_id: str, embedding: np.ndarray) -> None:
        """Add a single track embedding to the index.

        Parameters
        ----------
        track_id:
            Unique string identifier for the track.
        embedding:
            1-D float32 array (shape ``(dim,)``).
        """
        if track_id in self._track_to_id:
            logger.debug("Track %s already in index, skipping", track_id)
            return

        vec = self._normalise(embedding).reshape(1, -1)
        self._get_index().add(vec)  # type: ignore[arg-type]

        fid = self._next_id
        self._id_to_track[fid] = track_id
        self._track_to_id[track_id] = fid
        self._next_id += 1

    def add_batch(self, embeddings: dict[str, np.ndarray]) -> None:
        """Add multiple embeddings at once.

        Parameters
        ----------
        embeddings:
            Mapping track_id → embedding array.
        """
        new_ids = [tid for tid in embeddings if tid not in self._track_to_id]
        if not new_ids:
            return
        vecs = np.array([embeddings[tid] for tid in new_ids], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs /= norms
        self._get_index().add(vecs)
        for tid in new_ids:
            fid = self._next_id
            self._id_to_track[fid] = tid
            self._track_to_id[tid] = fid
            self._next_id += 1

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        exclude: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Return the k most similar tracks.

        Parameters
        ----------
        query:
            Query embedding (shape ``(dim,)``).
        k:
            Number of results to return.
        exclude:
            Track IDs to exclude from results (e.g., the query track itself).

        Returns
        -------
        list of ``(track_id, similarity_score)`` tuples, highest score first.
        """
        if self._index is None or self._next_id == 0:
            return []

        exclude = exclude or set()
        vec = self._normalise(query).reshape(1, -1)
        D, I = self._get_index().search(vec, min(k + len(exclude), self._next_id))  # type: ignore

        results = []
        for dist, fid in zip(D[0], I[0]):
            if fid < 0:
                continue
            tid = self._id_to_track[int(fid)]
            if tid in exclude:
                continue
            results.append((tid, float(dist)))
            if len(results) >= k:
                break

        return results

    def __len__(self) -> int:
        return self._next_id

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, prefix: str | Path) -> None:
        """Save index and metadata to ``<prefix>.faiss`` and ``<prefix>.json``."""
        import faiss  # type: ignore

        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._get_index(), str(prefix) + ".faiss")

        meta = {
            "dim": self.dim,
            "hnsw_m": self.hnsw_m,
            "next_id": self._next_id,
            "id_to_track": {str(k): v for k, v in self._id_to_track.items()},
        }
        Path(str(prefix) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        logger.info("Index saved to %s.{faiss,json} (%d tracks)", prefix, self._next_id)

    @classmethod
    def load(cls, prefix: str | Path) -> "EmbeddingIndex":
        """Load index from disk."""
        import faiss  # type: ignore

        prefix = Path(prefix)
        meta = json.loads(Path(str(prefix) + ".json").read_text())

        obj = cls(dim=meta["dim"], hnsw_m=meta["hnsw_m"])
        obj._index = faiss.read_index(str(prefix) + ".faiss")
        obj._next_id = meta["next_id"]
        obj._id_to_track = {int(k): v for k, v in meta["id_to_track"].items()}
        obj._track_to_id = {v: int(k) for k, v in meta["id_to_track"].items()}

        logger.info("Index loaded from %s (%d tracks)", prefix, obj._next_id)
        return obj

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_index(self):
        if self._index is None:
            try:
                import faiss  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "faiss-cpu (or faiss-gpu) is required. Install with: uv add faiss-cpu"
                ) from exc
            self._index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efSearch = 64
        return self._index

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
