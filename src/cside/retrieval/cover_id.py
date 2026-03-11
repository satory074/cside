"""Cover song identification using MuQ embeddings + FAISS retrieval.

Pipeline
--------
1. Extract a mean-pooled MuQ embedding for each candidate track.
2. Store embeddings in an EmbeddingIndex (HNSW).
3. For a query track, retrieve the k most similar tracks.
4. Optionally apply a linear projection head (trained with InfoNCE) to
   map 1024-dim → 128-dim before indexing (improves discriminative power).

Training support (Phase 3)
--------------------------
The ``CoverIdentifier.fine_tune()`` method accepts a DA-TACOS or Covers80
split and trains a projection head with contrastive InfoNCE loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .index import EmbeddingIndex
from ..features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class CoverIdentifier:
    """End-to-end cover song identification.

    Parameters
    ----------
    index_dim:
        Dimensionality of vectors stored in the FAISS index.
        128 when using a projection head, 1024 for raw MuQ.
    use_projection:
        If True, apply a trained projection head (Linear 768→128 + L2-norm).
    device:
        PyTorch device string.
    """

    def __init__(
        self,
        index_dim: int = 128,
        use_projection: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.index_dim = index_dim
        self.use_projection = use_projection
        self.device = device
        self._pipeline = FeaturePipeline(
            extractor="muq",
            aggregate=True,  # mean-pooled → (768,)
            device=device,
        )
        self._index = EmbeddingIndex(dim=index_dim)
        self._projection: Optional[object] = None

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def add_track(self, track_id: str, path: str | Path) -> None:
        """Add a single track to the search index."""
        emb = self._pipeline.extract(path)
        emb = self._project(emb)
        self._index.add(track_id, emb)

    def add_tracks(self, tracks: dict[str, str | Path]) -> None:
        """Add multiple tracks.

        Parameters
        ----------
        tracks:
            Mapping ``track_id → audio_path``.
        """
        for tid, path in tracks.items():
            try:
                self.add_track(tid, path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", tid, exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def identify(
        self,
        query_path: str | Path,
        k: int = 10,
        exclude_self: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """Find the k most similar tracks for a query audio file.

        Parameters
        ----------
        query_path:
            Path to the query audio file.
        k:
            Number of results.
        exclude_self:
            Track ID to exclude (i.e., the query itself if it is in the index).

        Returns
        -------
        list of ``(track_id, similarity_score)``, highest score first.
        """
        emb = self._pipeline.extract(query_path)
        emb = self._project(emb)
        exclude = {exclude_self} if exclude_self else None
        return self._index.search(emb, k=k, exclude=exclude)

    def identify_embedding(
        self,
        embedding: np.ndarray,
        k: int = 10,
        exclude: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Identify using a pre-computed embedding."""
        emb = self._project(embedding)
        return self._index.search(emb, k=k, exclude=exclude)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: str | Path) -> None:
        """Save index and (optionally) projection weights."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self._index.save(dir_path / "cover_index")

        if self._projection is not None:
            import torch  # type: ignore

            torch.save(self._projection.state_dict(), dir_path / "projection.pt")

        logger.info("CoverIdentifier saved to %s", dir_path)

    @classmethod
    def load(cls, dir_path: str | Path, device: Optional[str] = None) -> "CoverIdentifier":
        """Load from a previously saved directory."""
        dir_path = Path(dir_path)
        index = EmbeddingIndex.load(dir_path / "cover_index")

        obj = cls(index_dim=index.dim, device=device)
        obj._index = index

        proj_path = dir_path / "projection.pt"
        if proj_path.exists():
            obj._projection = obj._build_projection_head()
            import torch  # type: ignore

            obj._projection.load_state_dict(
                torch.load(proj_path, map_location=device or "cpu")
            )
            obj.use_projection = True

        return obj

    # ------------------------------------------------------------------
    # Contrastive fine-tuning (Phase 3)
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        cover_pairs: list[tuple[str, str]],
        audio_root: str | Path,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> None:
        """Train the projection head with InfoNCE loss on cover pairs.

        Parameters
        ----------
        cover_pairs:
            List of ``(track_id_a, track_id_b)`` tuples where a and b are
            versions of the same song (positive pairs).
        audio_root:
            Root directory containing audio files named ``<track_id>.mp3``
            (or any format supported by librosa).
        epochs:
            Training epochs.
        batch_size:
            Batch size (number of positive pairs per step).
        lr:
            Learning rate.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError("torch is required for fine-tuning") from exc

        logger.info(
            "Fine-tuning projection head on %d cover pairs for %d epochs",
            len(cover_pairs),
            epochs,
        )

        # Build a raw (no projection) pipeline for getting 768-dim embeddings
        raw_pipeline = FeaturePipeline(extractor="muq", aggregate=True, device=self.device)
        audio_root = Path(audio_root)

        # Pre-extract embeddings for all unique tracks
        unique_ids = {tid for pair in cover_pairs for tid in pair}
        embeddings: dict[str, np.ndarray] = {}
        for tid in unique_ids:
            candidates = list(audio_root.glob(f"{tid}.*"))
            if not candidates:
                logger.warning("No audio file found for track %s, skipping", tid)
                continue
            embeddings[tid] = raw_pipeline.extract(candidates[0])

        if not embeddings:
            raise RuntimeError("No audio files found in audio_root")

        # Filter to pairs where both tracks have embeddings
        valid_pairs = [(a, b) for a, b in cover_pairs if a in embeddings and b in embeddings]
        if not valid_pairs:
            raise RuntimeError("No valid cover pairs after filtering")

        proj = self._build_projection_head()
        optimiser = torch.optim.Adam(proj.parameters(), lr=lr)

        for epoch in range(epochs):
            np.random.shuffle(valid_pairs)  # type: ignore[arg-type]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(valid_pairs), batch_size):
                batch = valid_pairs[start : start + batch_size]
                a_vecs = torch.tensor(
                    np.stack([embeddings[a] for a, _ in batch]), dtype=torch.float32
                )
                b_vecs = torch.tensor(
                    np.stack([embeddings[b] for _, b in batch]), dtype=torch.float32
                )

                a_proj = F.normalize(proj(a_vecs), dim=-1)
                b_proj = F.normalize(proj(b_vecs), dim=-1)

                loss = self._infonce_loss(a_proj, b_proj)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1

            logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, epoch_loss / n_batches)

        self._projection = proj
        self.use_projection = True
        self.index_dim = 128
        self._index = EmbeddingIndex(dim=128)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _project(self, emb: np.ndarray) -> np.ndarray:
        if not self.use_projection or self._projection is None:
            return emb.astype(np.float32)

        import torch
        import torch.nn.functional as F

        vec = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = F.normalize(self._projection(vec), dim=-1)
        return out.squeeze(0).numpy()

    def _build_projection_head(self):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

    @staticmethod
    def _infonce_loss(a: "torch.Tensor", b: "torch.Tensor", temperature: float = 0.07):
        import torch
        import torch.nn.functional as F

        n = a.size(0)
        logits = torch.matmul(a, b.T) / temperature  # (n, n)
        labels = torch.arange(n, device=a.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss
