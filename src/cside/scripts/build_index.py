"""CLI: Build a FAISS search index from a set of audio files.

Usage
-----
    cside-index --input /path/to/audio --output data/indices/my_index
    cside-index --embeddings data/embeddings --output data/indices/my_index
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


@click.command()
@click.option(
    "--input", "-i", "input_path", default=None,
    help="Audio directory. Embeddings are extracted on-the-fly.",
)
@click.option(
    "--embeddings", default=None,
    help="Directory of pre-computed HDF5 embeddings (from cside-extract).",
)
@click.option(
    "--output", "-o", required=True,
    help="Output prefix for the FAISS index (creates <prefix>.faiss and <prefix>.json).",
)
@click.option(
    "--device", default=None,
    help="PyTorch device for on-the-fly extraction.",
)
@click.option(
    "--dim", default=1024, show_default=True,
    help="Embedding dimension (1024 for raw MuQ).",
)
@click.option("--verbose", "-v", is_flag=True)
def main(input_path, embeddings, output, device, dim, verbose):
    """Build a FAISS HNSW index from audio files or pre-computed embeddings."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    from ..retrieval.index import EmbeddingIndex
    from ..features.pipeline import FeaturePipeline

    if input_path is None and embeddings is None:
        raise click.UsageError("Provide either --input or --embeddings.")

    index = EmbeddingIndex(dim=dim)

    if embeddings:
        # Load pre-computed HDF5 embeddings
        import h5py
        import numpy as np

        emb_dir = Path(embeddings)
        h5_files = list(emb_dir.glob("*.h5"))
        click.echo(f"Loading {len(h5_files)} pre-computed embeddings from {emb_dir}")

        # We need a track_id → embedding mapping.
        # Convention: look for a manifest.json or use stem of the h5 file
        manifest_path = emb_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            # manifest: { track_id: h5_filename }
            for track_id, h5_name in manifest.items():
                with h5py.File(emb_dir / h5_name, "r") as f:
                    emb = f["embedding"][:]
                index.add(track_id, emb)
        else:
            # Fall back: use filename stem as track_id
            for h5_file in h5_files:
                with h5py.File(h5_file, "r") as f:
                    emb = f["embedding"][:]
                index.add(h5_file.stem, emb)

    else:
        # Extract on-the-fly
        audio_dir = Path(input_path)
        audio_files = [p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        click.echo(f"Indexing {len(audio_files)} audio files from {audio_dir}")

        pipeline = FeaturePipeline(extractor="muq", aggregate=True, device=device)
        for i, path in enumerate(audio_files, 1):
            try:
                emb = pipeline.extract(path)
                index.add(path.stem, emb)
                if i % 50 == 0:
                    click.echo(f"  {i}/{len(audio_files)}")
            except Exception as exc:
                logger.warning("Failed %s: %s", path, exc)

    out_prefix = Path(output)
    index.save(out_prefix)
    click.echo(f"Index saved: {output}.faiss  ({len(index)} tracks)")
