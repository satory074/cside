"""CLI: Extract and cache MuQ (or legacy) embeddings for a list of audio files.

Usage
-----
    cside-extract --input /path/to/audio --output data/embeddings --extractor muq
    cside-extract --input track.mp3 --output data/embeddings --extractor chroma
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


@click.command()
@click.option(
    "--input", "-i", "input_path", required=True,
    help="Audio file or directory of audio files.",
)
@click.option(
    "--output", "-o", required=True,
    help="Output directory for HDF5 embedding cache.",
)
@click.option(
    "--extractor", "-e", default="muq",
    type=click.Choice(["muq", "chroma", "cqt", "mfcc", "peak"]),
    help="Feature extractor to use (default: muq).",
)
@click.option(
    "--aggregate/--no-aggregate", default=True,
    help="For MuQ: mean-pool frame embeddings into a single vector (default: on).",
)
@click.option(
    "--device", default=None,
    help="PyTorch device (cpu/cuda/mps). Auto-detected by default.",
)
@click.option("--verbose", "-v", is_flag=True)
def main(input_path: str, output: str, extractor: str, aggregate: bool, device, verbose: bool):
    """Extract audio embeddings and cache them to disk."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    from ..features.pipeline import FeaturePipeline

    input_p = Path(input_path)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    if input_p.is_file():
        audio_files = [input_p]
    elif input_p.is_dir():
        audio_files = [p for p in input_p.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    else:
        raise click.BadParameter(f"Input path does not exist: {input_path}")

    if not audio_files:
        raise click.ClickException("No audio files found.")

    click.echo(f"Found {len(audio_files)} audio file(s). Extractor: {extractor}")

    pipeline = FeaturePipeline(
        extractor=extractor,
        cache_dir=str(out_dir),
        aggregate=aggregate,
        device=device,
    )

    failed = 0
    for i, path in enumerate(audio_files, 1):
        click.echo(f"[{i}/{len(audio_files)}] {path.name}", nl=False)
        try:
            emb = pipeline.extract(path)
            click.echo(f"  → {emb.shape}")
        except Exception as exc:
            click.echo(f"  FAILED: {exc}")
            failed += 1

    click.echo(f"\nDone. {len(audio_files) - failed} succeeded, {failed} failed.")
    click.echo(f"Embeddings cached to: {out_dir}")
