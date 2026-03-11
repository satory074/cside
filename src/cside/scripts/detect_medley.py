"""CLI: Detect song fragments in a medley audio file.

Usage
-----
    cside-medley --medley medley.mp3 --index data/indices/candidates \
                 --output results/medley_result.json

    # Sliding-window mode (no boundary detection):
    cside-medley --medley medley.mp3 --index data/indices/candidates \
                 --window 16 --stride 4 --output results/result.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option("--medley", "-m", required=True, help="Path to medley audio file.")
@click.option(
    "--index", "-i", required=True,
    help="Index prefix pointing to the candidate track FAISS index.",
)
@click.option(
    "--output", "-o", required=True,
    help="Path to write the JSON result.",
)
@click.option(
    "--window", default=0.0,
    help="Sliding-window size (seconds). 0 = use boundary detection (default).",
)
@click.option("--stride", default=4.0, show_default=True, help="Sliding-window stride (seconds).")
@click.option("--hop", default=1.0, show_default=True, help="Frame hop size (seconds).")
@click.option(
    "--mashup-threshold", default=0.85, show_default=True,
    help="Ratio threshold for mashup detection (0–1).",
)
@click.option("--top-k", default=5, show_default=True, help="Candidates per segment.")
@click.option("--device", default=None, help="PyTorch device.")
@click.option("--verbose", "-v", is_flag=True)
def main(medley, index, output, window, stride, hop, mashup_threshold, top_k, device, verbose):
    """Detect which candidate songs appear in each segment of a medley."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    import librosa
    import numpy as np

    from ..features.muq import MuQExtractor
    from ..retrieval.cover_id import CoverIdentifier
    from ..segmentation.boundary import BoundaryDetector
    from ..segmentation.medley import MedleyDetector

    # 1. Extract MuQ frame embeddings for the medley
    click.echo(f"Loading medley audio: {medley}")
    audio, sr = librosa.load(medley, sr=None, mono=True)
    duration = len(audio) / sr
    click.echo(f"  Duration: {duration:.1f}s")

    click.echo("Extracting MuQ frame embeddings …")
    extractor = MuQExtractor(device=device)
    medley_embeddings = extractor.encode(audio, sr, aggregate=False)  # (T, 768)
    click.echo(f"  Embeddings shape: {medley_embeddings.shape}")

    # 2. Load candidate index
    click.echo(f"Loading candidate index: {index}")
    identifier = CoverIdentifier.load(index, device=device)

    # 3. Build detector
    boundary_detector = BoundaryDetector(hop_sec=hop)
    detector = MedleyDetector(
        cover_identifier=identifier,
        boundary_detector=boundary_detector,
        hop_sec=hop,
        mashup_threshold=mashup_threshold,
        top_k=top_k,
    )

    # 4. Run detection
    if window > 0:
        click.echo(f"Running sliding-window detection (window={window}s, stride={stride}s) …")
        segments = detector.detect_sliding_window(
            medley_embeddings, window_sec=window, stride_sec=stride
        )
    else:
        click.echo("Running boundary-based detection …")
        segments = detector.detect(medley_embeddings)

    # 5. Write output
    result = {
        "medley": str(Path(medley).resolve()),
        "index": str(Path(index).resolve()),
        "duration_sec": float(duration),
        "segments": segments,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # 6. Print summary
    click.echo(f"\nDetected {len(segments)} segment(s):\n")
    click.echo(f"{'Start':>8}  {'End':>8}  {'Song ID':<35}  {'Score':>6}  Mashup")
    click.echo("-" * 70)
    for seg in segments:
        song_id = seg["song_id"] or "(none)"
        mashup_marker = "YES" if seg["is_mashup"] else ""
        click.echo(
            f"{seg['start']:>8.1f}  {seg['end']:>8.1f}  "
            f"{song_id:<35}  {seg['score']:>6.3f}  {mashup_marker}"
        )

    mashup_count = sum(1 for s in segments if s["is_mashup"])
    click.echo(f"\n{mashup_count} mashup segment(s) detected.")
    click.echo(f"Results written to: {output}")
