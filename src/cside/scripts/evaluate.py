"""CLI: Evaluate medley detection results against ground truth annotations.

Usage
-----
    cside-evaluate --result results/medley_result.json \
                   --ground-truth data/annotations/medley_001.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--result", "-r", required=True,
    help="JSON result file produced by cside-medley.",
)
@click.option(
    "--ground-truth", "-g", required=True,
    help="Ground truth annotation JSON file (MedleyGroundTruth schema).",
)
@click.option(
    "--hop", default=1.0, show_default=True,
    help="Frame hop size in seconds (must match detection hop).",
)
@click.option(
    "--boundary-tolerance", default=1.0, show_default=True,
    help="Boundary match tolerance in seconds.",
)
@click.option("--verbose", "-v", is_flag=True)
def main(result, ground_truth, hop, boundary_tolerance, verbose):
    """Evaluate medley detection against ground truth annotations."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    from ..data.annotation import MedleyGroundTruth
    from ..evaluation.metrics import evaluate_medley_detection

    # Load prediction
    pred = json.loads(Path(result).read_text())
    segments = pred["segments"]

    # Load ground truth
    gt = MedleyGroundTruth.from_json(ground_truth)

    # Evaluate
    metrics = evaluate_medley_detection(
        pred_segments=segments,
        ground_truth=gt,
        hop_sec=hop,
        boundary_tolerance=boundary_tolerance,
    )

    click.echo(f"\nEvaluation results for: {Path(result).name}")
    click.echo(f"  Ground truth:    {Path(ground_truth).name}")
    click.echo(f"  Hop size:        {hop}s")
    click.echo(f"  Boundary tol:    ±{boundary_tolerance}s")
    click.echo()
    click.echo(f"  Precision:        {metrics['precision']:.4f}")
    click.echo(f"  Recall:           {metrics['recall']:.4f}")
    click.echo(f"  F1:               {metrics['f1']:.4f}")
    click.echo(f"  Boundary Recall:  {metrics['boundary_recall']:.4f}")
