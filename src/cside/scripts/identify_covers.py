"""CLI: Identify covers of a query audio file using a pre-built FAISS index.

Usage
-----
    cside-identify --query query.mp3 --index data/indices/my_index --k 5
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option("--query", "-q", required=True, help="Path to query audio file.")
@click.option(
    "--index", "-i", required=True,
    help="Index prefix (without .faiss extension).",
)
@click.option("--k", default=10, show_default=True, help="Number of results to return.")
@click.option(
    "--exclude", default=None,
    help="Track ID to exclude from results (e.g., the query's own ID).",
)
@click.option("--device", default=None, help="PyTorch device.")
@click.option("--verbose", "-v", is_flag=True)
def main(query, index, k, exclude, device, verbose):
    """Find the top-k cover songs for a query audio file."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    from ..retrieval.cover_id import CoverIdentifier

    click.echo(f"Loading index from {index} …")
    identifier = CoverIdentifier.load(index, device=device)

    click.echo(f"Extracting query embedding: {query}")
    results = identifier.identify(query, k=k, exclude_self=exclude)

    click.echo(f"\nTop-{k} results for {Path(query).name}:")
    click.echo(f"{'Rank':<6}{'Track ID':<40}{'Score':>8}")
    click.echo("-" * 56)
    for rank, (track_id, score) in enumerate(results, 1):
        click.echo(f"{rank:<6}{track_id:<40}{score:>8.4f}")
