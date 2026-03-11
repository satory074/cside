"""Manage FAISS indices for the web app.

Two modes:
  - Demo index: pre-built index shipped with the repo in webapp/demo_data/demo_index/
  - User-uploaded: built on-the-fly from audio files uploaded through the Gradio UI
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Path to the pre-built demo index (committed to the repo)
_DEMO_INDEX_DIR = Path(__file__).parent / "demo_data" / "demo_index"


def get_demo_identifier(device: Optional[str] = None):
    """Load the pre-built demo CoverIdentifier.

    Returns None if the demo index has not been built yet.
    """
    from cside.retrieval.cover_id import CoverIdentifier

    if not (_DEMO_INDEX_DIR / "cover_index.faiss").exists():
        logger.warning("Demo index not found at %s — returning empty identifier", _DEMO_INDEX_DIR)
        # Return an empty identifier so the UI still works without crashing
        identifier = CoverIdentifier(index_dim=1024, use_projection=False, device=device)
        return identifier

    return CoverIdentifier.load(_DEMO_INDEX_DIR, device=device)


def build_from_uploads(
    file_paths: list[str],
    progress_cb: Optional[Callable[[float, str], None]] = None,
    device: Optional[str] = None,
):
    """Build a CoverIdentifier from user-uploaded audio files.

    Parameters
    ----------
    file_paths:
        List of temporary file paths provided by Gradio.
    progress_cb:
        Optional callback(fraction, message) for progress reporting.
    device:
        PyTorch device string.

    Returns
    -------
    CoverIdentifier with all uploaded tracks indexed.
    """
    from cside.retrieval.cover_id import CoverIdentifier

    identifier = CoverIdentifier(index_dim=1024, use_projection=False, device=device)

    n = len(file_paths)
    for i, fp in enumerate(file_paths):
        p = Path(fp)
        track_id = p.stem  # use filename (without extension) as ID
        if progress_cb:
            progress_cb(i / n, f"Indexing {p.name} ({i + 1}/{n})")
        try:
            identifier.add_track(track_id, p)
            logger.info("Indexed %s as '%s'", p.name, track_id)
        except Exception as exc:
            logger.warning("Failed to index %s: %s", p.name, exc)

    if progress_cb:
        progress_cb(1.0, f"Done — {n} track(s) indexed")

    return identifier
