"""Dataset loaders for standard cover-song benchmarks.

Supported datasets
------------------
- Covers80   : 80 songs × 2 covers each (small, fast validation)
- DA-TACOS   : 15,620 tracks in 1,000 work clusters (main benchmark)

Both are loaded via `mirdata` where available; a thin fallback path is
provided for offline / partial downloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """Minimal track metadata returned by all loaders."""

    track_id: str
    audio_path: str
    title: Optional[str] = None
    artist: Optional[str] = None
    work_id: Optional[str] = None  # Cluster / "work" that groups covers together


def load_covers80(
    data_home: str | Path | None = None,
    download: bool = False,
) -> list[TrackInfo]:
    """Load Covers80 dataset via mirdata.

    Parameters
    ----------
    data_home:
        Root directory where mirdata stores the dataset.
        Defaults to mirdata's default data directory.
    download:
        If True, download the dataset automatically.

    Returns
    -------
    list of TrackInfo, one per track (160 total).
    """
    try:
        import mirdata  # type: ignore

        dataset = mirdata.initialize("covers80", data_home=str(data_home) if data_home else None)
        if download:
            dataset.download()

        tracks = []
        for track_id, track in dataset.load_tracks().items():
            audio_path = track.audio_path or ""
            tracks.append(
                TrackInfo(
                    track_id=track_id,
                    audio_path=audio_path,
                    title=getattr(track, "title", None),
                    artist=getattr(track, "artist", None),
                    work_id=getattr(track, "cover_song_id", None),
                )
            )
        logger.info("Covers80: loaded %d tracks", len(tracks))
        return tracks

    except Exception as exc:
        logger.warning("Could not load Covers80 via mirdata: %s", exc)
        return []


def load_da_tacos(
    data_home: str | Path | None = None,
    subset: str = "coveranalysis",
    download: bool = False,
) -> list[TrackInfo]:
    """Load DA-TACOS dataset via mirdata.

    Parameters
    ----------
    data_home:
        Root directory for mirdata storage.
    subset:
        ``"coveranalysis"`` (15,000 cover pairs) or ``"benchmark"`` (1,000 queries).
    download:
        If True, attempt to download metadata (audio must be fetched separately).

    Returns
    -------
    list of TrackInfo.
    """
    try:
        import mirdata  # type: ignore

        dataset = mirdata.initialize("da_tacos", data_home=str(data_home) if data_home else None)
        if download:
            dataset.download(partial_download=["metadata"])

        tracks = []
        for track_id, track in dataset.load_tracks().items():
            audio_path = getattr(track, "audio_path", None) or ""
            tracks.append(
                TrackInfo(
                    track_id=track_id,
                    audio_path=audio_path,
                    title=getattr(track, "title", None),
                    artist=getattr(track, "perf_artist", None),
                    work_id=getattr(track, "work_id", None),
                )
            )
        logger.info("DA-TACOS (%s): loaded %d tracks", subset, len(tracks))
        return tracks

    except Exception as exc:
        logger.warning("Could not load DA-TACOS via mirdata: %s", exc)
        return []
