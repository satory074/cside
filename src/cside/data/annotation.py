"""Ground truth annotation schema for medley / cover-song detection.

JSON schema example
-------------------
{
  "medley_id": "jpop_medley_001",
  "audio_path": "data/audio/jpop_medley_001.mp3",
  "duration_sec": 240.0,
  "segments": [
    {
      "start": 0.0,
      "end": 32.5,
      "song_ids": ["song_A"],
      "is_mashup": false,
      "notes": ""
    },
    {
      "start": 32.5,
      "end": 48.0,
      "song_ids": ["song_B", "song_C"],
      "is_mashup": true,
      "notes": "B plays in verse, C in chorus"
    }
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class SegmentLabel(BaseModel):
    """A single annotated segment within a medley."""

    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    song_ids: list[str] = Field(..., description="IDs of songs present in this segment")
    is_mashup: bool = Field(default=False, description="True when ≥2 songs overlap simultaneously")
    notes: str = Field(default="", description="Optional human annotation note")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def primary_song_id(self) -> str:
        """Return the first song id (main song for non-mashup segments)."""
        return self.song_ids[0]


class MedleyGroundTruth(BaseModel):
    """Ground truth annotations for a single medley audio file."""

    medley_id: str
    audio_path: str
    duration_sec: float
    segments: list[SegmentLabel]

    @classmethod
    def from_json(cls, path: str | Path) -> "MedleyGroundTruth":
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    def to_frame_labels(self, hop_sec: float = 1.0) -> tuple[np.ndarray, list[list[str]]]:
        """Convert segment annotations to per-frame labels.

        Returns
        -------
        times : np.ndarray
            Centre time of each frame (seconds).
        labels : list of list[str]
            song_ids present in each frame (empty list = unlabelled gap).
        """
        n_frames = int(np.ceil(self.duration_sec / hop_sec))
        times = np.arange(n_frames) * hop_sec + hop_sec / 2
        labels: list[list[str]] = [[] for _ in range(n_frames)]

        for seg in self.segments:
            start_idx = int(seg.start / hop_sec)
            end_idx = min(int(np.ceil(seg.end / hop_sec)), n_frames)
            for idx in range(start_idx, end_idx):
                labels[idx] = seg.song_ids

        return times, labels


# Alias for backward compatibility
Annotation = MedleyGroundTruth
