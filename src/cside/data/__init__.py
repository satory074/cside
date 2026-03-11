"""Data loading and annotation utilities."""

from .annotation import Annotation, SegmentLabel, MedleyGroundTruth
from .datasets import load_covers80, load_da_tacos

__all__ = [
    "Annotation",
    "SegmentLabel",
    "MedleyGroundTruth",
    "load_covers80",
    "load_da_tacos",
]
