"""Feature extraction: MuQ embeddings and legacy chroma/CQT features."""

from .muq import MuQExtractor
from .legacy import LegacyExtractor
from .pipeline import FeaturePipeline

__all__ = ["MuQExtractor", "LegacyExtractor", "FeaturePipeline"]
