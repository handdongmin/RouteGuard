"""Shared configuration for the analysis pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime settings for a route-video analysis."""

    confidence_threshold: float = 0.35
    path_width_ratio: float = 0.5
    frame_sample_interval: int = 5
