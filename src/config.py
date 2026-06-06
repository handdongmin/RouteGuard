"""Shared configuration for the analysis pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime settings for a route-video analysis."""

    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    inference_image_size: int = 640
    path_width_ratio: float = 0.42
    frame_sample_interval: int = 5
    min_event_observations: int = 2
    max_video_seconds: int = 20
    draw_safe_detections: bool = False
