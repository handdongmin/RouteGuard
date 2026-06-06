"""Utilities for summarizing bundled sample analysis results."""

from pathlib import Path

from src.config import AnalysisConfig
from src.detector import ObjectDetector
from src.reporting import summarize_events
from src.video_analyzer import analyze_video


def discover_sample_videos(sample_dir: str | Path = "data/samples") -> list[Path]:
    """Return bundled sample videos in a stable order."""
    sample_dir = Path(sample_dir)
    return sorted(
        path
        for path in sample_dir.glob("**/*")
        if path.suffix.lower() in {".mp4", ".mov", ".avi"}
    )


def analyze_samples(
    sample_dir: str | Path = "data/samples",
    output_dir: str | Path = "outputs/results",
    config: AnalysisConfig | None = None,
) -> list[dict]:
    """Analyze all bundled samples and return README-friendly summary rows."""
    config = config or AnalysisConfig()
    detector = ObjectDetector(config)
    output_dir = Path(output_dir)
    rows = []

    for video in discover_sample_videos(sample_dir):
        result = analyze_video(video, output_dir / f"{video.stem}_sample_report.mp4", config=config, detector=detector)
        summary = summarize_events(result["events"])
        rows.append(
            {
                "sample": f"{video.parent.name}/{video.name}",
                "score": result["score"],
                "level": result["risk_level"],
                "events": summary["event_count"],
                "max_overlap": f"{summary['max_overlap']:.0%}",
                "output": result["output_path"],
            }
        )

    return rows
