"""End-to-end video analysis pipeline."""

from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2

from src.config import AnalysisConfig
from src.detector import ObjectDetector
from src.path_analyzer import PathAnalyzer
from src.risk_rules import build_risk, calculate_route_score, risk_level
from src.visualizer import draw_legend, draw_measurements, draw_path_overlay


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    return f"{minutes:02d}:{remaining:02d}"


def _merge_risk_event(events: list[dict], risk: dict, timestamp: float) -> None:
    """Merge nearby repeated risks so one object does not get counted every frame."""
    for event in reversed(events):
        same_kind = event["label"] == risk["label"]
        close_time = timestamp - event["last_seen_seconds"] <= 2.5
        if same_kind and close_time:
            event["last_seen_seconds"] = timestamp
            if risk["penalty"] > event["penalty"]:
                event["severity"] = risk["severity"]
                event["message"] = risk["message"]
                event["penalty"] = risk["penalty"]
            event["overlap_ratio"] = max(event["overlap_ratio"], risk["overlap_ratio"])
            event["observations"] += 1
            return

    events.append(
        {
            **risk,
            "timestamp": format_timestamp(timestamp),
            "start_seconds": timestamp,
            "last_seen_seconds": timestamp,
            "observations": 1,
        }
    )


def _finalize_events(events: list[dict], config: AnalysisConfig) -> list[dict]:
    """Keep persistent risks and avoid one-frame false alarms."""
    finalized = []
    for event in events:
        duration = event["last_seen_seconds"] - event["start_seconds"]
        persistent = event["observations"] >= config.min_event_observations or duration >= 0.6
        severe = event["severity"] == "danger" and event["overlap_ratio"] >= 0.75

        if event["risk_group"] == "custom":
            keep = event["observations"] >= 1
        elif event["risk_group"] == "large_clutter":
            keep = event["observations"] >= 2
        elif event["risk_group"] == "small":
            keep = event["observations"] >= 3 and event["overlap_ratio"] >= 0.55
        elif event["risk_group"] == "furniture":
            keep = event["observations"] >= 2 and event["overlap_ratio"] >= 0.35
        else:
            keep = persistent or severe

        if keep:
            finalized.append(event)
    return finalized


def _preview_rank(measurements: list[dict]) -> tuple[int, float]:
    """Rank annotated frames so the preview shows the clearest risk moment."""
    severity_rank = {"safe": 0, "caution": 1, "danger": 2}
    highest_severity = 0
    highest_overlap = 0.0
    for measurement in measurements:
        highest_severity = max(highest_severity, severity_rank.get(measurement.get("severity", "safe"), 0))
        highest_overlap = max(highest_overlap, float(measurement.get("overlap_ratio", 0.0)))
    return highest_severity, highest_overlap


def analyze_video(
    input_path: str | Path,
    output_path: str | Path,
    config: AnalysisConfig | None = None,
    detector: ObjectDetector | None = None,
) -> dict:
    """Analyze a video and save an annotated result video."""
    started_at = perf_counter()
    config = config or AnalysisConfig()
    detector = detector or ObjectDetector(config)
    path_analyzer = PathAnalyzer(config)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(config.max_video_seconds * fps) if config.max_video_seconds else 0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    events: list[dict] = []
    frame_index = 0
    last_measurements: list[dict] = []
    sampled_frames = 0
    risk_sampled_frames = 0
    detection_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    max_path_overlap = 0.0
    preview_path = output_path.with_name(f"{output_path.stem}_preview.jpg")
    preview_frame = None
    preview_rank = (-1, -1.0)
    preview_timestamp = 0.0

    while True:
        if max_frames and frame_index >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % config.frame_sample_interval == 0:
            sampled_frames += 1
            detections = detector.detect(frame)
            last_measurements = path_analyzer.analyze(frame, detections)
            timestamp = frame_index / fps
            frame_has_risk = False
            for measurement in last_measurements:
                detection_counts[measurement["label"]] += 1
                severity_counts[measurement["severity"]] += 1
                group_counts[measurement.get("risk_group", "other")] += 1
                max_path_overlap = max(max_path_overlap, float(measurement.get("overlap_ratio", 0.0)))
                risk = build_risk(measurement)
                if risk:
                    frame_has_risk = True
                    _merge_risk_event(events, risk, timestamp)
            if frame_has_risk:
                risk_sampled_frames += 1

        annotated = frame.copy()
        draw_path_overlay(annotated, path_analyzer.get_path_polygon(annotated))
        draw_measurements(annotated, last_measurements, draw_safe=config.draw_safe_detections)
        draw_legend(annotated)
        writer.write(annotated)

        current_rank = _preview_rank(last_measurements)
        if preview_frame is None or current_rank > preview_rank:
            preview_frame = annotated.copy()
            preview_rank = current_rank
            preview_timestamp = frame_index / fps if fps else 0.0
        frame_index += 1

    cap.release()
    writer.release()

    if preview_frame is not None:
        cv2.imwrite(str(preview_path), preview_frame)

    events = _finalize_events(events, config)
    score = calculate_route_score(events)
    elapsed_seconds = perf_counter() - started_at
    risk_frame_ratio = risk_sampled_frames / sampled_frames if sampled_frames else 0.0
    return {
        "output_path": str(output_path),
        "preview_path": str(preview_path) if preview_frame is not None else "",
        "preview_timestamp": preview_timestamp,
        "score": score,
        "risk_level": risk_level(score),
        "events": events,
        "frames": frame_index,
        "fps": fps,
        "duration_seconds": frame_index / fps if fps else 0,
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "risk_sampled_frames": risk_sampled_frames,
        "risk_frame_ratio": risk_frame_ratio,
        "detection_counts": dict(detection_counts),
        "severity_counts": dict(severity_counts),
        "group_counts": dict(group_counts),
        "max_path_overlap": max_path_overlap,
        "elapsed_seconds": elapsed_seconds,
        "processing_fps": frame_index / elapsed_seconds if elapsed_seconds else 0.0,
        "config": {
            "confidence_threshold": config.confidence_threshold,
            "inference_image_size": config.inference_image_size,
            "path_width_ratio": config.path_width_ratio,
            "frame_sample_interval": config.frame_sample_interval,
            "min_event_observations": config.min_event_observations,
            "max_video_seconds": config.max_video_seconds,
        },
    }
