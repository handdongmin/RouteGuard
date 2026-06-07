"""End-to-end video analysis pipeline."""

from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
from PIL import Image

from src.config import AnalysisConfig
from src.detector import ObjectDetector
from src.path_analyzer import PathAnalyzer
from src.risk_rules import build_risk, calculate_route_score, risk_level
from src.visualizer import draw_legend, draw_measurements, draw_path_overlay


PREVIEW_PREFERRED_LABELS = {
    "backpack",
    "chair",
    "couch",
    "handbag",
    "potted plant",
    "suitcase",
    "umbrella",
}

ANALYZER_VERSION = "event-preview-v2"


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


def _preview_rank(measurements: list[dict]) -> tuple[int, int, int, float, float]:
    """Rank annotated frames so the preview shows the clearest risk moment."""
    severity_rank = {"safe": 0, "caution": 1, "danger": 2}
    highest_severity = 0
    highest_overlap = 0.0
    highest_area = 0.0
    blocker_count = 0
    preferred_count = 0
    for measurement in measurements:
        severity = measurement.get("severity", "safe")
        if severity == "safe":
            continue
        blocker_count += 1
        if measurement.get("label") in PREVIEW_PREFERRED_LABELS:
            preferred_count += 1
        highest_severity = max(highest_severity, severity_rank.get(severity, 0))
        highest_overlap = max(highest_overlap, float(measurement.get("overlap_ratio", 0.0)))
        highest_area = max(highest_area, float(measurement.get("bbox_area_ratio", 0.0)))
    return blocker_count, preferred_count, highest_severity, highest_overlap, highest_area


def _open_video_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open the annotated MP4 writer."""
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer
    writer.release()
    raise RuntimeError(f"Could not open output video: {output_path}")


def _make_gif_frame(frame, max_width: int = 520) -> Image.Image:
    """Convert an annotated OpenCV frame to a compact GIF frame."""
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _target_event(events: list[dict]) -> dict | None:
    """Return the event that should drive the representative preview image."""
    severity_rank = {"safe": 0, "caution": 1, "danger": 2}
    if not events:
        return None
    return max(
        events,
        key=lambda event: (
            int(event.get("penalty", 0)),
            severity_rank.get(event.get("severity", "safe"), 0),
            float(event.get("overlap_ratio", 0.0)),
            int(event.get("observations", 0)),
        ),
    )


def _candidate_rank_for_event(candidate: dict, event: dict | None) -> tuple[int, int, float, float, int]:
    """Rank saved annotated frames against the highest-penalty event."""
    if event is None:
        blocker_count, preferred_count, severity, overlap, area = _preview_rank(candidate["measurements"])
        return (preferred_count, severity, overlap, area, blocker_count)

    severity_rank = {"safe": 0, "caution": 1, "danger": 2}
    target_label = event.get("label")
    matching = [measurement for measurement in candidate["measurements"] if measurement.get("label") == target_label]
    if not matching:
        return (0, 0, 0.0, 0.0, 0)

    best = max(
        matching,
        key=lambda measurement: (
            severity_rank.get(measurement.get("severity", "safe"), 0),
            float(measurement.get("overlap_ratio", 0.0)),
            float(measurement.get("bbox_area_ratio", 0.0)),
        ),
    )
    return (
        1,
        severity_rank.get(best.get("severity", "safe"), 0),
        float(best.get("overlap_ratio", 0.0)),
        float(best.get("bbox_area_ratio", 0.0)),
        len([measurement for measurement in candidate["measurements"] if measurement.get("severity") != "safe"]),
    )


def _select_preview_candidate(candidates: list[dict], events: list[dict]) -> dict | None:
    """Choose the preview frame from the highest-penalty timeline event."""
    if not candidates:
        return None
    target = _target_event(events)
    return max(candidates, key=lambda candidate: _candidate_rank_for_event(candidate, target))


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

    try:
        writer = _open_video_writer(output_path, fps, (width, height))
    except RuntimeError:
        cap.release()
        raise

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
    gif_path = output_path.with_name(f"{output_path.stem}_preview.gif")
    preview_candidates: list[dict] = []
    preview_timestamp = 0.0
    gif_frames: list[Image.Image] = []
    gif_interval = max(1, int(fps // 3))

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

        if any(measurement.get("severity") != "safe" for measurement in last_measurements):
            preview_candidates.append(
                {
                    "frame": annotated.copy(),
                    "measurements": [dict(measurement) for measurement in last_measurements],
                    "timestamp": frame_index / fps if fps else 0.0,
                }
            )
        if frame_index % gif_interval == 0 and len(gif_frames) < 45:
            gif_frames.append(_make_gif_frame(annotated))
        frame_index += 1

    cap.release()
    writer.release()

    events = _finalize_events(events, config)
    preview_candidate = _select_preview_candidate(preview_candidates, events) if events else None
    preview_event = _target_event(events)
    if preview_candidate:
        preview_timestamp = float(preview_candidate["timestamp"])
        cv2.imwrite(str(preview_path), preview_candidate["frame"])
    if gif_frames and events:
        gif_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=gif_frames[1:],
            duration=max(120, int(1000 * gif_interval / fps)) if fps else 250,
            loop=0,
            optimize=True,
        )

    score = calculate_route_score(events)
    elapsed_seconds = perf_counter() - started_at
    risk_frame_ratio = risk_sampled_frames / sampled_frames if sampled_frames else 0.0
    return {
        "output_path": str(output_path),
        "preview_path": str(preview_path) if preview_candidate else "",
        "gif_path": str(gif_path) if gif_frames and events else "",
        "preview_timestamp": preview_timestamp,
        "preview_event": preview_event or {},
        "analyzer_version": ANALYZER_VERSION,
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
