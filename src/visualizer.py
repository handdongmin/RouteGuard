"""Rendering helpers for annotated videos and reports."""

import cv2


COLORS = {
    "safe": (80, 180, 80),
    "caution": (0, 190, 255),
    "danger": (0, 70, 255),
}


def draw_path_overlay(frame, path_polygon):
    """Draw the candidate walking path on a frame."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [path_polygon], (80, 200, 90))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [path_polygon], True, (60, 180, 80), 3)
    return frame


def draw_legend(frame):
    """Draw a compact visual legend for the annotated video."""
    items = [
        ("Path", (60, 180, 80)),
        ("Caution", COLORS["caution"]),
        ("Danger", COLORS["danger"]),
    ]
    x = 18
    y = 34
    cv2.rectangle(frame, (10, 10), (360, 48), (20, 20, 20), -1)
    for label, color in items:
        cv2.rectangle(frame, (x, y - 16), (x + 18, y + 2), color, -1)
        cv2.putText(frame, label, (x + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        x += 112
    return frame


def draw_measurements(frame, measurements, draw_safe=False):
    """Draw detections and path-overlap status on a frame."""
    for measurement in measurements:
        severity = measurement.get("severity", "safe")
        if severity == "safe" and not draw_safe:
            continue
        color = COLORS.get(severity, (220, 220, 220))
        x1, y1, x2, y2 = measurement["bbox"]
        label = measurement["label"]
        confidence = measurement["confidence"]
        overlap = measurement.get("overlap_ratio", 0.0)
        text = f"{label} {confidence:.2f} path {overlap:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        y_text = max(28, y1 - 10)
        cv2.rectangle(frame, (x1, y_text - 25), (min(x1 + 360, frame.shape[1] - 1), y_text + 5), color, -1)
        cv2.putText(frame, text, (x1 + 6, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def render_report(risks: list[dict]) -> list[str]:
    """Convert risk records into short timestamped report lines."""
    return [
        f"{risk.get('timestamp', '00:00')}  {risk.get('message', 'Potential risk')}"
        for risk in risks
    ]
