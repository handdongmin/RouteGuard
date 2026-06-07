"""Rendering helpers for annotated videos and reports."""

import cv2


COLORS = {
    "safe": (80, 180, 80),
    "caution": (0, 190, 255),
    "danger": (0, 70, 255),
}

SEVERITY_TEXT = {
    "safe": "SAFE",
    "caution": "CAUTION",
    "danger": "DANGER",
}


def _display_label(label: str, severity: str) -> str:
    """Make labels honest when a pretrained model guesses the exact object class."""
    if severity == "safe":
        return label
    return f"obstacle ({label})"


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


def _clip_box(frame, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def _draw_text_panel(frame, x: int, y: int, text: str, color, scale: float = 0.62) -> None:
    """Draw readable text with a filled background."""
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x1, y1, x2, y2 = _clip_box(frame, x, y - text_height - 12, x + text_width + 16, y + baseline + 8)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(frame, text, (x1 + 8, y2 - baseline - 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)


def _draw_corner_box(frame, x1: int, y1: int, x2: int, y2: int, color, thickness: int) -> None:
    """Draw a strong detector box with emphasized corners."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    corner = max(16, min(x2 - x1, y2 - y1) // 5)
    cv2.line(frame, (x1, y1), (x1 + corner, y1), (255, 255, 255), thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), (255, 255, 255), thickness)
    cv2.line(frame, (x2, y1), (x2 - corner, y1), (255, 255, 255), thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), (255, 255, 255), thickness)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), (255, 255, 255), thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), (255, 255, 255), thickness)
    cv2.line(frame, (x2, y2), (x2 - corner, y2), (255, 255, 255), thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), (255, 255, 255), thickness)


def _draw_blocker_summary(frame, blockers: list[dict]) -> None:
    """Draw a compact list of the detected blockers on the frame."""
    if not blockers:
        return

    width = frame.shape[1]
    panel_width = min(520, width - 24)
    panel_height = 46 + min(len(blockers), 4) * 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 58), (12 + panel_width, 58 + panel_height), (5, 12, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (12, 58), (12 + panel_width, 58 + panel_height), (120, 170, 255), 2)
    cv2.putText(frame, "Blocking objects", (28, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    for index, blocker in enumerate(blockers[:4], start=1):
        y = 90 + index * 28
        severity = blocker.get("severity", "safe")
        color = COLORS.get(severity, (220, 220, 220))
        label = _display_label(blocker.get("label", "object"), severity)
        overlap = blocker.get("overlap_ratio", 0.0)
        text = f"{index}. {SEVERITY_TEXT.get(severity, severity).title()} - {label} / path {overlap:.0%}"
        cv2.circle(frame, (34, y - 5), 9, color, -1)
        cv2.putText(frame, text, (52, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 248, 255), 2)


def draw_measurements(frame, measurements, draw_safe=False):
    """Draw detections and path-overlap status on a frame."""
    visible_measurements = []
    for measurement in measurements:
        severity = measurement.get("severity", "safe")
        if severity == "safe" and not draw_safe:
            continue
        visible_measurements.append(measurement)

    visible_measurements.sort(
        key=lambda item: (item.get("severity") == "danger", item.get("severity") == "caution", item.get("overlap_ratio", 0.0)),
        reverse=True,
    )
    blockers = [measurement for measurement in visible_measurements if measurement.get("severity") != "safe"]
    _draw_blocker_summary(frame, blockers)

    for index, measurement in enumerate(visible_measurements, start=1):
        severity = measurement.get("severity", "safe")
        color = COLORS.get(severity, (220, 220, 220))
        x1, y1, x2, y2 = measurement["bbox"]
        x1, y1, x2, y2 = _clip_box(frame, x1, y1, x2, y2)
        label = _display_label(measurement["label"], severity)
        confidence = measurement["confidence"]
        overlap = measurement.get("overlap_ratio", 0.0)
        severity_label = SEVERITY_TEXT.get(severity, severity.upper())
        text = f"{severity_label}: {label} {confidence:.2f} / path {overlap:.0%}"

        if severity != "safe":
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

        thickness = 5 if severity == "danger" else 4
        _draw_corner_box(frame, x1, y1, x2, y2, color, thickness)

        badge_radius = 15
        cv2.circle(frame, (min(x2 - badge_radius, x1 + badge_radius + 2), max(y1 + badge_radius + 2, badge_radius + 2)), badge_radius, color, -1)
        cv2.putText(
            frame,
            str(index),
            (min(x2 - badge_radius, x1 + badge_radius - 6), max(y1 + badge_radius + 9, badge_radius + 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

        y_text = max(36, y1 - 10 - (index - 1) * 34)
        _draw_text_panel(frame, x1, y_text, text, color, scale=0.62)

    return frame


def render_report(risks: list[dict]) -> list[str]:
    """Convert risk records into short timestamped report lines."""
    return [
        f"{risk.get('timestamp', '00:00')}  {risk.get('message', 'Potential risk')}"
        for risk in risks
    ]
