"""Candidate evacuation-path analysis."""

import cv2
import numpy as np

from src.config import AnalysisConfig


LARGE_OBSTACLES = {
    "backpack",
    "chair",
    "couch",
    "handbag",
    "potted plant",
    "suitcase",
    "umbrella",
}

FURNITURE_OBSTACLES = {
    "bed",
    "bench",
    "dining table",
}

SMALL_CLUTTER = {
    "book",
    "bottle",
    "bowl",
    "cup",
    "keyboard",
    "laptop",
    "mouse",
    "skateboard",
    "sports ball",
    "teddy bear",
    "vase",
}

CUSTOM_RISKS = {
    "box",
    "cable",
    "cord",
    "electric outlet",
    "extension cord",
    "fire extinguisher",
    "fire_extinguisher",
    "multi tap",
    "outlet",
    "power outlet",
    "power strip",
    "power_strip",
    "socket",
    "wire",
}


class PathAnalyzer:
    """Measure how detected objects overlap a candidate walking path."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()

    def get_path_polygon(self, frame) -> np.ndarray:
        """Return a trapezoid representing the likely walking path."""
        height, width = frame.shape[:2]
        bottom_half_width = int(width * self.config.path_width_ratio / 2)
        top_half_width = int(width * self.config.path_width_ratio * 0.18)
        center_x = width // 2
        top_y = int(height * 0.45)

        return np.array(
            [
                [center_x - bottom_half_width, height - 1],
                [center_x + bottom_half_width, height - 1],
                [center_x + top_half_width, top_y],
                [center_x - top_half_width, top_y],
            ],
            dtype=np.int32,
        )

    def analyze(self, frame, detections, depth_map=None) -> list[dict]:
        """Return pathway occupancy measurements for a frame."""
        path_polygon = self.get_path_polygon(frame)
        path_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(path_mask, [path_polygon], 255)

        measurements = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(object_mask, (x1, y1), (x2, y2), 255, -1)

            object_area = int(np.count_nonzero(object_mask))
            overlap_area = int(np.count_nonzero(cv2.bitwise_and(object_mask, path_mask)))
            overlap_ratio = overlap_area / object_area if object_area else 0.0

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            bottom_center = ((x1 + x2) // 2, y2)
            center_in_path = cv2.pointPolygonTest(path_polygon, center, False) >= 0
            bottom_in_path = cv2.pointPolygonTest(path_polygon, bottom_center, False) >= 0
            frame_area = frame.shape[0] * frame.shape[1]
            bbox_area_ratio = object_area / frame_area if frame_area else 0.0
            risk_group = self._risk_group(detection["label"], bbox_area_ratio)
            severity = self._classify_severity(
                detection["label"],
                risk_group,
                bbox_area_ratio,
                overlap_ratio,
                center_in_path,
                bottom_in_path,
            )

            measurements.append(
                {
                    **detection,
                    "overlap_ratio": overlap_ratio,
                    "overlap_area": overlap_area,
                    "bbox_area_ratio": bbox_area_ratio,
                    "risk_group": risk_group,
                    "center_in_path": center_in_path,
                    "bottom_in_path": bottom_in_path,
                    "severity": severity,
                }
            )

        return measurements

    def _risk_group(self, label: str, bbox_area_ratio: float = 0.0) -> str:
        """Return a broad obstacle group for filtering and reporting."""
        if label in SMALL_CLUTTER:
            if bbox_area_ratio >= 0.12:
                return "large_clutter"
            return "small"
        if label in CUSTOM_RISKS:
            return "custom"
        if label in FURNITURE_OBSTACLES:
            return "furniture"
        if label in LARGE_OBSTACLES:
            return "large"
        return "other"

    def _classify_severity(
        self,
        label: str,
        risk_group: str,
        bbox_area_ratio: float,
        overlap_ratio: float,
        center_in_path: bool,
        bottom_in_path: bool,
    ) -> str:
        """Classify path risk with stricter rules for small clutter."""
        if risk_group == "large_clutter":
            min_area = 0.08
            danger_overlap = 0.18
            caution_overlap = 0.08
        elif label in SMALL_CLUTTER:
            min_area = 0.035
            danger_overlap = 0.62
            caution_overlap = 0.42
        elif risk_group == "custom":
            min_area = 0.01
            danger_overlap = 0.28
            caution_overlap = 0.12
        elif label in FURNITURE_OBSTACLES:
            min_area = 0.045
            danger_overlap = 0.45
            caution_overlap = 0.25
        elif label in LARGE_OBSTACLES:
            min_area = 0.022
            danger_overlap = 0.5
            caution_overlap = 0.25
        else:
            min_area = 0.04
            danger_overlap = 0.6
            caution_overlap = 0.4

        if bbox_area_ratio < min_area:
            return "safe"
        if overlap_ratio >= danger_overlap or (overlap_ratio >= danger_overlap * 0.75 and center_in_path):
            return "danger"
        if overlap_ratio >= caution_overlap and (center_in_path or bottom_in_path):
            return "caution"
        return "safe"
