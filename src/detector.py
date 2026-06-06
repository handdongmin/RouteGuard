"""YOLO-based obstacle detection."""

from collections.abc import Collection
from typing import Any

from src.config import AnalysisConfig


DEFAULT_OBSTACLE_CLASSES = {
    "backpack",
    "bed",
    "bench",
    "book",
    "bottle",
    "bowl",
    "chair",
    "couch",
    "cup",
    "dining table",
    "handbag",
    "keyboard",
    "laptop",
    "mouse",
    "potted plant",
    "skateboard",
    "sports ball",
    "suitcase",
    "teddy bear",
    "umbrella",
    "vase",
}


class ObjectDetector:
    """Detect candidate obstacles in video frames."""

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        obstacle_classes: Collection[str] | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or AnalysisConfig()
        self.obstacle_classes = set(obstacle_classes or DEFAULT_OBSTACLE_CLASSES)

        if model is None:
            from ultralytics import YOLO

            model = YOLO(self.config.model_name)

        self.model = model

    def detect(self, frame) -> list[dict]:
        """Return obstacle candidates with labels, confidence scores, and boxes."""
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            imgsz=self.config.inference_image_size,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detections = []
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            label = result.names[int(class_id)]
            if label not in self.obstacle_classes:
                continue

            x1, y1, x2, y2 = (int(value) for value in box)
            detections.append(
                {
                    "label": label,
                    "confidence": float(confidence),
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return detections
