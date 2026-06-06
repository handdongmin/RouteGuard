"""Tests for YOLO obstacle result conversion."""

from types import SimpleNamespace

from src.detector import ObjectDetector


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    def predict(self, **kwargs):
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[10.2, 20.8, 100.9, 200.1], [0, 0, 20, 20]]),
            conf=FakeTensor([0.91, 0.8]),
            cls=FakeTensor([0, 1]),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "chair", 1: "person"})]


def test_detect_returns_only_obstacle_classes():
    detector = ObjectDetector(model=FakeModel())

    detections = detector.detect(frame=object())

    assert detections == [
        {
            "label": "chair",
            "confidence": 0.91,
            "bbox": (10, 20, 100, 200),
        }
    ]
