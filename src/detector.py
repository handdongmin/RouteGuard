"""Object detection interface."""


class ObjectDetector:
    """Detect candidate obstacles in video frames."""

    def detect(self, frame):
        """Return detected obstacle candidates for a frame."""
        raise NotImplementedError
