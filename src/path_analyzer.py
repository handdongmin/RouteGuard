"""Candidate evacuation-path analysis."""


class PathAnalyzer:
    """Measure how detected objects overlap a candidate walking path."""

    def analyze(self, frame, detections, depth_map):
        """Return pathway occupancy measurements for a frame."""
        raise NotImplementedError
