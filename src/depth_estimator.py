"""Relative depth estimation interface."""


class DepthEstimator:
    """Estimate relative scene depth from a video frame."""

    def estimate(self, frame):
        """Return a relative depth map for a frame."""
        raise NotImplementedError
