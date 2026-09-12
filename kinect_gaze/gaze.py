"""Gaze math utilities and simple smoothing filter."""

import numpy as np


class GazeFilter:
    def __init__(self, alpha: float = 0.12):
        self.alpha = alpha
        self.state = None

    def apply(self, value: float) -> float:
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state


def iris_displacement(
    eye_iris_idx: int,
    l_corner_idx: int,
    r_corner_idx: int,
    face_landmarks,
    w: int,
    h: int,
) -> tuple[float, float, tuple[int, int]]:
    """Compute normalized dx, dy for one eye and return iris pixel coords.

    dx, dy are normalized relative to eye width: roughly -1..1 range.
    Returns: (dx, dy, (px, py))
    """
    iris = face_landmarks.landmark[eye_iris_idx]
    lc = face_landmarks.landmark[l_corner_idx]
    rc = face_landmarks.landmark[r_corner_idx]

    px, py = int(iris.x * w), int(iris.y * h)

    dx = (iris.x - (lc.x + rc.x) / 2.0) / ((rc.x - lc.x) / 2.0)
    dy = (iris.y - (lc.y + rc.y) / 2.0) / ((rc.x - lc.x) / 4.0)

    return dx, dy, (px, py)


def raw_depth_to_meters(raw_depth: int) -> float:
    """Convert Kinect v1 raw depth to meters (heuristic).

    Returns 0.0 for invalid or too-far values.
    """
    if raw_depth is None:
        return 0.0
    try:
        if raw_depth < 2047:
            return 0.1236 * np.tan(raw_depth / 2842.5 + 1.1863)
    except Exception:
        pass
    return 0.0
