"""Gaze geometry and smoothing.

Unit convention for the whole package: **depth is always metres**, and 0.0
means "no reading". Backends in :mod:`kinect_gaze.capture` convert from their
native units once, at the source, so nothing here needs to know whether the
frame came from a RealSense, a Kinect or a plain webcam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# MediaPipe FaceMesh landmark indices. The iris landmarks (468+) only exist
# when FaceMesh is constructed with refine_landmarks=True.
LEFT_IRIS = 468
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_IRIS = 473
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Eye width is the denominator of the normalisation below. At a steep head
# angle the two corners converge and the ratio explodes to inf/nan, which
# then latches permanently into the smoothing filter. Anything narrower than
# this (as a fraction of frame width) is reported as unmeasurable instead.
MIN_EYE_WIDTH = 1e-3


@dataclass(frozen=True)
class EyeMeasurement:
    """Normalised iris offset for a single eye."""

    dx: float
    dy: float
    px: int
    py: int


@dataclass(frozen=True)
class GazeSample:
    """Both-eye average gaze offset plus the distance to the operator."""

    dx: float
    dy: float
    z_m: float
    px: int
    py: int


class GazeFilter:
    """Exponential moving average with a non-finite guard.

    A single inf/nan reaching a naive EMA poisons its state forever, and the
    monitor goes on drawing a confident cursor from garbage. Non-finite inputs
    are dropped here instead of being folded into the state.
    """

    def __init__(self, alpha: float = 0.12):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.state: float | None = None

    def apply(self, value: float) -> float | None:
        """Fold ``value`` into the average; return the new state.

        Returns ``None`` only while the filter has never seen a finite value.
        """
        value = float(value)
        if not math.isfinite(value):
            return self.state
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state

    def reset(self) -> None:
        self.state = None


def eye_displacement(
    face_landmarks,
    iris_idx: int,
    corner_a_idx: int,
    corner_b_idx: int,
    w: int,
    h: int,
) -> EyeMeasurement | None:
    """Normalised iris offset for one eye, or ``None`` if unmeasurable.

    ``dx`` is the iris offset from the eye centre in units of half the eye
    width, so it spans roughly -1..1. ``dy`` uses a quarter of the eye *width*
    as its scale: the palpebral fissure is far shorter vertically than
    horizontally, and this keeps the two axes on a comparable scale without
    needing eyelid landmarks, which move when the operator blinks.
    """
    iris = face_landmarks.landmark[iris_idx]
    a = face_landmarks.landmark[corner_a_idx]
    b = face_landmarks.landmark[corner_b_idx]

    eye_width = abs(b.x - a.x)
    if eye_width < MIN_EYE_WIDTH:
        return None

    cx = (a.x + b.x) / 2.0
    cy = (a.y + b.y) / 2.0
    dx = (iris.x - cx) / (eye_width / 2.0)
    dy = (iris.y - cy) / (eye_width / 4.0)

    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None

    return EyeMeasurement(
        dx=dx,
        dy=dy,
        px=int(np.clip(iris.x, 0.0, 1.0) * (w - 1)),
        py=int(np.clip(iris.y, 0.0, 1.0) * (h - 1)),
    )


def depth_at(depth_m: np.ndarray | None, px: int, py: int, patch: int = 5) -> float:
    """Median valid depth in metres over a small patch around (px, py).

    Depth maps are holey: a single-pixel probe on a RealSense lands in a
    shadow or a specular dropout often enough to matter, and returns 0. The
    median of a patch, ignoring invalid pixels, is far steadier for the price
    of a few dozen comparisons.

    Returns 0.0 when there is no usable reading.
    """
    if depth_m is None:
        return 0.0

    h, w = depth_m.shape[:2]
    if not (0 <= px < w and 0 <= py < h):
        return 0.0

    r = max(0, patch // 2)
    window = depth_m[
        max(0, py - r) : min(h, py + r + 1),
        max(0, px - r) : min(w, px + r + 1),
    ]
    valid = window[np.isfinite(window) & (window > 0.0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def gaze_from_landmarks(
    face_landmarks, w: int, h: int, depth_m: np.ndarray | None = None
) -> GazeSample | None:
    """Average both eyes into one gaze sample, or ``None`` if unmeasurable.

    Both eyes must be measurable. A one-eyed estimate means the operator is in
    near-profile, where the iris offset no longer tracks gaze direction, and
    reporting it as a confident sample is worse than reporting nothing.
    """
    left = eye_displacement(
        face_landmarks, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, w, h
    )
    right = eye_displacement(
        face_landmarks, RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, w, h
    )
    if left is None or right is None:
        return None

    px = (left.px + right.px) // 2
    py = (left.py + right.py) // 2
    return GazeSample(
        dx=(left.dx + right.dx) / 2.0,
        dy=(left.dy + right.dy) / 2.0,
        z_m=depth_at(depth_m, px, py),
        px=px,
        py=py,
    )
