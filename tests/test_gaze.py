"""Tests for the gaze geometry helpers.

These cover the failure modes that used to reach the UI silently: a
non-finite displacement latching into the smoothing filter, and a
single-pixel depth probe landing in a hole.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from gaze_monitor.gaze import (
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_IRIS,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_IRIS,
    GazeFilter,
    depth_at,
    eye_displacement,
    gaze_from_landmarks,
)


class FakeLandmarks:
    """Stands in for a MediaPipe NormalizedLandmarkList."""

    def __init__(self, points):
        size = max(points) + 1
        self.landmark = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(size)]
        for idx, (x, y) in points.items():
            self.landmark[idx] = SimpleNamespace(x=x, y=y, z=0.0)


def centered_eye(iris_x=0.5, iris_y=0.5, half_width=0.05):
    return {
        LEFT_IRIS: (iris_x, iris_y),
        LEFT_EYE_OUTER: (iris_x - half_width, iris_y),
        LEFT_EYE_INNER: (iris_x + half_width, iris_y),
        RIGHT_IRIS: (iris_x + 0.2, iris_y),
        RIGHT_EYE_INNER: (iris_x + 0.2 - half_width, iris_y),
        RIGHT_EYE_OUTER: (iris_x + 0.2 + half_width, iris_y),
    }


# --- GazeFilter -----------------------------------------------------------


def test_filter_converges_toward_input():
    f = GazeFilter(alpha=0.5)
    assert f.apply(1.0) == 1.0  # first sample seeds the state
    assert f.apply(0.0) == pytest.approx(0.5)
    assert f.apply(0.0) == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_filter_does_not_latch_on_non_finite(bad):
    """A single inf/nan used to poison the EMA state permanently."""
    f = GazeFilter(alpha=0.5)
    f.apply(1.0)
    assert f.apply(bad) == 1.0  # rejected, state untouched
    assert math.isfinite(f.state)
    assert f.apply(0.0) == pytest.approx(0.5)  # still responsive afterwards


def test_filter_returns_none_until_first_finite_sample():
    f = GazeFilter()
    assert f.apply(float("nan")) is None
    assert f.state is None


def test_filter_reset_clears_state():
    f = GazeFilter(alpha=0.5)
    f.apply(1.0)
    f.reset()
    assert f.state is None
    assert f.apply(3.0) == 3.0


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_filter_rejects_out_of_range_alpha(alpha):
    with pytest.raises(ValueError):
        GazeFilter(alpha=alpha)


# --- eye_displacement -----------------------------------------------------


def test_centered_iris_gives_zero_displacement():
    lm = FakeLandmarks(centered_eye())
    eye = eye_displacement(lm, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, 640, 480)
    assert eye is not None
    assert eye.dx == pytest.approx(0.0)
    assert eye.dy == pytest.approx(0.0)


def test_iris_at_corner_gives_unit_displacement():
    pts = centered_eye(half_width=0.05)
    pts[LEFT_IRIS] = (0.55, 0.5)  # sitting on the inner corner
    lm = FakeLandmarks(pts)
    eye = eye_displacement(lm, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, 640, 480)
    assert eye.dx == pytest.approx(1.0)


def test_degenerate_eye_returns_none_instead_of_inf():
    """Near-profile heads collapse the eye width; this used to divide by ~0."""
    pts = centered_eye(half_width=0.0)
    lm = FakeLandmarks(pts)
    assert (
        eye_displacement(lm, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, 640, 480)
        is None
    )


def test_pixel_coordinates_stay_in_bounds():
    pts = centered_eye(iris_x=1.4, iris_y=-0.3)  # landmarks may fall outside the frame
    lm = FakeLandmarks(pts)
    eye = eye_displacement(lm, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, 640, 480)
    assert 0 <= eye.px < 640
    assert 0 <= eye.py < 480


# --- depth_at -------------------------------------------------------------


def test_depth_at_returns_zero_without_depth():
    assert depth_at(None, 10, 10) == 0.0


def test_depth_at_ignores_holes():
    """A zero at the exact probe pixel must not be read as "0 metres away"."""
    depth = np.full((20, 20), 1.5, dtype=np.float32)
    depth[10, 10] = 0.0  # dropout right where we probe
    assert depth_at(depth, 10, 10) == pytest.approx(1.5)


def test_depth_at_all_holes_is_zero():
    depth = np.zeros((20, 20), dtype=np.float32)
    assert depth_at(depth, 10, 10) == 0.0


def test_depth_at_is_median_not_mean():
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    depth[9, 9] = 40.0  # a flyer that would drag a mean badly
    assert depth_at(depth, 10, 10, patch=5) == pytest.approx(2.0)


def test_depth_at_out_of_bounds_is_zero():
    depth = np.full((20, 20), 1.5, dtype=np.float32)
    assert depth_at(depth, 999, 5) == 0.0
    assert depth_at(depth, -1, 5) == 0.0


def test_depth_at_ignores_non_finite():
    depth = np.full((20, 20), np.nan, dtype=np.float32)
    depth[10, 11] = 1.2
    assert depth_at(depth, 10, 10) == pytest.approx(1.2)


# --- gaze_from_landmarks --------------------------------------------------


def test_gaze_averages_both_eyes():
    lm = FakeLandmarks(centered_eye())
    sample = gaze_from_landmarks(lm, 640, 480)
    assert sample is not None
    assert sample.dx == pytest.approx(0.0)
    assert sample.z_m == 0.0  # no depth map supplied


def test_gaze_requires_both_eyes():
    pts = centered_eye()
    pts[RIGHT_EYE_INNER] = (0.7, 0.5)
    pts[RIGHT_EYE_OUTER] = (0.7, 0.5)  # right eye collapsed
    lm = FakeLandmarks(pts)
    assert gaze_from_landmarks(lm, 640, 480) is None


def test_gaze_reads_depth_when_available():
    lm = FakeLandmarks(centered_eye())
    depth = np.full((480, 640), 0.85, dtype=np.float32)
    sample = gaze_from_landmarks(lm, 640, 480, depth_m=depth)
    assert sample.z_m == pytest.approx(0.85)
