"""Frame-quality checks: telling a blinded depth camera from an empty room.

The thresholds here are the only thing standing between "the lens is covered"
and "the operator walked away", so the bench numbers they were derived from
are pinned as tests.
"""

import numpy as np
import pytest

from gaze_monitor.quality import (
    MIN_DEPTH_FRACTION,
    depth_is_blind,
    valid_depth_fraction,
)

# Measured on a D435i at 640x480, 60 frames covered and 90 clear.
# See jetson/CI_NOTES.md.
COVERED_BEST = 0.0121  # most valid depth seen with paper over the lens
CLEAR_WORST = 0.8830  # least valid depth seen with the lens clear


def _depth(valid_fraction: float, size: int = 10000) -> np.ndarray:
    """A depth frame with a known fraction of real readings."""
    frame = np.zeros(size, dtype=np.float32)
    frame[: int(round(size * valid_fraction))] = 1.5
    return frame


# -- valid_depth_fraction --------------------------------------------------


def test_fraction_counts_only_real_readings():
    """0.0 is the 'no reading' sentinel, not a distance of zero metres."""
    frame = np.array([0.0, 0.0, 1.5, 2.0], dtype=np.float32)
    assert valid_depth_fraction(frame) == 0.5


def test_fraction_of_a_fully_valid_frame_is_one():
    assert valid_depth_fraction(np.full(100, 2.0, dtype=np.float32)) == 1.0


def test_fraction_of_an_empty_or_absent_frame_is_zero():
    assert valid_depth_fraction(np.array([], dtype=np.float32)) == 0.0
    assert valid_depth_fraction(None) == 0.0


# -- depth_is_blind --------------------------------------------------------


def test_a_camera_without_depth_is_never_blind():
    """A webcam has no depth stream to lose. Faulting it would take every
    non-depth backend offline on the first frame."""
    assert depth_is_blind(None) is False


def test_covered_lens_reads_as_blind():
    assert depth_is_blind(_depth(COVERED_BEST)) is True
    assert depth_is_blind(_depth(0.0)) is True


def test_clear_lens_does_not_read_as_blind():
    assert depth_is_blind(_depth(CLEAR_WORST)) is False
    assert depth_is_blind(_depth(0.98)) is False


def test_the_threshold_sits_between_the_measured_populations():
    """The margin is the whole point: a threshold inside either population
    would either miss a covered lens or fault a working camera."""
    assert COVERED_BEST < MIN_DEPTH_FRACTION < CLEAR_WORST
    assert MIN_DEPTH_FRACTION / COVERED_BEST > 3
    assert CLEAR_WORST / MIN_DEPTH_FRACTION > 15


@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.049])
def test_below_threshold_is_blind(fraction):
    assert depth_is_blind(_depth(fraction), threshold=0.05) is True


@pytest.mark.parametrize("fraction", [0.05, 0.06, 0.5, 1.0])
def test_at_or_above_threshold_is_not_blind(fraction):
    assert depth_is_blind(_depth(fraction), threshold=0.05) is False


def test_a_zero_threshold_disables_the_check():
    """The escape hatch for an install where low depth return is normal."""
    assert depth_is_blind(_depth(0.0), threshold=0.0) is False
