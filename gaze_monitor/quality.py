"""Is a frame usable?

A camera whose lens is covered does not stop working. It goes on delivering
well-formed frames at the full rate, so :meth:`CameraSource.read` has nothing
to report and the monitor's camera-health timer never starts. What changes is
the *content*: a depth camera stops getting returns.

Measured on a D435i at 640x480 (see jetson/CI_NOTES.md):

===================  ==========================
lens clear           88.3% - 98.1% valid depth
lens covered (paper)  0.0% -  1.2% valid depth
===================  ==========================

Two populations separated by a factor of about 70, which is what makes this
worth testing for at all. Note that colour brightness does **not** separate
them: the covered frames were *brighter* and flatter than the clear ones
(mean 113.6 against 63.0), because the paper was lit. Anything keyed on
darkness would have been backwards.

Kept out of capture.py, which needs OpenCV, so the threshold logic can be
tested on a machine with no vision stack.
"""

from __future__ import annotations

import numpy as np

# Below this fraction of pixels carrying a reading, the depth stream is
# returning essentially nothing. 5% sits about 17x below the worst clear frame
# measured and about 4x above the best obstructed one, so it is nowhere near
# either population.
MIN_DEPTH_FRACTION = 0.05


def valid_depth_fraction(depth_m: np.ndarray | None) -> float:
    """Fraction of pixels carrying a real distance reading.

    Depth is metres with 0.0 meaning "no reading" -- see
    :mod:`gaze_monitor.capture`, which converts every backend to that
    convention at the source.
    """
    if depth_m is None or depth_m.size == 0:
        return 0.0
    return float((depth_m > 0.0).mean())


def depth_is_blind(
    depth_m: np.ndarray | None, threshold: float = MIN_DEPTH_FRACTION
) -> bool:
    """True if a depth stream that should be returning something is not.

    ``None`` is not blind. A camera with no depth at all -- a webcam, a CSI
    sensor -- has no depth stream to lose, so the question does not apply to
    it and this must not be read as a fault. That is why ``None`` gives a
    fraction of 0.0 here but is still not blind: the two mean different
    things, and conflating them would fault every webcam on the first frame.

    A threshold of 0.0 disables the check.
    """
    if depth_m is None or threshold <= 0.0:
        return False
    return valid_depth_fraction(depth_m) < threshold
