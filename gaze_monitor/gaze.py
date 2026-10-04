"""Gaze geometry and smoothing.

Unit convention for the whole package: **depth is always metres**, and 0.0
means "no reading". Backends in :mod:`gaze_monitor.capture` convert from their
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

# The iris model emits five points per eye: a centre and four on the limbus
# at quarter turns. Only the centre was being used, which threw away four
# further estimates of the same quantity for nothing. Averaging all five is
# the cheapest noise reduction available here.
#
# This leaves the mean position alone as long as the four ring points are
# symmetric about the centre, which they are for a circle fit. The test below
# pins that down for a synthetic ring; on a real face it is an assumption
# about MediaPipe's output rather than something this code can enforce, so
# the calibration should be re-taken rather than trusted blindly.
LEFT_IRIS_RING = (469, 470, 471, 472)
RIGHT_IRIS_RING = (474, 475, 476, 477)

# FaceMesh emits exactly 468 landmarks, or exactly 478 with refine_landmarks.
# There is no third case, so the ring is either all there or none of it is.
# Checking indices one at a time is not the same test: a list long enough to
# contain the left ring but not the right would pass it per index and quietly
# measure one eye from placeholder zeros.
REFINED_LANDMARK_COUNT = 478

# The eye-width denominator changes only with head pose and distance, so it
# is a slow signal riding under fast landmark noise. The iris offset in the
# numerator is the fast signal and must not be smoothed here, or the whole
# measurement lags. Smoothing one and not the other is what stops denominator
# noise multiplying into the result, at no lag cost to the thing being
# measured. 0.05 is about a 20-frame time constant, under a second at 30 fps,
# which tracks a head turn without chasing the jitter.
WIDTH_ALPHA = 0.05

# Eye width is the denominator of the normalisation below. At a steep head
# angle the two corners converge and the ratio explodes to inf/nan, which
# then latches permanently into the smoothing filter. A narrower eye than
# this, in pixels, is reported as unmeasurable instead.
#
# Pixels rather than a fraction of frame width, because all the geometry here
# is done in pixels now. Normalised landmark coordinates divide x by the
# frame width and y by its height, so they are anisotropic on any frame that
# is not square: a circle in the image is an ellipse in normalised space, and
# an expression mixing the two axes silently picks up a factor of the aspect
# ratio. The old dy did exactly that, dividing a normalised-y offset by a
# scale taken from normalised x, so the same physical eye read 33% higher on
# 16:9 than on 4:3. Working in pixels makes that class of mistake impossible.
MIN_EYE_WIDTH_PX = 1.0


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
    # How far the two eyes disagreed vertically on this frame.
    #
    # Free information, and the only unambiguous noise measurement available
    # here. Eye movements are conjugate: the eyes rotate together, and the
    # one legitimate difference between them is vergence, which is horizontal
    # and set by target distance. There is no such thing as vertical
    # vergence above this instrument's noise floor, so whatever appears in
    # dy_left - dy_right was put there by the measurement and not by the
    # operator. Large values mean this frame should not be trusted: a blink
    # halfway through, an occluded eye, a head angle too steep for one iris.
    disparity: float = 0.0


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


def iris_centre(face_landmarks, iris_idx: int, ring: tuple[int, ...]):
    """Mean of the iris centre and its limbus ring, as normalised (x, y).

    Falls back to the centre landmark alone when the ring is absent, which is
    what a FaceMesh built without ``refine_landmarks`` gives and what the
    older synthetic fixtures provide.
    """
    points = [face_landmarks.landmark[iris_idx]]
    if ring and len(face_landmarks.landmark) >= REFINED_LANDMARK_COUNT:
        points.extend(face_landmarks.landmark[i] for i in ring)
    n = len(points)
    return (
        sum(p.x for p in points) / n,
        sum(p.y for p in points) / n,
    )


def eye_displacement(
    face_landmarks,
    iris_idx: int,
    corner_a_idx: int,
    corner_b_idx: int,
    w: int,
    h: int,
    ring: tuple[int, ...] = (),
    eye_width: float | None = None,
) -> EyeMeasurement | None:
    """Normalised iris offset for one eye, or ``None`` if unmeasurable.

    ``dx`` is the iris offset from the eye centre in units of half the eye
    width, so it spans roughly -1..1. ``dy`` uses a quarter of the eye *width*
    as its scale: the palpebral fissure is far shorter vertically than
    horizontally, and this keeps the two axes on a comparable scale without
    needing eyelid landmarks, which move when the operator blinks.
    """
    nx, ny = iris_centre(face_landmarks, iris_idx, ring)
    a = face_landmarks.landmark[corner_a_idx]
    b = face_landmarks.landmark[corner_b_idx]

    # Into pixels before any geometry. See MIN_EYE_WIDTH_PX.
    ix, iy = nx * w, ny * h
    ax, ay = a.x * w, a.y * h
    bx, by = b.x * w, b.y * h

    # The eye's own axes, taken from this frame: along the palpebral fissure
    # and across it. Measuring in image axes instead makes a head tilt read
    # as a vertical eye movement, because the offset is then projected onto
    # the wrong pair of directions. It went as tan(roll), so ten degrees of
    # head tilt -- which people do constantly -- produced a tenth of the
    # calibrated zone in false elevation.
    vx, vy = bx - ax, by - ay
    measured_width = math.hypot(vx, vy)
    if measured_width < MIN_EYE_WIDTH_PX:
        return None
    # A caller holding a smoothed width passes it in; the raw measurement
    # still has to pass the degeneracy check above, because a smoothed width
    # from a better frame would otherwise license dividing by this bad one.
    if eye_width is None:
        eye_width = measured_width
    elif eye_width < MIN_EYE_WIDTH_PX:
        return None

    # Orientation from this frame, scale possibly smoothed from many. Head
    # orientation is fast and has to be current; the scale is slow.
    ux, uy = vx / measured_width, vy / measured_width
    ox = ix - (ax + bx) / 2.0
    oy = iy - (ay + by) / 2.0
    along = ox * ux + oy * uy
    across = -ox * uy + oy * ux

    dx = along / (eye_width / 2.0)
    dy = across / (eye_width / 4.0)

    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None

    return EyeMeasurement(
        dx=dx,
        dy=dy,
        px=int(np.clip(nx, 0.0, 1.0) * (w - 1)),
        py=int(np.clip(ny, 0.0, 1.0) * (h - 1)),
    )


# The eyes, as (iris index, corner a, corner b), in the same argument order
# eye_displacement takes them. One tuple so the forward and inverse transforms
# cannot disagree about which corner is which.
EYES = (
    (LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_IRIS_RING),
    (RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_IRIS_RING),
)


def project_gaze(landmarks: np.ndarray, dx: float, dy: float, w: int, h: int):
    """Where each iris would sit if both eyes shared one gaze estimate.

    The exact inverse of :func:`eye_displacement`, and deliberately written
    next to it: the two have to use the same normalisation, and a change to
    one that misses the other produces a preview that disagrees with the
    measurement in a way nobody would spot by looking.

    Takes the plain ``(N, 3)`` array the preview carries rather than
    MediaPipe's protobuf, because this is called on an HTTP thread where the
    protobuf is long gone.

    This is what makes the preview honest about binocular vision. Eyes move
    together, so one shared ``(dx, dy)`` drives both markers and they move
    together by construction. The raw per-eye landmarks do not, because
    MediaPipe infers each iris from its own eye crop with no binocular
    constraint anywhere in the model.

    Needs the frame size, because the geometry is done in pixels and
    normalised coordinates are anisotropic on a frame that is not square.

    Returns one normalised ``(x, y)`` per eye, or ``None`` for an eye whose
    corners are too close together to divide by.
    """
    out = []
    for _, a_idx, b_idx, _ring in EYES:
        if max(a_idx, b_idx) >= len(landmarks):
            out.append(None)
            continue
        ax, ay = float(landmarks[a_idx][0]) * w, float(landmarks[a_idx][1]) * h
        bx, by = float(landmarks[b_idx][0]) * w, float(landmarks[b_idx][1]) * h
        vx, vy = bx - ax, by - ay
        eye_width = math.hypot(vx, vy)
        if eye_width < MIN_EYE_WIDTH_PX:
            out.append(None)
            continue
        ux, uy = vx / eye_width, vy / eye_width
        along = dx * eye_width / 2.0
        across = dy * eye_width / 4.0
        cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
        # Back out of the eye's frame into pixels, then into normalised.
        out.append(
            (
                (cx + along * ux - across * uy) / w,
                (cy + along * uy + across * ux) / h,
            )
        )
    return out


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


class GazeMeasurer:
    """Landmarks into a fused gaze sample, holding the slow-moving state.

    The state is one smoothed eye width per eye, and it is the only reason
    this is a class rather than a function. ``dx`` is a ratio, so noise in
    the denominator multiplies into the answer: because the two eyes measure
    their widths separately, that noise is also uncorrelated between them and
    shows up as the eyes appearing to disagree. Eye width moves only with
    head pose and distance, which is slow, so smoothing it costs the
    measurement no lag while removing most of that term.

    Per eye rather than one shared face scale, deliberately. A shared scale
    would cancel better, but under head yaw the two eyes foreshorten by
    different amounts and a single number would misscale whichever eye is
    further away.
    """

    def __init__(self, width_alpha: float = WIDTH_ALPHA) -> None:
        self._widths = tuple(GazeFilter(alpha=width_alpha) for _ in EYES)

    def reset(self) -> None:
        """Forget the smoothed widths. For a new face, or a new calibration."""
        for f in self._widths:
            f.reset()

    def measure(
        self, face_landmarks, w: int, h: int, depth_m: np.ndarray | None = None
    ) -> GazeSample | None:
        """Average both eyes into one gaze sample, or ``None`` if unmeasurable.

        Both eyes must be measurable. A one-eyed estimate means the operator
        is in near-profile, where the iris offset no longer tracks gaze
        direction, and reporting it as a confident sample is worse than
        reporting nothing.
        """
        eyes = []
        for (iris, a_idx, b_idx, ring), width in zip(EYES, self._widths, strict=True):
            a = face_landmarks.landmark[a_idx]
            b = face_landmarks.landmark[b_idx]
            raw_width = math.hypot((b.x - a.x) * w, (b.y - a.y) * h)
            smoothed = width.apply(raw_width)
            eyes.append(
                eye_displacement(
                    face_landmarks,
                    iris,
                    a_idx,
                    b_idx,
                    w,
                    h,
                    ring=ring,
                    eye_width=smoothed,
                )
            )
        left, right = eyes
        if left is None or right is None:
            return None

        px = (left.px + right.px) // 2
        py = (left.py + right.py) // 2
        return GazeSample(
            # The version component: the eyes' shared rotation, which is the
            # gaze direction. Averaging also cancels vergence, the one real
            # difference between the two eyes, which is why it is the right
            # operation and not merely a convenient one.
            dx=(left.dx + right.dx) / 2.0,
            dy=(left.dy + right.dy) / 2.0,
            z_m=depth_at(depth_m, px, py),
            px=px,
            py=py,
            disparity=abs(left.dy - right.dy),
        )


def gaze_from_landmarks(
    face_landmarks, w: int, h: int, depth_m: np.ndarray | None = None
) -> GazeSample | None:
    """One-shot measurement with no smoothed width. Single frames and tests.

    Equivalent to a fresh :class:`GazeMeasurer` on its first frame, where the
    filters have no state and the raw width is used as-is.
    """
    return GazeMeasurer().measure(face_landmarks, w, h, depth_m)
