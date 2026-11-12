"""Head pose, and gaze as head pose plus the eye's rotation in the head.

The iris-in-eye measurement in :mod:`gaze_monitor.gaze` sees only the eye's
rotation *in the head*. That is gaze only while the head stays where it was
at calibration. An operator who looks away by turning their head, eyes
centred, barely moves it; one who turns their head while keeping their eyes
on the blade moves it the other way. So the line of sight is the head's
orientation and the eye's rotation in the head, combined.

Head orientation comes from fitting MediaPipe's canonical face model to the
landmarks on the rigid parts of the face -- forehead, nose, cheekbones, eye
corners -- never the lids, brows or mouth, which move on their own:

* **With depth**, each landmark is lifted to a 3D point in the camera frame
  from the aligned depth image and the camera intrinsics, and the model is
  fitted to those points by a similarity transform (Umeyama), dropping
  outliers. That is metric: it also gives the face's scale, and so the
  operator's actual eye width, which converts the iris offset to an angle.
* **Without depth** (a webcam, or a frame where too much depth is missing),
  the same fit runs on FaceMesh's own relative depth. Weak-perspective and
  noisier, but always available.

The output is a gaze direction in the camera frame, as yaw and pitch in
degrees. Calibration and the attention zone work in those units unchanged.

Canonical model coordinates below are MediaPipe's canonical face model,
Copyright The MediaPipe Authors, Apache License 2.0
(mediapipe/modules/face_geometry/data/canonical_face_model.obj), in metres:
``+x`` is the subject's left, ``+y`` up, ``+z`` out of the face.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

# Rigid landmarks: index -> canonical position, metres.
CANONICAL = {
    10: (+0.00000, +0.08262, +0.04482),
    151: (+0.00000, +0.06545, +0.05027),
    9: (+0.00000, +0.04886, +0.05385),
    8: (+0.00000, +0.04019, +0.05285),
    168: (+0.00000, +0.03271, +0.05236),
    6: (+0.00000, +0.02473, +0.05789),
    197: (+0.00000, +0.01728, +0.06317),
    195: (+0.00000, +0.01059, +0.06775),
    5: (+0.00000, +0.00366, +0.07243),
    4: (+0.00000, -0.00463, +0.07587),
    1: (+0.00000, -0.01127, +0.07476),
    33: (-0.04446, +0.02664, +0.03173),
    133: (-0.01856, +0.02585, +0.03758),
    362: (+0.01856, +0.02585, +0.03758),
    263: (+0.04446, +0.02664, +0.03173),
    116: (-0.06465, +0.00937, +0.01690),
    345: (+0.06465, +0.00937, +0.01690),
    123: (-0.06338, -0.00529, +0.01881),
    352: (+0.06338, -0.00529, +0.01881),
    50: (-0.04766, -0.00702, +0.03535),
    280: (+0.04766, -0.00702, +0.03535),
    117: (-0.05259, +0.00946, +0.02974),
    346: (+0.05259, +0.00946, +0.02974),
    234: (-0.07664, +0.00673, -0.02436),
    454: (+0.07664, +0.00673, -0.02436),
    127: (-0.07743, +0.02365, -0.02005),
    356: (+0.07743, +0.02365, -0.02005),
    54: (-0.06279, +0.06615, +0.01426),
    284: (+0.06279, +0.06615, +0.01426),
    103: (-0.05133, +0.07486, +0.02660),
    332: (+0.05133, +0.07486, +0.02660),
    109: (-0.01891, +0.08236, +0.04275),
    338: (+0.01891, +0.08236, +0.04275),
    67: (-0.03524, +0.08006, +0.03729),
    297: (+0.03524, +0.08006, +0.03729),
    104: (-0.04539, +0.06320, +0.03683),
    333: (+0.04539, +0.06320, +0.03683),
    69: (-0.03064, +0.06566, +0.04530),
    299: (+0.03064, +0.06566, +0.04530),
}
RIGID = np.array(sorted(CANONICAL))
MODEL = np.array([CANONICAL[i] for i in RIGID])
# Eye width of the canonical face, corner to corner, metres.
CANONICAL_EYE_WIDTH = 0.026557

# Distance from the eye's centre of rotation to the plane of the iris. With
# the eye width, it converts the iris offset FaceMesh measures (a fraction
# of the half-width) into an angle. About 10.5 mm in adults, and one of the
# least variable dimensions of the eye.
IRIS_PLANE_RADIUS = 0.0105

MIN_POINTS = 8
OUTLIER_M = 0.012  # a lifted point this far off the fitted model is dropped


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of the colour image the landmarks were found in."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in the camera frame (OpenCV axes).

    ``rotation`` maps the canonical face frame into the camera frame.
    ``scale`` is the face's size relative to the canonical face (metric
    poses only). ``method`` is ``"depth"`` or ``"landmarks"``.
    """

    rotation: np.ndarray
    scale: float
    method: str
    points: int
    residual_m: float = 0.0

    @property
    def eye_width_m(self) -> float:
        if self.method == "depth":
            return CANONICAL_EYE_WIDTH * self.scale
        return CANONICAL_EYE_WIDTH

    def angles_deg(self) -> tuple[float, float, float]:
        """Yaw, pitch, roll of the face, degrees.

        Yaw is positive with the face turned towards the camera's +x (image
        right), pitch positive with the face tipped down, roll positive
        clockwise in the image. A face looking straight into the camera is
        (0, 0, 0).
        """
        fwd = self.rotation @ np.array([0.0, 0.0, 1.0])  # out of the face
        right = self.rotation @ np.array([1.0, 0.0, 0.0])  # towards image right
        yaw = math.degrees(math.atan2(fwd[0], -fwd[2]))
        pitch = math.degrees(math.atan2(fwd[1], math.hypot(fwd[0], fwd[2])))
        # Roll about the face's own forward axis: how far the line across the
        # face tilts from level, level meaning square to the camera's
        # vertical. Measured against the camera's axes instead, a head that
        # is only turned and tipped would read as rolled.
        down = np.array([0.0, 1.0, 0.0])
        right_ref = np.cross(fwd, down)
        n = np.linalg.norm(right_ref)
        if n < 1e-9:
            return yaw, pitch, 0.0
        right_ref /= n
        down_ref = np.cross(right_ref, fwd)
        roll = math.degrees(math.atan2(right @ down_ref, right @ right_ref))
        return yaw, pitch, roll


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform minimising |dst - (s R src + t)|^2.

    Returns (s, R, t), R a proper rotation. Umeyama (1991).
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    a = src - mu_s
    b = dst - mu_d
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var = (a**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if var > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def lift(landmarks_px: np.ndarray, depth_m: np.ndarray, k: Intrinsics) -> np.ndarray:
    """Pixel positions and depths to camera-frame points, metres."""
    z = depth_m
    x = (landmarks_px[:, 0] - k.cx) * z / k.fx
    y = (landmarks_px[:, 1] - k.cy) * z / k.fy
    return np.stack([x, y, z], axis=1)


def _robust_fit(model: np.ndarray, pts: np.ndarray, tol: float):
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(3):
        if keep.sum() < MIN_POINTS:
            return None
        s, R, t = umeyama(model[keep], pts[keep])
        res = np.linalg.norm(pts - (s * model @ R.T + t), axis=1)
        new = res <= max(tol, 2.5 * float(np.median(res[keep])))
        if np.array_equal(new, keep):
            break
        keep = new
    if keep.sum() < MIN_POINTS:
        return None
    return s, R, t, int(keep.sum()), float(np.median(res[keep]))


def estimate(
    landmarks: np.ndarray,
    w: int,
    h: int,
    landmark_depth: np.ndarray | None = None,
    intrinsics: Intrinsics | None = None,
) -> HeadPose | None:
    """Head pose from FaceMesh landmarks, using depth where it can.

    ``landmarks`` is the ``(N, 3)`` normalised array FaceMesh produces (x
    and y as fractions of the frame, z its relative depth in units of the
    frame width). ``landmark_depth`` is the depth image's reading at each
    landmark, metres, 0 where it had none.
    """
    lm = np.asarray(landmarks, dtype=float)
    if len(lm) < 468:
        return None
    px = lm[RIGID, :2] * np.array([w, h])
    if landmark_depth is not None and intrinsics is not None:
        z = np.asarray(landmark_depth, dtype=float)[RIGID]
        ok = np.isfinite(z) & (z > 0.1)
        if ok.sum() >= MIN_POINTS:
            pts = lift(px[ok], z[ok], intrinsics)
            fit = _robust_fit(MODEL[ok], pts, OUTLIER_M)
            if fit is not None:
                s, R, _, n, res = fit
                return HeadPose(R, s, "depth", n, res)
    # FaceMesh's own depth: x and y in pixels, z in the same units (it is
    # scaled like x). Weak perspective, but always there.
    pts = np.column_stack([px, lm[RIGID, 2] * w])
    fit = _robust_fit(MODEL, pts, OUTLIER_M * 1e9)
    if fit is None:
        return None
    s, R, _, n, _ = fit
    return HeadPose(R, s, "landmarks", n, 0.0)


def eye_direction(dx: float, dy: float, eye_width_m: float) -> np.ndarray:
    """The eye's visual axis in the face frame, from the iris offset.

    ``dx`` and ``dy`` are :func:`gaze_monitor.gaze.eye_displacement`'s
    offsets: dx in half eye-widths along the eye's corner-to-corner axis
    (towards the subject's left, image right), dy in quarter eye-widths
    across it (down). The iris sits on a circle of radius
    ``IRIS_PLANE_RADIUS`` about the eye's centre of rotation, so the offset
    over that radius is the sine of the rotation.
    """
    sy = np.clip(dx * (eye_width_m / 2.0) / IRIS_PLANE_RADIUS, -0.95, 0.95)
    sp = np.clip(dy * (eye_width_m / 4.0) / IRIS_PLANE_RADIUS, -0.95, 0.95)
    yaw = math.asin(float(sy))
    pitch = math.asin(float(sp))  # down positive
    cp = math.cos(pitch)
    return np.array([math.sin(yaw) * cp, -math.sin(pitch), math.cos(yaw) * cp])


def gaze_angles_deg(pose: HeadPose, dx: float, dy: float) -> tuple[float, float]:
    """Line of sight in the camera frame: (yaw, pitch) degrees.

    Zero is straight into the camera; yaw positive towards image right,
    pitch positive downwards. The eye's direction in the face is turned by
    the head's rotation, so head yaw, pitch and roll are all accounted for.
    """
    g = pose.rotation @ eye_direction(dx, dy, pose.eye_width_m)
    yaw = math.degrees(math.atan2(g[0], -g[2]))
    pitch = math.degrees(math.atan2(g[1], math.hypot(g[0], g[2])))
    return yaw, pitch


def sample_landmark_depth(
    depth_m: np.ndarray | None, landmarks: np.ndarray, w: int, h: int
) -> np.ndarray | None:
    """The depth image's reading at each landmark: median of a 3x3 patch.

    0 where the patch has no valid reading. Vectorised over landmarks; this
    runs every frame.
    """
    if depth_m is None:
        return None
    lm = np.asarray(landmarks, dtype=float)
    hh, ww = depth_m.shape[:2]
    x = np.clip(np.round(lm[:, 0] * w).astype(int), 1, ww - 2)
    y = np.clip(np.round(lm[:, 1] * h).astype(int), 1, hh - 2)
    patch = np.stack(
        [depth_m[y + j, x + i] for j in (-1, 0, 1) for i in (-1, 0, 1)], axis=1
    ).astype(float)
    patch[~np.isfinite(patch) | (patch <= 0)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN patches
        out = np.nanmedian(patch, axis=1)
    out[~np.isfinite(out)] = 0.0
    return out
