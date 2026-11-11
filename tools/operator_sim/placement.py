"""Where to put the camera: score candidate mounts on how they see the operator.

Gaze is measured from the iris inside the eye opening, so what a mount has
to deliver is a good view *of the eyes*, for as much of the shift as
possible:

``face_deg``
    Angle between the face's forward direction and the direction to the
    camera. Head-on is 0. FaceMesh's landmarks degrade as the face turns
    away, and past about 40-50 degrees the far eye is lost behind the nose.
``iris_deg``
    Angle between the eyes' visual axes and the direction to the camera:
    how obliquely the iris is seen. The operator looks *down* at the blade
    with the upper lid following, so a camera below the line of sight looks
    up under the lid, while one above it looks down onto the lid.
``camera_below_deg``
    Positive when the camera is below the line of sight, which is the side
    the upper lid does not cover.
``iris_px``
    The iris's diameter in the image, from distance and focal length.
``blocked``
    Fraction of frames in which the line of sight from the camera to either
    eye is blocked by the saw, the product, or the operator's own arms.
``usable``
    Fraction of *all* monitored frames with the face within 35 degrees of
    head-on, both eyes unobstructed, and both in the field of view.

Each candidate is aimed at the operator's median eye position while they
watch the cut, as a bracket would be set up at commissioning. Everything is
computed from ground truth, without rendering, so a grid of dozens of mounts
takes seconds; confirm the shortlist with ``generate``/``evaluate``, which
put FaceMesh in the loop.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .mathutil import look_at, normalize
from .raster import Mesh
from .scene import (
    HEAD_BACK_Z,
    HEAD_BOTTOM_Y,
    HEAD_FRONT_Z,
    HEAD_LEFT_X,
    HEAD_RIGHT_X,
    CameraMount,
    Scene,
)

FACE_LIMIT_DEG = 35.0
# Product envelope in plan (carriage plus fixed table) and the clearance the
# camera body keeps above it. A D435i is 25 mm tall.
PRODUCT_X = (-0.58, 0.45)
PRODUCT_Z = (-0.36, 0.38)
CAMERA_HALF_HEIGHT = 0.0125
CLEARANCE = 0.01
INCH = 0.0254


@dataclasses.dataclass
class _Frames:
    eyes: np.ndarray  # (n, 2, 3)
    forward: np.ndarray  # (n, 3) face forward
    axes: np.ndarray  # (n, 2, 3) visual axes, world
    watching: np.ndarray  # (n,) bool: watching the cut
    body: list  # per frame: triangles (m, 3, 3) of the arms and torso


def collect(
    scenario: str = "shift", duration: float = 60.0, seed: int = 0, step: int = 3
):
    """Behaviour states, every ``step``-th frame, without the calibration."""
    from .sim import Simulation

    sim = Simulation(
        scenario, duration=duration, seed=seed, render=False, calibrate=False
    )
    eyes, fwd, axes, watching, body = [], [], [], [], []
    for _ in range(sim.frame_count):
        i, st = sim.next_state()
        if i % step:
            continue
        eyes.append(st.eye_centres_world)
        fwd.append(st.head_R @ np.array([0.0, 0.0, 1.0]))
        axes.append(st.eye_dirs_head @ st.head_R.T)
        watching.append(st.segment.activity in ("watch_cut", "watch_guide", "fixation"))
        body.append(_triangles(sim.operator._body(st)))
    return _Frames(
        eyes=np.asarray(eyes),
        forward=np.asarray(fwd),
        axes=np.asarray(axes),
        watching=np.asarray(watching),
        body=body,
    )


def _triangles(meshes: list[Mesh]) -> np.ndarray:
    return np.concatenate([m.vertices[m.faces] for m in meshes])


def _blocked(origin: np.ndarray, targets: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """For each segment origin->target, does it cross any triangle?"""
    if len(tris) == 0:
        return np.zeros(len(targets), dtype=bool)
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    d = targets - origin  # (r, 3), not normalised: t in (0, 1) is on the segment
    p = np.cross(d[:, None, :], e2[None])  # (r, t, 3)
    det = np.einsum("rtk,tk->rt", p, e1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - v0  # (t, 3)
    u = np.einsum("rtk,tk->rt", p, s) * inv
    q = np.cross(s, e1)  # (t, 3)
    v = np.einsum("rk,tk->rt", d, q) * inv
    t = np.einsum("tk,tk->t", e2, q)[None] * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0.01) & (t < 0.97)
    return hit.any(axis=1)


def _static_triangles(scene: Scene, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    tris = _triangles(scene.static_meshes())
    tmin = tris.min(axis=1)
    tmax = tris.max(axis=1)
    keep = np.all((tmax >= lo) & (tmin <= hi), axis=1)
    return tris[keep]


def score_mount(position, frames: _Frames, static_tris: np.ndarray, mount=None) -> dict:
    """Metrics for one camera position over the collected frames."""
    base = mount or CameraMount()
    pos = np.asarray(position, float)
    aim = np.median(frames.eyes[frames.watching].mean(axis=1), axis=0)
    R = look_at(pos, aim)
    fx = (base.width / 2) / math.tan(math.radians(base.hfov_deg) / 2)
    tan_h = math.tan(math.radians(base.hfov_deg) / 2)
    tan_v = math.tan(math.radians(base.vfov_deg) / 2)

    cyc = frames.eyes.mean(axis=1)
    to_cam = normalize(pos - cyc)
    face = np.degrees(
        np.arccos(np.clip(np.einsum("ij,ij->i", frames.forward, to_cam), -1, 1))
    )
    eye_to_cam = normalize(pos[None, None, :] - frames.eyes)
    iris = np.degrees(
        np.arccos(np.clip(np.einsum("nek,nek->ne", frames.axes, eye_to_cam), -1, 1))
    )
    gaze = normalize(frames.axes.mean(axis=1))
    # Camera below the line of sight: positive. Measured in the vertical
    # plane containing the gaze direction.
    side = normalize(np.cross(gaze, np.array([0.0, 1.0, 0.0])))
    up = np.cross(side, gaze)
    below = np.degrees(np.arcsin(np.clip(-np.einsum("ij,ij->i", to_cam, up), -1, 1)))
    dist = np.linalg.norm(pos - cyc, axis=1)
    cam = (frames.eyes - pos) @ R.T  # (n, 2, 3)
    in_view = np.all(
        (cam[..., 2] > 0)
        & (np.abs(cam[..., 0] / cam[..., 2]) < tan_h * 0.95)
        & (np.abs(cam[..., 1] / cam[..., 2]) < tan_v * 0.95),
        axis=1,
    )
    blocked = np.zeros(len(cyc), dtype=bool)
    for k in range(len(cyc)):
        tris = np.concatenate([static_tris, frames.body[k]])
        blocked[k] = _blocked(pos, frames.eyes[k], tris).any()
    usable = (face < FACE_LIMIT_DEG) & ~blocked & in_view
    w = frames.watching
    return {
        "x": round(float(pos[0]), 3),
        "y": round(float(pos[1]), 3),
        "z": round(float(pos[2]), 3),
        "height_in": round(float(pos[1]) / INCH, 1),
        "face_deg": round(float(np.median(face[w])), 1),
        "face_p90_deg": round(float(np.percentile(face[w], 90)), 1),
        "iris_deg": round(float(np.median(iris[w])), 1),
        "camera_below_deg": round(float(np.median(below[w])), 1),
        "dist_m": round(float(np.median(dist[w])), 3),
        "iris_px": round(float(np.median(fx * 0.0117 / dist[w])), 1),
        "blocked": round(float(blocked.mean()), 3),
        "blocked_watching": round(float(blocked[w].mean()), 3),
        "usable": round(float(usable.mean()), 3),
        "usable_watching": round(float(usable[w].mean()), 3),
    }


def mountable(p) -> bool:
    """Reachable with standoffs or a short bracket, and out of the way.

    Within 0.15 m of the upper housing, not inside it, and with the camera
    body clear of the product envelope. The underside of the housing *is*
    the saw's maximum cut height: product on the carriage, and the pieces
    coming off the blade, pass beneath it across the full width of the
    table, so nothing may hang below it over the table.
    """
    x, y, z = p
    over_table = PRODUCT_X[0] < x < PRODUCT_X[1] and PRODUCT_Z[0] < z < PRODUCT_Z[1]
    if over_table and y - CAMERA_HALF_HEIGHT < HEAD_BOTTOM_Y + CLEARANCE:
        return False
    inside = (
        HEAD_LEFT_X < x < HEAD_RIGHT_X
        and y > HEAD_BOTTOM_Y
        and HEAD_BACK_Z < z < HEAD_FRONT_Z
    )
    if inside:
        return False
    if -0.05 < x < 0.05 and y < HEAD_BOTTOM_Y and -0.06 < z < 0.06:
        return False  # the guide, the blade and their travel
    if y < 1.15:
        return False
    dx = max(HEAD_LEFT_X - x, 0.0, x - HEAD_RIGHT_X)
    dy = max(HEAD_BOTTOM_Y - y, 0.0)
    dz = max(HEAD_BACK_Z - z, 0.0, z - HEAD_FRONT_Z)
    return math.sqrt(dx * dx + dy * dy + dz * dz) <= 0.15


def grid(xs, ys, zs) -> list[tuple]:
    return [(x, y, z) for x in xs for y in ys for z in zs if mountable((x, y, z))]


def study(
    candidates: list[tuple],
    scenario: str = "shift",
    duration: float = 60.0,
    seed: int = 0,
    step: int = 3,
) -> list[dict]:
    frames = collect(scenario, duration, seed, step)
    scene = Scene()
    pts = np.asarray(candidates, float)
    allp = np.concatenate([pts, frames.eyes.reshape(-1, 3)])
    static = _static_triangles(scene, allp.min(axis=0) - 0.05, allp.max(axis=0) + 0.05)
    return [score_mount(c, frames, static) for c in candidates]
