"""The operator as a whole: head, body and arms, posed from a behaviour state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import materials as M
from .behaviour import HEAD_PIVOT, Behaviour, Segment, State, Traits
from .face import Appearance, FaceModel
from .mathutil import normalize
from .meshes import tube, uv_sphere
from .raster import Mesh
from .scene import Scene

UPPER_ARM = 0.29
FOREARM = 0.27


@dataclass
class PosedOperator:
    head_meshes: list  # world frame
    body_meshes: list  # world frame
    landmarks_world: np.ndarray  # (478, 3)
    eye_axes_world: np.ndarray  # (2, 3)
    state: State


class Operator:
    """Builds the operator's meshes for each behaviour state."""

    def __init__(self, appearance: Appearance | None = None):
        self.appearance = appearance or Appearance()
        self.face = FaceModel(self.appearance)
        self.eye_centres_head = np.stack([e.centre for e in self.face.eyes])

    def behaviour(
        self,
        scene: Scene,
        segments: list[Segment],
        traits: Traits | None = None,
        seed: int = 0,
    ) -> Behaviour:
        return Behaviour(
            scene,
            segments,
            self.eye_centres_head,
            stature=self.appearance.stature,
            traits=traits,
            seed=seed,
        )

    def pose(self, state: State) -> PosedOperator:
        head = self.face.pose(
            state.eye_dirs_head, blink=state.blink, pupil_radius=state.pupil_radius
        )
        R = state.head_R
        origin = state.head_pivot - R @ HEAD_PIVOT
        head_meshes = [m.transformed(R, origin) for m in head.meshes]
        # The eye shader wants each eye's visual axis; attach the world axis
        # and let the renderer rotate it into the camera frame.
        axes_world = head.eye_axes @ R.T
        k = 0
        for m in head_meshes:
            if np.all(np.atleast_1d(m.material) == M.EYE):
                m.attrs = np.concatenate(
                    [m.attrs[:, :7], np.tile(axes_world[k], (len(m.vertices), 1))],
                    axis=1,
                )
                k += 1
        landmarks = head.landmarks @ R.T + origin
        return PosedOperator(
            head_meshes=head_meshes,
            body_meshes=self._body(state),
            landmarks_world=landmarks,
            eye_axes_world=axes_world,
            state=state,
        )

    def _body(self, s: State) -> list[Mesh]:
        """Torso, apron, arms, legs and boots, posed for one instant."""
        from .mathutil import rot_y, rotation_between

        a = self.appearance
        st = a.stature / 1.75
        T = s.torso_R
        up = T @ _x90()
        meshes = []
        add = meshes.append

        # Torso: abdomen and chest, in the white coat.
        spine = s.neck_base - s.hips
        add(
            _place(
                uv_sphere(
                    radii=(0.165 * st, 0.115 * st, 0.20 * st),
                    rings=10,
                    segments=18,
                    material=M.COAT,
                ),
                up,
                s.hips + spine * 0.38,
            )
        )
        add(
            _place(
                uv_sphere(
                    radii=(0.18 * st, 0.12 * st, 0.21 * st),
                    rings=10,
                    segments=20,
                    material=M.COAT,
                ),
                up,
                s.hips + spine * 0.70,
            )
        )
        add(
            _place(
                uv_sphere(
                    radii=(0.205 * st, 0.11 * st, 0.075 * st),
                    rings=8,
                    segments=18,
                    material=M.COAT,
                ),
                up,
                s.neck_base - T @ np.array([0.0, 0.03, 0.0]),
            )
        )
        # Pelvis, in work trousers.
        pelvis = s.hips + np.array([0.0, -0.04, 0.0])
        add(
            _place(
                uv_sphere(
                    radii=(0.16 * st, 0.12 * st, 0.11 * st),
                    rings=8,
                    segments=16,
                    material=M.TROUSERS,
                ),
                up,
                pelvis,
            )
        )
        # Long apron from the chest to below the knee, over the front.
        front = T @ np.array([0.0, 0.0, -1.0])
        apron_c = (
            s.hips + spine * 0.22 + front * 0.045 * st + np.array([0.0, -0.10, 0.0])
        )
        add(
            _place(
                uv_sphere(
                    radii=(0.185 * st, 0.115 * st, 0.47 * st),
                    rings=12,
                    segments=18,
                    material=M.APRON,
                ),
                up,
                apron_c,
            )
        )
        add(
            tube(
                s.neck_base,
                s.head_pivot + np.array([0.0, 0.03, 0.0]),
                0.052,
                0.048,
                segments=14,
                rings=2,
                material=M.SKIN_PLAIN,
            )
        )

        # Legs: hip to knee to ankle, feet planted where the operator stands.
        yaw = rot_y(0.3 * math.atan2(T[0, 2], T[2, 2]))
        foot_fwd = yaw @ np.array([0.0, 0.0, -1.0])
        for side in (-1.0, 1.0):
            hip = pelvis + T @ np.array([side * 0.09 * st, -0.03, 0.0])
            ankle = np.array([s.stance[0] + side * 0.12 * st, 0.09, s.stance[1] + 0.02])
            knee = _elbow(
                hip,
                ankle,
                0.43 * st,
                0.43 * st,
                foot_fwd + np.array([side * 0.15, 0.0, 0.0]),
            )
            add(
                tube(
                    hip,
                    knee,
                    0.075 * st,
                    0.058 * st,
                    segments=12,
                    rings=2,
                    material=M.TROUSERS,
                )
            )
            shin = normalize(knee - ankle)
            add(
                tube(
                    knee,
                    ankle + shin * 0.28,
                    0.056 * st,
                    0.05 * st,
                    segments=12,
                    rings=2,
                    material=M.TROUSERS,
                )
            )
            # White rubber boots: a shaft to mid-calf and the foot.
            add(
                tube(
                    ankle - np.array([0.0, 0.03, 0.0]),
                    ankle + shin * 0.30,
                    0.062 * st,
                    0.064 * st,
                    segments=14,
                    rings=2,
                    material=M.BOOT,
                )
            )
            foot = uv_sphere(
                radii=(0.052, 0.048, 0.135), rings=6, segments=12, material=M.BOOT
            )
            add(
                _place(
                    foot,
                    rotation_between(np.array([0, 0, 1.0]), foot_fwd),
                    np.array([ankle[0], 0.05, ankle[2]]) + foot_fwd * 0.06,
                )
            )

        # Arms, in coat sleeves, with work gloves.
        for side, hand in zip((-1.0, 1.0), s.hands, strict=True):
            shoulder = s.neck_base + T @ np.array([side * 0.19 * st, -0.05, 0.0])
            elbow = _elbow(
                shoulder,
                hand,
                UPPER_ARM * st,
                FOREARM * st,
                np.array([side * 0.6, -1.0, 0.25]),
            )
            add(
                tube(
                    shoulder, elbow, 0.05, 0.044, segments=12, rings=2, material=M.COAT
                )
            )
            add(tube(elbow, hand, 0.042, 0.036, segments=12, rings=2, material=M.COAT))
            fwd = normalize(hand - elbow)
            add(
                tube(
                    hand - fwd * 0.03,
                    hand + fwd * 0.02,
                    0.036,
                    0.034,
                    segments=10,
                    rings=1,
                    material=M.GLOVE,
                )
            )
            palm = uv_sphere(
                radii=(0.045, 0.024, 0.085), rings=6, segments=10, material=M.GLOVE
            )
            add(
                _place(
                    palm,
                    rotation_between(np.array([0, 0, 1.0]), fwd),
                    hand + fwd * 0.075,
                )
            )
        return meshes


def _x90() -> np.ndarray:
    """Turn a +z-pole sphere so its pole points up: the torso's long axis."""
    from .mathutil import rot_x

    return rot_x(-math.pi / 2)


def _place(mesh: Mesh, R: np.ndarray, t: np.ndarray) -> Mesh:
    return mesh.transformed(R, t)


def _elbow(shoulder, wrist, a, b, pole) -> np.ndarray:
    """Two-link inverse kinematics: the elbow, bent towards ``pole``."""
    d = wrist - shoulder
    dist = float(np.linalg.norm(d))
    dist = min(max(dist, abs(a - b) + 1e-3), a + b - 1e-3)
    u = normalize(d)
    x = (a * a - b * b + dist * dist) / (2 * dist)
    h = math.sqrt(max(a * a - x * x, 0.0))
    p = normalize(pole - (pole @ u) * u)
    return shoulder + u * x + p * h
