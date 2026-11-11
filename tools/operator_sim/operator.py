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
        a = self.appearance
        st = a.stature / 1.75
        T = s.torso_R
        meshes = []
        chest_c = s.hips + T @ np.array([0.0, 0.30 * a.stature * 0.58, 0.0])
        torso = uv_sphere(
            radii=(0.175 * st, 0.115 * st, 0.30 * st),
            rings=12,
            segments=20,
            material=M.COAT,
        )
        meshes.append(_place(torso, T @ _x90(), chest_c))
        shoulders = uv_sphere(
            radii=(0.205 * st, 0.11 * st, 0.075 * st),
            rings=8,
            segments=18,
            material=M.COAT,
        )
        meshes.append(
            _place(shoulders, T @ _x90(), s.neck_base - T @ np.array([0.0, 0.03, 0.0]))
        )
        # Apron over the front of the torso. The operator faces -z, so the
        # front of the body is along the torso's -z.
        apron = uv_sphere(
            radii=(0.17 * st, 0.03, 0.27 * st), rings=10, segments=16, material=M.APRON
        )
        meshes.append(
            _place(apron, T @ _x90(), chest_c + T @ np.array([0.0, -0.06, -0.10 * st]))
        )
        meshes.append(
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
        for side, hand in zip((-1.0, 1.0), s.hands, strict=True):
            shoulder = s.neck_base + T @ np.array([side * 0.19 * st, -0.05, 0.0])
            elbow = _elbow(
                shoulder,
                hand,
                UPPER_ARM * st,
                FOREARM * st,
                np.array([side * 0.6, -1.0, 0.25]),
            )
            meshes.append(
                tube(
                    shoulder, elbow, 0.048, 0.042, segments=12, rings=2, material=M.COAT
                )
            )
            meshes.append(
                tube(elbow, hand, 0.040, 0.032, segments=12, rings=2, material=M.COAT)
            )
            fwd = normalize(hand - elbow)
            palm = uv_sphere(
                radii=(0.045, 0.022, 0.08), rings=6, segments=10, material=M.APRON
            )
            from .mathutil import rotation_between

            meshes.append(
                _place(
                    palm,
                    rotation_between(np.array([0, 0, 1.0]), fwd),
                    hand + fwd * 0.07,
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
