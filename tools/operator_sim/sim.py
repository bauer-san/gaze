"""The simulation: scenario in, rendered frames and ground truth out."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .behaviour import Segment, Traits, build_segments
from .face import PRESETS, Appearance
from .mathutil import normalize, yaw_pitch
from .operator import Operator
from .render import Renderer, RenderResult, SensorModel
from .scene import OPERATOR_FACE, Scene, angular_size_deg


@dataclass
class SimFrame:
    index: int
    t: float
    render: RenderResult
    truth: dict


class Simulation:
    """One operator, one scenario, at a fixed frame rate.

    ``truth`` for every frame records what the operator was really doing, in
    the units the pipeline reports and in physical ones, so the two can be
    compared directly:

    * ``zone_u``, ``zone_v``: where the line of sight meets the attention
      zone's plane, 0..1 inside it, the same convention as
      ``AttentionZone.project``.
    * ``eye_yaw_deg``, ``eye_pitch_deg``: each eye's rotation in the head,
      which is what an iris-in-eye measurement actually observes.
    * ``gaze_yaw_deg``, ``gaze_pitch_deg``: the line of sight in the world.
    * ``head_*_deg``, ``distance_m`` (camera to the midpoint of the eyes),
      ``blink``, ``iris_visible``, ``landmarks_px`` and more.
    """

    def __init__(
        self,
        scenario: str | list[Segment] = "shift",
        duration: float = 60.0,
        fps: float = 30.0,
        seed: int = 0,
        appearance: Appearance | str = "default",
        traits: Traits | None = None,
        scene: Scene | None = None,
        camera: tuple | None = None,
        aim: tuple | None = None,
        sensor: SensorModel | None = None,
        ssaa: int = 2,
        calibrate: bool = True,
        calibration_head: str = "natural",
        render: bool = True,
    ):
        if isinstance(appearance, str):
            appearance = PRESETS[appearance]
        self.scene = scene or Scene()
        if camera is not None or aim is not None:
            # A camera position, in the world frame, aimed at the operator's
            # eyes unless told otherwise.
            mount = dataclasses.replace(
                self.scene.mount,
                position=(
                    tuple(camera) if camera is not None else self.scene.mount.position
                ),
                target=tuple(aim) if aim is not None else OPERATOR_FACE,
            )
            self.scene = dataclasses.replace(self.scene, mount=mount)
        self.fps = fps
        self.operator = Operator(appearance)
        if isinstance(scenario, str):
            segments = build_segments(
                scenario,
                duration,
                seed=seed,
                calibrate=calibrate,
                calibration_head=calibration_head,
            )
            self.scenario = scenario
        else:
            segments = list(scenario)
            self.scenario = "custom"
        self.segments = segments
        self.duration = sum(s.duration for s in segments)
        self.behaviour = self.operator.behaviour(
            self.scene, segments, traits=traits, seed=seed
        )
        self.renderer = Renderer(
            self.scene, sensor or SensorModel(seed=seed), ssaa=ssaa, static=render
        )
        self.frame_count = int(math.floor(self.duration * fps))
        self.seed = seed
        self._index = 0

    @property
    def intrinsics(self) -> dict:
        r = self.renderer
        return {
            "fx": r.fx,
            "fy": r.fy,
            "cx": r.cx,
            "cy": r.cy,
            "width": r.width,
            "height": r.height,
        }

    def __iter__(self) -> Iterator[SimFrame]:
        while self._index < self.frame_count:
            yield self.next_frame()

    def next_state(self):
        """Advance the behaviour one frame. Cheap; must be called in order."""
        i = self._index
        self._index += 1
        return i, self.behaviour.step_to(i / self.fps)

    def render_state(self, index: int, state) -> SimFrame:
        """Render one behaviour state. Independent of every other frame.

        The sensor noise is seeded from the frame index, so a frame renders
        identically whichever process renders it and in whatever order --
        which is what lets generation run on a pool of workers.
        """
        self.renderer.sensor.rng = np.random.default_rng((self.seed, index))
        posed = self.operator.pose(state)
        result = self.renderer.render(
            posed,
            self.operator.appearance,
            self.operator.face.texture,
            props=self.props(state),
        )
        return SimFrame(
            index=index,
            t=state.t,
            render=result,
            truth=self._truth(state, posed, result),
        )

    def truth_state(self, index: int, state) -> SimFrame:
        """Ground truth for one state without rasterising anything.

        Poses the head and projects its landmarks, which is all a perfect
        landmark detector would see. Orders of magnitude cheaper than
        :meth:`render_state`, for experiments on the motion and the gaze
        geometry that do not need FaceMesh in the loop. ``render.color`` and
        ``render.depth_m`` are ``None``; an iris counts as visible when the
        lids are less than half closed.
        """
        head = self.operator.face.pose(
            state.eye_dirs_head, blink=state.blink, pupil_radius=state.pupil_radius
        )
        from .behaviour import HEAD_PIVOT

        R = state.head_R
        origin = state.head_pivot - R @ HEAD_PIVOT
        landmarks = head.landmarks @ R.T + origin
        r = self.renderer
        result = RenderResult(
            color=None,
            depth_m=None,
            landmarks_px=r.project(landmarks),
            landmarks_z=r.to_camera(landmarks)[:, 2],
            iris_visible=np.array([state.blink < 0.5] * 2),
        )
        return SimFrame(
            index=index,
            t=state.t,
            render=result,
            truth=self._truth(state, None, result),
        )

    def props(self, state) -> list:
        """Moving scenery for a state: the carriage at its travel."""
        from .scene import carriage_meshes

        return carriage_meshes(state.carriage) if self.renderer.dynamic_carriage else []

    def next_frame(self) -> SimFrame:
        return self.render_state(*self.next_state())

    def _truth(self, state, posed, result: RenderResult) -> dict:
        zone = self.scene.zone
        cyc = state.cyclopean
        gaze_dir = normalize(state.fixation - cyc)
        u, v = zone.project_ray(cyc, gaze_dir)
        gy, gp = yaw_pitch(np.array([-gaze_dir[0], gaze_dir[1], -gaze_dir[2]]))
        cam = self.renderer.to_camera(cyc)
        zw, zh = angular_size_deg(zone, cyc)
        # Facing the camera: angle between the face's forward axis and the
        # direction to the camera.
        fwd = state.head_R @ np.array([0.0, 0.0, 1.0])
        to_cam = normalize(self.renderer.C - cyc)
        seg = state.segment
        return {
            "t": round(state.t, 4),
            "activity": seg.activity,
            "target": (
                seg.target
                if isinstance(seg.target, str)
                else list(map(float, seg.target))
            ),
            "calib_corner": seg.calib_corner,
            "saccade": bool(state.saccade),
            "zone_u": u,
            "zone_v": v,
            "zone_size_deg": [zw, zh],
            "gaze_yaw_deg": math.degrees(gy),
            "gaze_pitch_deg": math.degrees(gp),
            "eye_yaw_deg": np.degrees(state.eye_angles[:, 0]).tolist(),
            "eye_pitch_deg": np.degrees(state.eye_angles[:, 1]).tolist(),
            "head_yaw_deg": math.degrees(state.head_angles[0]),
            "head_pitch_deg": math.degrees(state.head_angles[1]),
            "head_roll_deg": math.degrees(state.head_angles[2]),
            "face_to_camera_deg": math.degrees(
                math.acos(float(np.clip(fwd @ to_cam, -1, 1)))
            ),
            "distance_m": float(cam[2]),
            "eye_centres": state.eye_centres_world.tolist(),
            "fixation": state.fixation.tolist(),
            "blink": float(state.blink),
            "iris_visible": result.iris_visible.tolist(),
            "landmarks_px": result.landmarks_px,
        }
