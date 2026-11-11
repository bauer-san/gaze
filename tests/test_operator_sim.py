"""The simulated saw operator (tools/operator_sim).

numpy only, like the rest of the suite: the renderer, the motion model and
the replay of the real pipeline on ground-truth landmarks all run without
OpenCV or MediaPipe. Rendering is done on a small frame to keep this quick.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools.operator_sim import face as F
from tools.operator_sim.behaviour import Segment, build_segments
from tools.operator_sim.evaluate import Params, score, simulate_truth
from tools.operator_sim.mathutil import saccade_duration
from tools.operator_sim.meshes import box, tube, uv_sphere
from tools.operator_sim.raster import rasterize
from tools.operator_sim.scene import BLADE_ZONE, CameraMount, Scene
from tools.operator_sim.sim import Simulation


def small_scene() -> Scene:
    # Same field of view as the D435i at 1280x720, a quarter of the pixels.
    return Scene(mount=CameraMount(width=320, height=180))


# -- geometry ----------------------------------------------------------------


def test_primitives_wind_outwards():
    for mesh in (
        uv_sphere(),
        box((0, 0, 0), (1, 2, 3)),
        tube((0, 0, 0), (0, 1, 0), 0.2),
    ):
        v, f = mesh.vertices, mesh.faces
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        out = v[f].mean(axis=1) - v.mean(axis=0)
        big = np.linalg.norm(n, axis=1) > 1e-12
        assert np.mean(np.einsum("ij,ij->i", n[big], out[big]) > 0) > 0.95


def test_rasterizer_depth_is_perspective_correct():
    # A square at a slant: depth at each pixel must match the ray-plane hit.
    verts = np.array(
        [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [1.0, 1.0, 4.0], [-1.0, 1.0, 4.0]]
    )
    faces = np.array([[0, 2, 1], [0, 3, 2]])
    normals = np.tile([0.0, 0.0, -1.0], (4, 1))
    g = rasterize(
        verts, normals, faces, np.zeros(2, np.int16), np.zeros((4, 1)),
        100.0, 100.0, 50.0, 50.0, 100, 100, cull_backfaces=False,
    )  # fmt: skip
    yy, xx = np.divmod(g.pixel, 100)
    ray_y = (yy + 0.5 - 50.0) / 100.0
    # Plane: z = 3 + y, so along a ray z = 3 + ray_y * z.
    expected = 3.0 / (1.0 - ray_y)
    assert len(g.pixel) > 500
    np.testing.assert_allclose(g.depth, expected, rtol=1e-6)


def test_zone_projection_round_trips():
    eye = np.array([0.02, 1.6, 0.55])
    for u, v in ((0.0, 0.0), (1.0, 0.0), (0.3, 0.7), (1.2, -0.1)):
        p = BLADE_ZONE.point(u, v)
        got = BLADE_ZONE.project_ray(eye, p - eye)
        assert got == pytest.approx((u, v), abs=1e-9)


# -- the face ------------------------------------------------------------------


@pytest.fixture(scope="module")
def face_model():
    return F.FaceModel()


def test_landmarks_follow_mediapipe_topology(face_model):
    head = face_model.pose(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
    lm = head.landmarks
    assert lm.shape == (478, 3)
    # MediaPipe's "left" eye (33/133, iris 468) is the subject's right: -x.
    assert lm[33, 0] < 0 and lm[263, 0] > 0
    assert lm[468, 0] < 0 and lm[473, 0] > 0
    # Interpupillary distance of an adult, and the iris ring is 11.7 mm across.
    ipd = np.linalg.norm(lm[468] - lm[473])
    assert 0.055 < ipd < 0.07
    ring = np.linalg.norm(lm[469] - lm[471])
    assert ring == pytest.approx(2 * F.IRIS_RADIUS, abs=1e-6)


def test_eyes_rotate_independently_of_the_face(face_model):
    ahead = face_model.pose(np.array([[0.0, 0.0, 1.0]] * 2))
    yaw = math.radians(20.0)
    turned = face_model.pose(np.array([[math.sin(yaw), 0.0, math.cos(yaw)]] * 2))
    corners = [33, 133, 362, 263]
    np.testing.assert_allclose(ahead.landmarks[corners], turned.landmarks[corners])
    # The iris moves by R sin(yaw), towards +x.
    shift = turned.landmarks[468, 0] - ahead.landmarks[468, 0]
    depth = math.sqrt(F.EYEBALL_RADIUS**2 - F.IRIS_RADIUS**2)
    assert shift == pytest.approx(depth * math.sin(yaw), rel=1e-6)


def test_blink_closes_the_lids(face_model):
    gap = []
    for blink in (0.0, 1.0):
        lm = face_model.pose(np.array([[0.0, 0.0, 1.0]] * 2), blink=blink).landmarks
        gap.append(lm[159, 1] - lm[145, 1])  # upper and lower lid margin, mid-eye
    assert gap[0] > 0.006
    assert abs(gap[1]) < 0.0015


def test_upper_lid_follows_downward_gaze(face_model):
    level = face_model.pose(np.array([[0.0, 0.0, 1.0]] * 2)).landmarks[159, 1]
    down = math.radians(25.0)
    lowered = face_model.pose(
        np.array([[0.0, -math.sin(down), math.cos(down)]] * 2)
    ).landmarks[159, 1]
    assert lowered < level - 0.002


# -- motion --------------------------------------------------------------------


def test_main_sequence():
    assert saccade_duration(10) == pytest.approx(0.043)
    assert saccade_duration(40) > 2 * saccade_duration(5)


def _states(segments, seconds, fps=30.0, seed=0):
    sim = Simulation(segments, fps=fps, seed=seed, scene=small_scene(), render=False)
    return [sim.next_state()[1] for _ in range(int(seconds * fps))]


def test_vor_keeps_gaze_on_target_while_the_head_turns():
    states = _states(build_segments("vor", 10.0, calibrate=False), 10.0)[30:]
    head_yaw = np.degrees([s.head_angles[0] for s in states])
    eye_yaw = np.degrees([s.eye_angles[:, 0].mean() for s in states])
    assert np.ptp(head_yaw) > 20.0
    # Eyes counter-rotate in the head, one for one.
    assert np.corrcoef(head_yaw, eye_yaw)[0, 1] < -0.95
    cut = Scene().targets()["cut"]
    for s in states:
        aim = (s.fixation - s.cyclopean) / np.linalg.norm(s.fixation - s.cyclopean)
        want = (cut - s.cyclopean) / np.linalg.norm(cut - s.cyclopean)
        assert math.degrees(math.acos(min(1.0, float(aim @ want)))) < 1.5


def test_eyes_only_holds_the_head():
    segs = [Segment("cut", 1.0, head="still"), Segment("tray", 1.0, head="still")]
    states = _states(segs, 2.0)
    head = np.degrees([s.head_angles[:2] for s in states])
    eye = np.degrees([s.eye_angles[0] for s in states])
    assert np.ptp(head[:, 0]) < 5.0
    assert np.ptp(eye[:, 0]) > 25.0


def test_large_gaze_shifts_recruit_the_head():
    segs = [Segment("cut", 1.0), Segment("coworker_left", 2.0)]
    states = _states(segs, 3.0)
    assert math.degrees(states[-1].head_angles[0]) > 40.0
    # ...and the head stops short, leaving the eyes off-centre in the head.
    assert 0.0 < math.degrees(states[-1].eye_angles[:, 0].mean()) < 20.0


def test_vergence_converges_on_near_targets():
    s = _states([Segment("cut", 1.0, head="still")], 1.0)[-1]
    # The left eye (subject's right, -x) turns towards +x and the right eye
    # towards -x in the head frame when converging.
    assert s.eye_angles[0, 0] > s.eye_angles[1, 0]


def test_blinks_happen_at_a_human_rate():
    states = _states([Segment("cut", 60.0, scan=0.02)], 60.0)
    closed = np.array([s.blink > 0.9 for s in states])
    onsets = np.sum(closed[1:] & ~closed[:-1])
    assert 6 <= onsets <= 35


# -- rendering -------------------------------------------------------------------


def test_render_produces_a_realsense_frame():
    sim = Simulation(
        "fixation", duration=1.0, fps=10, seed=1, scene=small_scene(), ssaa=1
    )
    frame = sim.next_frame()
    r = frame.render
    assert r.color.shape == (180, 320, 3) and r.color.dtype == np.uint8
    assert r.depth_m.shape == (180, 320) and r.depth_m.dtype == np.float32
    # Depth on the face is the depth of the surface there, to a few mm; the
    # eyes' centres are a few centimetres further back.
    x, y = np.round(r.landmarks_px[168]).astype(int)  # nose bridge
    patch = r.depth_m[y - 1 : y + 2, x - 1 : x + 2]
    assert np.median(patch[patch > 0]) == pytest.approx(r.landmarks_z[168], abs=0.006)
    assert 0.0 < frame.truth["distance_m"] - r.landmarks_z[168] < 0.05
    # The face is in frame and the eyes are open and showing.
    assert np.all((r.landmarks_px >= 0) & (r.landmarks_px < [320, 180]))
    assert all(frame.truth["iris_visible"])
    # The face is lit and the iris is darker than the sclera beside it.
    iris = (
        r.color[int(r.landmarks_px[468, 1]), int(r.landmarks_px[468, 0])]
        .astype(int)
        .sum()
    )
    assert iris < 500


# -- the real pipeline, on perfect landmarks ---------------------------------------


def test_pipeline_on_perfect_landmarks_tracks_the_true_zone():
    """With a perfect detector and a still-headed calibration, the pipeline
    should get the zone right almost all the time: the truth, the geometry
    and the replay all agree with each other."""
    run = simulate_truth(
        "eyes-only", duration=15.0, fps=30.0, seed=2, calibration_head="still"
    )
    rep = score(run, Params(), landmarks="truth", ideal=False)
    assert rep["calibration_ok"]
    assert rep["eye_in_head"]["yaw_resid_rms_deg"] < 1.5
    assert rep["eye_in_head"]["pitch_resid_rms_deg"] < 1.5
    assert rep["gaze_error"]["median_deg"] < 4.0
    assert rep["in_zone"]["accuracy"] > 0.85


def test_vor_is_where_eye_in_head_stops_being_gaze():
    """Gaze never leaves the zone in the VOR scenario, yet an eye-in-head
    measurement calls it out of zone a good part of the time, even with
    perfect landmarks. This is a property of the method, pinned here so it
    is noticed if the pipeline ever starts compensating for head pose."""
    run = simulate_truth(
        "vor", duration=15.0, fps=30.0, seed=3, calibration_head="still"
    )
    rep = score(run, Params(), landmarks="truth", ideal=False)
    assert rep["calibration_ok"]
    assert rep["in_zone"]["false_out"] > 0.2


# -- camera placement -------------------------------------------------------------


def test_nothing_hangs_into_the_product_envelope():
    from tools.operator_sim.placement import mountable
    from tools.operator_sim.scene import HEAD_BOTTOM_Y

    # Over the table, below the housing: product passes there.
    assert not mountable((-0.05, HEAD_BOTTOM_Y - 0.02, 0.10))
    # The same spot raised clear of the underside is fine.
    assert mountable((-0.05, HEAD_BOTTOM_Y + 0.025, 0.10))
    # Inside the housing is not a mount, nor is anywhere far from it.
    assert not mountable((0.2, 1.6, 0.0))
    assert not mountable((-0.05, 1.4, 0.40))


def test_placement_prefers_a_camera_in_line_with_the_operator():
    from tools.operator_sim.placement import study

    centred, offset = study(
        [(-0.03, 1.365, 0.13), (0.20, 1.365, 0.13)], "fixation", duration=4.0
    )
    assert centred["face_deg"] < offset["face_deg"] - 5
    assert centred["blocked"] == 0.0
    # The camera is above the line of sight to the cut, looking down at the lid.
    assert centred["camera_below_deg"] < 0


def test_calibration_tolerates_a_mount_that_creeps():
    """Turning the camera moves the iris and the eye corners together, so a
    ball head that creeps after commissioning does not move the zone; only
    a change of viewpoint does, and slightly."""
    from tools.operator_sim.placement import drift_study

    drifts = {
        "none": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        "tips": ((0.0, 0.0, 0.0), (2.0, 3.0, 5.0)),
        "sags": ((0.0, -0.015, 0.0), (0.0, 0.0, 0.0)),
    }
    rows = {
        r["drift"]: r for r in drift_study(drifts, scenario="eyes-only", duration=8.0)
    }
    base = rows["none"]["gaze_median_deg"]
    assert abs(rows["tips"]["gaze_median_deg"] - base) < 0.3
    assert abs(rows["sags"]["gaze_median_deg"] - base) < 0.8
