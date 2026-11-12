"""Head pose, and gaze as head pose plus the eye's rotation in the head."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gaze_monitor import headpose as H
from gaze_monitor.attention import AttentionZone
from gaze_monitor.calibration import GazeCalibrator
from gaze_monitor.config import ConfigError, MonitorConfig

K = H.Intrinsics(fx=905.0, fy=907.0, cx=640.0, cy=360.0)
W, HGT = 1280, 720


def rot(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0) -> np.ndarray:
    """Face-to-camera rotation for a face turned by these angles from facing
    the camera head-on (yaw to image right, pitch down, roll clockwise)."""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    facing = np.diag([1.0, -1.0, -1.0])  # face looking straight at the camera
    Ry = np.array(
        [[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]]
    )
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]]
    )
    Rz = np.array(
        [[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]]
    )
    return Ry @ Rx @ Rz @ facing


def face(R: np.ndarray, t=(0.0, 0.0, 0.45), scale=1.0):
    """A full landmark array with the rigid points placed by (R, t, scale),
    normalised like FaceMesh's output, and the true depth at each."""
    lm = np.zeros((478, 3))
    depth = np.zeros(478)
    pts = scale * H.MODEL @ R.T + np.asarray(t)
    u = K.fx * pts[:, 0] / pts[:, 2] + K.cx
    v = K.fy * pts[:, 1] / pts[:, 2] + K.cy
    lm[H.RIGID, 0] = u / W
    lm[H.RIGID, 1] = v / HGT
    zref = np.median(pts[:, 2])
    lm[H.RIGID, 2] = (pts[:, 2] - zref) * K.fx / zref / W
    depth[H.RIGID] = pts[:, 2]
    return lm, depth


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    R = rot(25, -10, 5)
    dst = 1.3 * src @ R.T + np.array([0.1, -0.2, 0.5])
    s, R2, t = H.umeyama(src, dst)
    assert s == pytest.approx(1.3)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t, [0.1, -0.2, 0.5], atol=1e-9)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@pytest.mark.parametrize("angles", [(0, 0, 0), (25, 0, 0), (0, 35, 0), (-15, 40, 8)])
def test_depth_pose_recovers_the_head(angles):
    R = rot(*angles)
    lm, depth = face(R, scale=1.08)
    pose = H.estimate(lm, W, HGT, depth, K)
    assert pose.method == "depth"
    assert pose.scale == pytest.approx(1.08, rel=1e-6)
    assert rotation_error_deg(pose.rotation, R) < 1e-4


def test_pose_angles_read_naturally():
    for angles in [(0, 0, 0), (25, 0, 0), (0, 35, 0), (-20, 30, 0), (0, 0, 10)]:
        lm, depth = face(rot(*angles))
        pose = H.estimate(lm, W, HGT, depth, K)
        np.testing.assert_allclose(pose.angles_deg(), angles, atol=1e-4)


def test_depth_pose_survives_dropouts_and_outliers():
    R = rot(20, 30, 0)
    lm, depth = face(R)
    depth[H.RIGID[:6]] = 0.0  # no reading: the stereo shadow
    depth[H.RIGID[10:13]] += 0.05  # wild readings: a hand in front
    pose = H.estimate(lm, W, HGT, depth, K)
    assert pose.method == "depth"
    assert rotation_error_deg(pose.rotation, R) < 0.5


def test_without_depth_it_falls_back_to_facemesh_depth():
    for angles in [(0, 0, 0), (20, 30, 0), (-15, 40, 8)]:
        R = rot(*angles)
        lm, _ = face(R)
        pose = H.estimate(lm, W, HGT, None, None)
        assert pose.method == "landmarks"
        # Weak perspective at half a metre: a couple of degrees, not exact.
        assert rotation_error_deg(pose.rotation, R) < 3.0


def eye_offset(yaw_deg, pitch_deg, eye_width=H.CANONICAL_EYE_WIDTH):
    """The (dx, dy) an eye rotated this far in the head would show."""
    sy = math.sin(math.radians(yaw_deg))
    sp = math.sin(math.radians(pitch_deg))
    return sy * H.IRIS_PLANE_RADIUS / (eye_width / 2), sp * H.IRIS_PLANE_RADIUS / (
        eye_width / 4
    )


def test_gaze_is_head_plus_eye():
    lm, depth = face(rot(0, 0, 0))
    pose = H.estimate(lm, W, HGT, depth, K)
    assert H.gaze_angles_deg(pose, 0.0, 0.0) == pytest.approx((0.0, 0.0), abs=1e-6)
    dx, dy = eye_offset(10, 15)
    yaw, pitch = H.gaze_angles_deg(pose, dx, dy)
    assert yaw == pytest.approx(10, abs=0.5)
    assert pitch == pytest.approx(15, abs=0.5)


def offset_for(direction_in_face: np.ndarray, eye_width=H.CANONICAL_EYE_WIDTH):
    """The (dx, dy) of an eye whose visual axis, in the face frame, is this."""
    d = direction_in_face / np.linalg.norm(direction_in_face)
    yaw = math.degrees(math.atan2(d[0], d[2]))
    pitch = math.degrees(math.asin(-d[1]))
    return eye_offset(yaw, pitch, eye_width)


@pytest.mark.parametrize("head_yaw", [-20, -8, 0, 12, 25])
def test_gaze_held_while_the_head_turns_does_not_move(head_yaw):
    """VOR: the head turns and the eyes counter-rotate to hold the same line
    of sight. Eye-in-head moves; the line of sight, which is what this
    measures, must not."""
    R0 = rot(0, 30, 0)
    held = R0 @ H.eye_direction(*eye_offset(0, 10), H.CANONICAL_EYE_WIDTH)
    lm0, depth0 = face(R0)
    want = H.gaze_angles_deg(H.estimate(lm0, W, HGT, depth0, K), *eye_offset(0, 10))

    R = rot(head_yaw, 30, 0)
    lm, depth = face(R)
    pose = H.estimate(lm, W, HGT, depth, K)
    dx, dy = offset_for(R.T @ held)
    assert abs(dx - eye_offset(0, 10)[0]) > 0.05 or head_yaw == 0  # eye moved
    assert H.gaze_angles_deg(pose, dx, dy) == pytest.approx(want, abs=1e-3)


def test_landmark_depth_sampling_ignores_holes():
    depth = np.full((HGT, W), 0.5, dtype=np.float32)
    depth[100:103, 200:203] = 0.0  # a hole under one landmark's patch...
    depth[101, 201] = 0.6  # ...with one valid reading in it
    lm = np.array([[201 / W, 101 / HGT, 0.0], [400 / W, 300 / HGT, 0.0]])
    out = H.sample_landmark_depth(depth, lm, W, HGT)
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(0.5)
    depth[:] = 0.0
    assert H.sample_landmark_depth(depth, lm, W, HGT)[0] == 0.0


def test_degree_zones_scale_about_their_centre():
    zone = AttentionZone(-10.0, 10.0, 20.0, 40.0, calib_z=0.4, units="deg")
    # The centre stays the centre at any distance...
    assert zone.project(0.0, 30.0, 0.8) == pytest.approx((0.5, 0.5))
    # ...and further away the zone looks smaller, so the same direction off
    # its centre lies further out in zone units.
    u_near, _ = zone.project(5.0, 30.0, 0.4)
    u_far, _ = zone.project(5.0, 30.0, 0.8)
    assert u_near == pytest.approx(0.75)
    assert u_far == pytest.approx(1.0)


def test_zone_units_survive_the_calibration_record():
    zone = AttentionZone(-10.0, 10.0, 20.0, 40.0, units="deg")
    assert AttentionZone.from_dict(zone.to_dict()).units == "deg"
    old = zone.to_dict()
    del old["units"]
    assert AttentionZone.from_dict(old).units == "eye"


def test_calibrator_builds_zones_in_its_units():
    cal = GazeCalibrator(sample_duration=0.1, sigma_threshold=3.0, units="deg")
    for corner in ((-10, 20), (10, 20), (-10, 40), (10, 40)):
        cal.start_collection(0.0)
        for k in range(6):
            cal.collect(k * 0.02, corner[0] + 0.1 * k, corner[1])
        assert cal.validate_and_save()[0]
    assert cal.build_zone().units == "deg"


def test_config_validates_the_gaze_model():
    assert MonitorConfig().gaze_model == "eye"
    assert MonitorConfig(gaze_model="head").gaze_model == "head"
    with pytest.raises(ConfigError):
        MonitorConfig(gaze_model="iris")
