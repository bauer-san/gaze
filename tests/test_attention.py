"""Tests for the attention state machine.

Time is injected, so escalation and hysteresis are exercised without sleeping.
"""

from dataclasses import dataclass

import pytest

from kinect_gaze.attention import (
    AttentionMonitor,
    AttentionState,
    AttentionZone,
)


@dataclass
class FakeGaze:
    """Minimal stand-in for kinect_gaze.gaze.GazeSample."""

    dx: float
    dy: float
    z_m: float = 0.0


def square_zone(margin=0.0, calib_z=0.0):
    """A zone spanning -0.5..0.5 on both axes."""
    return AttentionZone.from_corners(
        [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)],
        calib_z=calib_z,
        margin=margin,
    )


def monitor(**kwargs):
    params = dict(warn_after=1.0, alert_after=3.0, clear_after=0.4, fault_after=1.0)
    params.update(kwargs)
    return AttentionMonitor(zone=square_zone(), **params)


# --- AttentionZone --------------------------------------------------------


def test_from_corners_averages_opposing_edges():
    z = AttentionZone.from_corners([(-0.4, -0.6), (0.6, -0.4), (-0.6, 0.4), (0.4, 0.6)])
    assert z.dx_min == pytest.approx(-0.5)
    assert z.dx_max == pytest.approx(0.5)
    assert z.dy_min == pytest.approx(-0.5)
    assert z.dy_max == pytest.approx(0.5)


def test_from_corners_requires_four_points():
    with pytest.raises(ValueError):
        AttentionZone.from_corners([(0.0, 0.0), (1.0, 1.0)])


def test_degenerate_zone_is_rejected():
    with pytest.raises(ValueError, match="Degenerate"):
        AttentionZone.from_corners([(0.0, 0.0)] * 4)


def test_project_maps_corners_to_unit_square():
    z = square_zone()
    assert z.project(-0.5, -0.5) == pytest.approx((0.0, 0.0))
    assert z.project(0.5, 0.5) == pytest.approx((1.0, 1.0))
    assert z.project(0.0, 0.0) == pytest.approx((0.5, 0.5))


def test_mirrored_zone_still_projects():
    """A mirrored camera gives dx_min > dx_max; the signed span handles it."""
    z = AttentionZone.from_corners([(0.5, -0.5), (-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    assert z.contains(0.0, 0.0)
    assert not z.contains(2.0, 0.0)


def test_contains_respects_margin():
    tight = square_zone(margin=0.0)
    loose = square_zone(margin=0.2)
    # u = 1.1, just outside the raw zone
    assert not tight.contains(0.6, 0.0)
    assert loose.contains(0.6, 0.0)


def test_contains_rejects_non_finite():
    assert not square_zone().contains(float("nan"), 0.0)


def test_depth_compensation_scales_displacement():
    """Stepping back widens the apparent displacement for the same gaze."""
    z = square_zone(calib_z=1.0)
    # Calibrated at 1.0 m; at 2.0 m the same dx projects twice as far out.
    assert z.project(0.25, 0.0, z_m=1.0)[0] == pytest.approx(0.75)
    assert z.project(0.25, 0.0, z_m=2.0)[0] == pytest.approx(1.0)


def test_depth_compensation_skipped_without_readings():
    z = square_zone(calib_z=0.0)
    assert z.project(0.25, 0.0, z_m=2.0)[0] == pytest.approx(0.75)


def test_zone_roundtrips_through_dict():
    z = square_zone(margin=0.15, calib_z=1.25)
    assert AttentionZone.from_dict(z.to_dict()) == z


def test_zone_from_dict_rejects_malformed():
    with pytest.raises(ValueError):
        AttentionZone.from_dict({"dx_min": 0.0})


# --- escalation -----------------------------------------------------------


def test_gaze_inside_zone_stays_attentive():
    m = monitor()
    for t in (0.0, 1.0, 5.0, 60.0):
        status = m.update(t, FakeGaze(0.0, 0.0))
    assert status.state is AttentionState.ATTENTIVE
    assert status.in_zone


def test_escalates_attentive_warning_alert():
    m = monitor()
    away = FakeGaze(5.0, 0.0)

    assert m.update(0.0, away).state is AttentionState.ATTENTIVE  # grace
    assert m.update(0.9, away).state is AttentionState.ATTENTIVE
    assert m.update(1.0, away).state is AttentionState.WARNING
    assert m.update(2.9, away).state is AttentionState.WARNING
    assert m.update(3.0, away).state is AttentionState.ALERT
    assert m.update(10.0, away).is_alarming


def test_away_seconds_tracks_continuous_absence():
    m = monitor()
    away = FakeGaze(5.0, 0.0)
    m.update(0.0, away)
    assert m.update(4.5, away).away_seconds == pytest.approx(4.5)


def test_untracked_face_counts_as_looking_away():
    """No measurable eyes is indistinguishable from turning away; alarm."""
    m = monitor()
    m.update(0.0, None)
    status = m.update(3.5, None)
    assert status.state is AttentionState.ALERT
    assert not status.tracked


# --- hysteresis -----------------------------------------------------------


def test_returning_briefly_does_not_clear_alert():
    m = monitor()
    away, back = FakeGaze(5.0, 0.0), FakeGaze(0.0, 0.0)
    m.update(0.0, away)
    assert m.update(3.0, away).state is AttentionState.ALERT
    # Back in the zone, but only for 0.2 s -- under clear_after.
    assert m.update(3.2, back).state is AttentionState.ALERT


def test_sustained_return_clears_alert():
    m = monitor()
    away, back = FakeGaze(5.0, 0.0), FakeGaze(0.0, 0.0)
    m.update(0.0, away)
    m.update(3.0, away)
    m.update(3.1, back)
    assert m.update(3.6, back).state is AttentionState.ATTENTIVE


def test_flicker_out_during_clear_window_restarts_it():
    m = monitor()
    away, back = FakeGaze(5.0, 0.0), FakeGaze(0.0, 0.0)
    m.update(0.0, away)
    m.update(3.0, away)
    m.update(3.1, back)
    m.update(3.3, away)  # flickers back out before clearing
    m.update(3.4, back)
    assert m.update(3.7, back).state is AttentionState.ALERT
    assert m.update(3.9, back).state is AttentionState.ATTENTIVE


def test_away_timer_frozen_during_clear_window():
    m = monitor()
    away, back = FakeGaze(5.0, 0.0), FakeGaze(0.0, 0.0)
    m.update(0.0, away)
    first = m.update(3.0, away).away_seconds
    assert m.update(3.2, back).away_seconds == pytest.approx(first + 0.2)


# --- faults ---------------------------------------------------------------


def test_brief_camera_dropout_holds_previous_state():
    m = monitor()
    m.update(0.0, FakeGaze(0.0, 0.0))
    status = m.update(0.5, None, camera_ok=False)
    assert status.state is AttentionState.ATTENTIVE


def test_sustained_camera_failure_raises_fault():
    m = monitor()
    m.update(0.0, FakeGaze(0.0, 0.0))
    m.update(0.5, None, camera_ok=False)
    assert m.update(1.5, None, camera_ok=False).state is AttentionState.FAULT


def test_fault_is_alarming():
    assert AttentionState.FAULT.is_alarming


def test_camera_recovery_leaves_fault():
    m = monitor()
    m.update(0.0, None, camera_ok=False)
    m.update(2.0, None, camera_ok=False)
    assert m.state is AttentionState.FAULT
    assert m.update(2.1, FakeGaze(0.0, 0.0)).state is AttentionState.ATTENTIVE


def test_uncalibrated_monitor_does_not_alarm():
    m = AttentionMonitor(zone=None)
    assert m.update(0.0, None).state is AttentionState.ATTENTIVE
    assert m.update(100.0, None).state is AttentionState.ATTENTIVE


def test_uncalibrated_monitor_still_reports_camera_fault():
    m = AttentionMonitor(zone=None, fault_after=1.0)
    m.update(0.0, None, camera_ok=False)
    assert m.update(2.0, None, camera_ok=False).state is AttentionState.FAULT


# --- transitions and config ----------------------------------------------


def test_transition_hook_fires_once_per_change():
    seen = []
    m = monitor(on_transition=lambda old, new, st: seen.append((old, new)))
    away = FakeGaze(5.0, 0.0)
    m.update(0.0, away)
    m.update(1.0, away)
    m.update(1.5, away)  # still WARNING -- no second callback
    m.update(3.0, away)
    assert seen == [
        (AttentionState.ATTENTIVE, AttentionState.WARNING),
        (AttentionState.WARNING, AttentionState.ALERT),
    ]


def test_reset_returns_to_attentive():
    m = monitor()
    away = FakeGaze(5.0, 0.0)
    m.update(0.0, away)
    m.update(5.0, away)
    m.reset()
    assert m.state is AttentionState.ATTENTIVE
    assert m.update(5.1, away).away_seconds == pytest.approx(0.0)


def test_rejects_warn_after_above_alert_after():
    with pytest.raises(ValueError):
        AttentionMonitor(zone=square_zone(), warn_after=5.0, alert_after=1.0)
