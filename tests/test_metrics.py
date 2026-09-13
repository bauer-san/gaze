"""The Prometheus exporter.

The counters here exist to answer a question the log cannot: whether a brief
glance back while alarming cleared the alarm. That case is exercised directly
below, because it is the reason the module exists.
"""

import pytest

from gaze_monitor.attention import AttentionState, AttentionStatus
from gaze_monitor.metrics import Metrics

pytest.importorskip("prometheus_client")


def _metrics() -> Metrics:
    """An exporter with every metric live but no socket bound."""
    m = Metrics(port=9091)
    assert m.start(serve=False) is True
    return m


def _status(state=AttentionState.ATTENTIVE, in_zone=True, **kwargs):
    return AttentionStatus(state=state, in_zone=in_zone, **kwargs)


# -- staying inert ---------------------------------------------------------


def test_no_port_means_no_exporter():
    m = Metrics(port=0)
    assert m.start(serve=False) is False
    assert m.enabled is False


def test_an_inert_exporter_accepts_every_call():
    """The run loop calls these unconditionally; they must not blow up when
    metrics are switched off, which is the default."""
    m = Metrics(port=0)
    m.start(serve=False)
    m.observe(_status(), 30.0, True, True)
    m.record_transition(AttentionState.ATTENTIVE, AttentionState.WARNING)
    m.set_calibration_distance(0.65)
    assert m.value("gaze_frames_total") is None


# -- state and gauges ------------------------------------------------------


def test_only_the_current_state_is_set():
    m = _metrics()
    m.observe(_status(state=AttentionState.ALERT), 30.0, True, True)
    assert m.value("gaze_state", state="alert") == 1
    assert m.value("gaze_state", state="attentive") == 0
    assert m.value("gaze_state", state="fault") == 0


def test_gauges_follow_the_status():
    m = _metrics()
    m.observe(
        _status(
            state=AttentionState.WARNING,
            away_seconds=1.5,
            in_zone=False,
            tracked=True,
            z_m=1.62,
        ),
        24.5,
        True,
        True,
    )
    assert m.value("gaze_away_seconds") == pytest.approx(1.5)
    assert m.value("gaze_in_zone") == 0
    assert m.value("gaze_tracked") == 1
    assert m.value("gaze_operator_distance_meters") == pytest.approx(1.62)
    assert m.value("gaze_frame_rate_fps") == pytest.approx(24.5)


def test_calibration_distance_is_exported_for_comparison():
    """Plotted against the live distance, this is the backing-up check."""
    m = _metrics()
    m.set_calibration_distance(0.649)
    assert m.value("gaze_calibration_distance_meters") == pytest.approx(0.649)


def test_dropped_frames_are_counted_separately():
    m = _metrics()
    m.observe(_status(), 30.0, True, True)
    m.observe(_status(), 30.0, False, False)
    assert m.value("gaze_frames_total") == 1
    assert m.value("gaze_dropped_frames_total") == 1
    assert m.value("gaze_camera_ok") == 0


# -- the counters that survive a slow scrape -------------------------------


def test_zone_reentry_counts_the_edge_not_the_level():
    """A gauge sampled every 15s cannot see a glance; a counter can."""
    m = _metrics()
    m.observe(_status(in_zone=False), 30.0, True, True)
    assert m.value("gaze_zone_reentries_total") == 0
    m.observe(_status(in_zone=True), 30.0, True, True)
    assert m.value("gaze_zone_reentries_total") == 1
    m.observe(_status(in_zone=True), 30.0, True, True)  # still in: no new edge
    assert m.value("gaze_zone_reentries_total") == 1
    m.observe(_status(in_zone=False), 30.0, True, True)
    m.observe(_status(in_zone=True), 30.0, True, True)
    assert m.value("gaze_zone_reentries_total") == 2


def test_transitions_are_counted_by_edge():
    m = _metrics()
    m.record_transition(AttentionState.WARNING, AttentionState.ALERT)
    m.record_transition(AttentionState.WARNING, AttentionState.ALERT)
    m.record_transition(AttentionState.ALERT, AttentionState.ATTENTIVE)
    assert (
        m.value("gaze_state_transitions_total", from_state="warning", to_state="alert")
        == 2
    )
    assert (
        m.value(
            "gaze_state_transitions_total", from_state="alert", to_state="attentive"
        )
        == 1
    )


def test_a_brief_glance_back_while_alarming_is_distinguishable():
    """The whole point of the module.

    The operator is alarming, glances back into the zone for less than
    clear_after, and looks away again. Hysteresis holds, so the state never
    changes and the log stays silent. The re-entry counter still moves, and
    the alarm-cleared counter does not -- which is the difference, visible at
    any scrape interval.
    """
    m = _metrics()
    alerting = dict(state=AttentionState.ALERT, tracked=True)

    m.observe(_status(in_zone=False, **alerting), 30.0, True, True)
    m.observe(_status(in_zone=True, **alerting), 30.0, True, True)  # the glance
    m.observe(_status(in_zone=False, **alerting), 30.0, True, True)

    assert m.value("gaze_zone_reentries_total") == 1
    assert (
        m.value(
            "gaze_state_transitions_total",
            from_state="alert",
            to_state="attentive",
        )
        is None
    )
