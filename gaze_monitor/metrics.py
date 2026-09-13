"""Prometheus metrics, for watching a unit that has no display.

The log is the installed unit's annunciator, and it is a poor instrument: it
records *transitions* and nothing else. Two of the by-hand checks in
jetson/CI_NOTES.md cannot be settled from it at all. The hysteresis check is
the clearest case -- a glance back shorter than ``clear_after`` deliberately
changes no state, so it writes no line, and "the hysteresis held" looks
exactly like "the operator never glanced back".

Counters fix that, and they fix it in a way that survives slow scraping, which
matters because a scrape interval of seconds will step straight over a glance
of a third of a second:

* ``gaze_zone_reentries_total`` rises every time the gaze comes back inside
  the zone, whatever the state machine decides to do about it.
* ``gaze_state_transitions_total{from_state="alert",to_state="attentive"}``
  rises only when an alarm actually clears.

A brief glance back while alarming moves the first and not the second. That
difference is the hysteresis, it needs no stopwatch, and it is still there in
Grafana tomorrow. A gauge sampled every 15 seconds could never show it.

Everything here degrades to nothing: no port configured, or no
prometheus_client installed, and the calls become no-ops so the monitor runs
exactly as before.
"""

from __future__ import annotations

import logging

from .attention import AttentionState

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whether the import lands
    from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

DEFAULT_PORT = 9091


class Metrics:
    """Exports the monitor's state. Inert unless started with a port.

    Counter names omit the ``_total`` suffix: prometheus_client appends it,
    and spelling it here produces ``gaze_frames_total_total``.
    """

    def __init__(self, port: int = 0) -> None:
        self.port = port
        self.enabled = False
        self._registry = None
        self._was_in_zone = False

    # -- lifecycle ---------------------------------------------------------

    def start(self, serve: bool = True) -> bool:
        """Bring the exporter up. False means it stayed inert, and why."""
        if self.port <= 0:
            return False
        if not AVAILABLE:
            log.warning(
                "metrics_port is %d but prometheus_client is not installed; "
                "no metrics will be exported (pip install prometheus-client)",
                self.port,
            )
            return False

        # A private registry rather than the default one, so building a second
        # Metrics in one process -- which is what the tests do -- does not
        # collide with the first on every metric name.
        self._registry = CollectorRegistry()
        reg = self._registry

        self._state = Gauge(
            "gaze_state",
            "Current attention state, 1 for the active one and 0 for the rest",
            ["state"],
            registry=reg,
        )
        self._away = Gauge(
            "gaze_away_seconds",
            "Seconds the operator has continuously been away from the zone",
            registry=reg,
        )
        self._in_zone = Gauge(
            "gaze_in_zone", "1 when the gaze is inside the zone", registry=reg
        )
        self._tracked = Gauge(
            "gaze_tracked", "1 when the operator's eyes were measurable", registry=reg
        )
        self._camera_ok = Gauge(
            "gaze_camera_ok",
            "1 when the camera is delivering usable frames",
            registry=reg,
        )
        self._distance = Gauge(
            "gaze_operator_distance_meters",
            "Measured operator distance; absent as 0 when there is no reading",
            registry=reg,
        )
        self._calibration_distance = Gauge(
            "gaze_calibration_distance_meters",
            "Distance the zone was calibrated at, for comparison with the live one",
            registry=reg,
        )
        self._fps = Gauge(
            "gaze_frame_rate_fps",
            "Frames per second through the pipeline",
            registry=reg,
        )
        self._frames = Counter(
            "gaze_frames", "Frames read from the camera", registry=reg
        )
        self._dropped = Counter(
            "gaze_dropped_frames", "Frames the camera failed to deliver", registry=reg
        )
        self._transitions = Counter(
            "gaze_state_transitions",
            "Attention state changes",
            ["from_state", "to_state"],
            registry=reg,
        )
        self._reentries = Counter(
            "gaze_zone_reentries",
            "Times the gaze re-entered the zone, whether or not the state changed",
            registry=reg,
        )

        for state in AttentionState:
            self._state.labels(state=state.value).set(0)

        if serve:
            try:
                start_http_server(self.port, registry=reg)
            except OSError as exc:
                # Exporting is diagnostics; watching the operator is the job.
                # Losing the first must not cost the second.
                log.error(
                    "Could not serve metrics on port %d: %s. Monitoring continues.",
                    self.port,
                    exc,
                )
                self._registry = None
                return False
            log.info("Metrics exported on port %d", self.port)
        self.enabled = True
        return True

    # -- recording ---------------------------------------------------------

    def set_calibration_distance(self, z_m: float) -> None:
        if self.enabled:
            self._calibration_distance.set(z_m)

    def observe(self, status, fps: float, camera_ok: bool, got_frame: bool) -> None:
        """Record one frame."""
        if not self.enabled:
            return

        if got_frame:
            self._frames.inc()
        else:
            self._dropped.inc()

        for state in AttentionState:
            self._state.labels(state=state.value).set(1 if state is status.state else 0)
        self._away.set(status.away_seconds)
        self._in_zone.set(1 if status.in_zone else 0)
        self._tracked.set(1 if status.tracked else 0)
        self._camera_ok.set(1 if camera_ok else 0)
        self._distance.set(status.z_m)
        self._fps.set(fps)

        # The edge, not the level: this is what a slow scrape would otherwise
        # miss entirely.
        if status.in_zone and not self._was_in_zone:
            self._reentries.inc()
        self._was_in_zone = status.in_zone

    def record_transition(self, old: AttentionState, new: AttentionState) -> None:
        if self.enabled:
            self._transitions.labels(from_state=old.value, to_state=new.value).inc()

    # -- reading back, for tests and diagnostics ---------------------------

    def value(self, name: str, **labels) -> float | None:
        """Current value of one metric, or None if the exporter is inert."""
        if not self.enabled or self._registry is None:
            return None
        return self._registry.get_sample_value(name, labels or None)
