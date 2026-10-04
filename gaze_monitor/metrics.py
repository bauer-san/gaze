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
        self._disparity = Gauge(
            "gaze_eye_disparity",
            "Vertical disagreement between the two eyes; a noise floor, "
            "since conjugate eye movements have no vertical vergence",
            registry=reg,
        )
        self._disparity_rejected = Counter(
            "gaze_frames_rejected_disparity",
            "Frames discarded because the two eyes disagreed too far",
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

        # -- safety output --
        self._safety_enabled = Gauge(
            "gaze_safety_enabled", "1 when a safety output is configured", registry=reg
        )
        self._safety_armed = Gauge(
            "gaze_safety_armed",
            "1 once the output has been armed and may issue permits",
            registry=reg,
        )
        self._safety_permitted = Gauge(
            "gaze_safety_permitted",
            "1 when the last write was a permit, 0 when a stop was demanded",
            registry=reg,
        )
        self._safety_silent = Gauge(
            "gaze_safety_silent",
            "1 when renewals have stopped because the vision result went stale",
            registry=reg,
        )
        self._vision_staleness = Gauge(
            "gaze_vision_staleness_seconds",
            "Age of the most recent vision result, as the safety output sees it",
            registry=reg,
        )
        self._safety_writes = Counter(
            "gaze_safety_writes", "Permit renewals written", registry=reg
        )
        self._safety_write_errors = Counter(
            "gaze_safety_write_errors", "Permit renewals that failed", registry=reg
        )
        self._last_writes = 0
        self._last_write_errors = 0

        # -- machine state, read from the drive --
        self._machine_enabled = Gauge(
            "gaze_machine_enabled",
            "1 when a machine reader is configured",
            registry=reg,
        )
        self._machine_running = Gauge(
            "gaze_machine_running",
            "1 when the drive reports the motor turning in either direction",
            registry=reg,
        )
        self._machine_faulted = Gauge(
            "gaze_machine_faulted",
            "1 when the drive reports a fault trip",
            registry=reg,
        )
        self._machine_brake_released = Gauge(
            "gaze_machine_brake_released",
            "1 when the drive is asserting its brake release signal",
            registry=reg,
        )
        self._machine_current = Gauge(
            "gaze_machine_current_amps", "Drive output current", registry=reg
        )
        self._machine_hz = Gauge(
            "gaze_machine_output_hertz", "Drive output frequency", registry=reg
        )
        self._machine_age = Gauge(
            "gaze_machine_reading_age_seconds",
            "Age of the most recent good read from the drive",
            registry=reg,
        )
        self._machine_reads = Counter(
            "gaze_machine_reads", "Drive polls that returned data", registry=reg
        )
        self._machine_read_errors = Counter(
            "gaze_machine_read_errors", "Drive polls that failed", registry=reg
        )
        self._last_reads = 0
        self._last_read_errors = 0

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

    def observe(
        self,
        status,
        fps: float,
        camera_ok: bool,
        got_frame: bool,
        disparity: float | None = None,
        disparity_rejected: int = 0,
    ) -> None:
        """Record one frame.

        ``disparity`` is how far the two eyes disagreed vertically. Exported
        because it is a direct read on this instrument's own noise, which
        nothing else here provides: eye movements are conjugate, so vertical
        disagreement between the eyes is measurement error by construction.
        Watch it to tell a tuning problem from a tracking problem.
        """
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
        if disparity is not None:
            self._disparity.set(disparity)
        if disparity_rejected:
            self._disparity_rejected.inc(disparity_rejected)

        # The edge, not the level: this is what a slow scrape would otherwise
        # miss entirely.
        if status.in_zone and not self._was_in_zone:
            self._reentries.inc()
        self._was_in_zone = status.in_zone

    def observe_safety(self, safety, now: float) -> None:
        """Record the safety output's state.

        The output owns absolute counts, so the deltas are applied here --
        a Counter can only be incremented, never set.
        """
        if not self.enabled:
            return
        self._safety_enabled.set(1 if safety.enabled else 0)
        if not safety.enabled:
            return
        self._safety_armed.set(1 if safety.armed else 0)
        self._safety_permitted.set(1 if safety.last_written else 0)
        self._safety_silent.set(1 if safety.silent else 0)
        self._vision_staleness.set(safety.staleness(now))

        if safety.writes > self._last_writes:
            self._safety_writes.inc(safety.writes - self._last_writes)
            self._last_writes = safety.writes
        if safety.write_errors > self._last_write_errors:
            self._safety_write_errors.inc(safety.write_errors - self._last_write_errors)
            self._last_write_errors = safety.write_errors

    def observe_machine(self, machine, now: float) -> None:
        """Record what the drive last said about itself.

        Read-only context. Nothing here feeds back into the attention logic or
        the permit -- it is here so that "was the saw actually cutting?" can be
        answered afterwards from the same dashboard as everything else.
        """
        if not self.enabled:
            return
        self._machine_enabled.set(1 if machine.enabled else 0)
        if not machine.enabled:
            return
        self._machine_age.set(machine.age(now))
        latest = machine.latest()
        if latest is not None:
            self._machine_running.set(1 if latest.running else 0)
            self._machine_faulted.set(1 if latest.faulted else 0)
            self._machine_brake_released.set(1 if latest.brake_released else 0)
            self._machine_current.set(latest.current_a)
            self._machine_hz.set(latest.output_hz)

        if machine.reads > self._last_reads:
            self._machine_reads.inc(machine.reads - self._last_reads)
            self._last_reads = machine.reads
        if machine.read_errors > self._last_read_errors:
            self._machine_read_errors.inc(machine.read_errors - self._last_read_errors)
            self._last_read_errors = machine.read_errors

    def record_transition(self, old: AttentionState, new: AttentionState) -> None:
        if self.enabled:
            self._transitions.labels(from_state=old.value, to_state=new.value).inc()

    # -- reading back, for tests and diagnostics ---------------------------

    def value(self, name: str, **labels) -> float | None:
        """Current value of one metric, or None if the exporter is inert."""
        if not self.enabled or self._registry is None:
            return None
        return self._registry.get_sample_value(name, labels or None)
