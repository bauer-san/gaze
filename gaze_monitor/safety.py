"""Optional safety output: a permit that has to be continuously renewed.

Disabled unless configured. Nothing here runs, connects or writes anything
until ``brake_after`` and ``safety_host`` are both set.

**The output is a permit, not a stop command, and that inversion is the whole
design.** A naive integration sends "stop" when the operator is inattentive,
and fails dangerously by omission: a hung process, a crashed container or a
cut cable sends nothing, and the machine runs on none the wiser. Writing a
permit instead makes silence mean stop. Every failure on this side --
including the ones nobody thought of -- lands on the machine not being
permitted to run.

That only holds if the far end treats loss of communication as a demand to
stop. A drive with a communications-loss timeout configured to fault or brake
provides that; one configured to ignore the timeout, or to hold its last
value, turns this module into decoration. Verify it by pulling the cable and
watching the machine stop, not by reading a configuration page.

Three states are deliberately distinguished:

* **permit** -- attention verified, the vision pipeline is fresh, and the
  output has been armed. A ``True`` write.
* **explicit refusal** -- something is known to be wrong: the operator is
  inattentive past ``brake_after``, or the output has not been armed yet. A
  ``False`` write. Communication continues, so the far end can tell the
  difference between a considered refusal and a dead link.
* **silence** -- the vision result has gone stale, so this module does not
  know what is happening and declines to claim anything. Writes stop, and the
  far end's own timeout is left to act.

The distinction between the second and third matters when something goes
wrong at three in the morning and somebody has to work out which part failed.

No OpenCV, no numpy: this is one of the modules that has to be testable on a
machine with no vision stack, and the tests drive it against a loopback
Modbus server with no hardware at all.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from .attention import AttentionState

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whether the import lands
    from pymodbus.client import ModbusTcpClient

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

COIL = "coil"
REGISTER = "register"
KINDS = (COIL, REGISTER)


class Backend:
    """Somewhere a permit can be written.

    Kept behind an interface because the far end is not settled: a register on
    a drive and a coil on a discrete I/O module are the same operation from
    here, and swapping one for the other should not reach any other module.
    """

    def write(self, permitted: bool) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ModbusBackend(Backend):
    """Modbus/TCP. Reconnects on its own, because a permit writer that gives
    up after one network blip is a permit writer that stops the machine."""

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        kind: str = COIL,
        address: int = 0,
    ) -> None:
        if not AVAILABLE:
            raise RuntimeError(
                "pymodbus is required for the safety output: "
                "pip install -r requirements.txt"
            )
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.kind = kind
        self.address = address
        self._client = ModbusTcpClient(host, port=port)

    def write(self, permitted: bool) -> None:
        if not self._client.connected:
            # connect() returning False is reported by the write below, which
            # keeps one error path instead of two.
            self._client.connect()
        if self.kind == COIL:
            result = self._client.write_coil(
                self.address, bool(permitted), device_id=self.unit_id
            )
        else:
            result = self._client.write_register(
                self.address, 1 if permitted else 0, device_id=self.unit_id
            )
        if result is None or (hasattr(result, "isError") and result.isError()):
            raise OSError(f"Modbus write rejected: {result!r}")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - closing must not raise
            pass


class SafetyOutput:
    """Renews a permit while attention is verified.

    ``tick`` is the whole state machine and takes the time explicitly, so the
    tests can step it without threads or sleeps. ``start`` runs it on a
    background thread for real use -- deliberately not on the capture loop,
    because a slow frame must not look like a dead process.
    """

    def __init__(self, config, backend: Backend | None = None) -> None:
        self.brake_after = float(getattr(config, "brake_after", 0.0) or 0.0)
        self.brake_on_fault = bool(getattr(config, "brake_on_fault", False))
        self.interval = float(getattr(config, "safety_interval", 0.1))
        self.stale_after = float(getattr(config, "safety_stale_after", 0.5))
        self._host = getattr(config, "safety_host", "") or ""

        # Enabled when there is somewhere to write: a configured host, or an
        # injected backend, which is how the tests reach it without a network.
        self.enabled = backend is not None or bool(self._host)

        self._backend = backend
        self._config = config
        self._armed = bool(getattr(config, "safety_auto_arm", False))

        self._lock = threading.Lock()
        self._permitted = False
        self._seen_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # counters, read by the metrics exporter
        self.writes = 0
        self.write_errors = 0
        self.last_written: bool | None = None
        self.silent = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Connect and begin renewing. False means it stayed disabled."""
        if not self.enabled:
            return False
        if self._backend is None:
            self._backend = ModbusBackend(
                host=self._host,
                port=int(getattr(self._config, "safety_port", 502)),
                unit_id=int(getattr(self._config, "safety_unit_id", 1)),
                kind=getattr(self._config, "safety_kind", COIL),
                address=int(getattr(self._config, "safety_address", 0)),
            )

        if self.brake_after <= 0:
            log.warning(
                "Safety output is connected but brake_after is 0, so a stop "
                "will never be demanded. This is a link test, not protection."
            )
        if self._armed:
            log.warning(
                "Safety output armed automatically (safety_auto_arm). The "
                "start-up check that the machine cannot run is skipped."
            )
        else:
            log.info(
                "Safety output is refusing a permit until armed. Confirm the "
                "machine cannot start, then arm it (SIGUSR1)."
            )

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="gaze-safety", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Refuse a permit, then shut down. Order matters."""
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._backend is not None:
            try:
                self._backend.write(False)
            except Exception as exc:
                log.warning("Could not refuse the permit on shutdown: %s", exc)
            self._backend.close()
        self.last_written = False

    def arm(self) -> None:
        """Allow permits to be issued, once the start-up check has passed."""
        if not self.enabled:
            return
        with self._lock:
            already, self._armed = self._armed, True
        if not already:
            log.warning("Safety output ARMED: the machine may now be permitted.")

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    # -- from the capture loop ---------------------------------------------

    def observe(self, status, now: float) -> None:
        """Record this frame's verdict. Cheap, and safe to call every frame."""
        if not self.enabled:
            return
        permitted = True
        if self.brake_after > 0 and status.away_seconds >= self.brake_after:
            permitted = False
        if self.brake_on_fault and status.state is AttentionState.FAULT:
            permitted = False
        with self._lock:
            self._permitted = permitted
            self._seen_at = now

    # -- the state machine -------------------------------------------------

    def tick(self, now: float) -> None:
        """One renewal decision. The thread calls this; so do the tests."""
        if not self.enabled or self._backend is None:
            return

        with self._lock:
            permitted, seen_at, armed = self._permitted, self._seen_at, self._armed

        if seen_at is None or (now - seen_at) > self.stale_after:
            # Nothing recent to base a claim on. Say nothing and let the far
            # end's timeout decide -- that is what it is for.
            self.silent = True
            return

        self.silent = False
        value = bool(permitted and armed)
        try:
            self._backend.write(value)
        except Exception as exc:
            self.write_errors += 1
            if self.write_errors == 1 or self.write_errors % 100 == 0:
                log.error(
                    "Safety output write failed (%d so far): %s",
                    self.write_errors,
                    exc,
                )
            return
        self.writes += 1
        if value != self.last_written:
            log.log(
                logging.INFO if value else logging.WARNING,
                "Safety output -> %s",
                "PERMIT" if value else "STOP DEMANDED",
            )
        self.last_written = value

    def staleness(self, now: float) -> float:
        """Seconds since the capture loop last reported. 0 if never."""
        with self._lock:
            seen_at = self._seen_at
        return 0.0 if seen_at is None else max(0.0, now - seen_at)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick(time.monotonic())
            self._stop.wait(self.interval)


class _Terminated(KeyboardInterrupt):
    """SIGTERM, routed through the same path as Ctrl-C."""


def install_signal_handlers(output: SafetyOutput) -> list[str]:
    """Arm on SIGUSR1, and make SIGTERM reach the cleanup path.

    SIGTERM matters more than it looks. Python does not turn it into an
    exception, so the default disposition kills the process outright and the
    monitor's cleanup never runs -- which leaves the camera open, the terminal
    in cbreak mode, and, worst of all, a permit still standing. `docker stop`
    sends SIGTERM, so that is the ordinary shutdown, not an edge case.
    Raising from the handler routes it through the same path as Ctrl-C.

    Returns the names of the signals actually installed.
    """

    def _terminate(*_args):
        raise _Terminated()

    installed: list[str] = []
    handlers = [("SIGTERM", _terminate)]
    if output.enabled:
        handlers.append(("SIGUSR1", lambda *_: output.arm()))

    for name, handler in handlers:
        sig = getattr(signal, name, None)
        if sig is None:  # pragma: no cover - not POSIX
            continue
        try:
            signal.signal(sig, handler)
        except ValueError:  # pragma: no cover - not the main thread
            log.warning("Could not install the %s handler", name)
            continue
        installed.append(name)
    return installed
