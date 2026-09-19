"""Optional safety output: a permit that has to be continuously renewed.

Disabled unless configured. Nothing here runs, opens a port or writes
anything until ``brake_after`` is set together with somewhere to write it.

Three transports, chosen by ``safety_transport``:

* ``tcp`` -- Modbus/TCP to ``safety_host``. Also what the loopback tests use.
* ``rtu`` -- Modbus RTU over a serial port, for a device on an RS-485 bus.
* ``relay`` -- a USB relay module, which speaks no Modbus at all and simply
  opens and closes one contact. Unplugging it de-energises the relay, which
  opens the contact, which is the correct failure.

They differ only in how the permit reaches the far end. Everything above the
backend -- the renewal thread, the staleness rule, the refusal until armed --
is identical, and that is the point of the ``Backend`` interface.

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
Modbus server and a pseudo-terminal, with no hardware at all.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from .attention import AttentionState

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whether the import lands
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

try:  # pragma: no cover - only the relay transport needs it
    import serial

    SERIAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    SERIAL_AVAILABLE = False

COIL = "coil"
REGISTER = "register"
KINDS = (COIL, REGISTER)

TCP = "tcp"
RTU = "rtu"
RELAY = "relay"
TRANSPORTS = (TCP, RTU, RELAY)

NUMATO = "numato"
LCUS = "lcus"
RELAY_PROTOCOLS = (NUMATO, LCUS)

# Numato boards address relays 0-9 then A onwards; a 1- or 2-channel module
# never leaves the first column, but getting this silently wrong on a larger
# board would switch the wrong contact.
_NUMATO_CHANNELS = "0123456789ABCDEFGHIJKLMNOPQRSTUV"


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


class _ModbusBackend(Backend):
    """The Modbus write path, independent of what carries it.

    TCP and RTU differ only in how the client is constructed. The coil-or-
    register decision, the reconnect and the error handling are identical, and
    duplicating them across two classes is how the two quietly drift apart.

    Reconnects on its own, because a permit writer that gives up after one
    blip is a permit writer that stops the machine.
    """

    def __init__(
        self,
        client,
        unit_id: int,
        kind: str,
        address: int,
        where: str,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self._client = client
        self.unit_id = unit_id
        self.kind = kind
        self.address = address
        self.where = where

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

    def describe(self) -> str:
        return f"Modbus {self.kind} {self.address} at {self.where}"


def _require_pymodbus() -> None:
    if not AVAILABLE:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "pymodbus is required for the safety output: "
            "pip install -r requirements.txt"
        )


class ModbusBackend(_ModbusBackend):
    """Modbus/TCP."""

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        kind: str = COIL,
        address: int = 0,
    ) -> None:
        _require_pymodbus()
        self.host = host
        self.port = port
        super().__init__(
            ModbusTcpClient(host, port=port),
            unit_id=unit_id,
            kind=kind,
            address=address,
            where=f"{host}:{port} unit {unit_id}",
        )


class ModbusSerialBackend(_ModbusBackend):
    """Modbus RTU over a serial port, typically a USB to RS-485 adapter.

    The line settings have to match the far end exactly. They fail as silence
    rather than as an error, which on a two-wire bus is the hardest kind of
    fault to find -- so they are configuration, not guesses.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        parity: str = "N",
        stopbits: int = 1,
        bytesize: int = 8,
        unit_id: int = 1,
        kind: str = COIL,
        address: int = 0,
        timeout: float = 1.0,
    ) -> None:
        _require_pymodbus()
        self.port = port
        super().__init__(
            # framer defaults to RTU for the serial client; left implicit
            # rather than named, because the enum has moved once already.
            ModbusSerialClient(
                port,
                baudrate=baudrate,
                parity=parity,
                stopbits=stopbits,
                bytesize=bytesize,
                timeout=timeout,
            ),
            unit_id=unit_id,
            kind=kind,
            address=address,
            where=f"{port} {baudrate} {bytesize}{parity}{stopbits} unit {unit_id}",
        )


class SerialRelayBackend(Backend):
    """A USB relay module: one contact, open or closed, and no protocol.

    These modules are the cheapest possible way to get a dry contact out of a
    PC, and their failure behaviour happens to be the one we want. Unplug the
    module and it loses bus power, the coil de-energises, the contact opens,
    and a machine wired to interpret an open contact as a stop does so without
    anything in software having noticed. The write raising afterwards is a
    courtesy, not the mechanism.

    Two dialects, because there is no standard:

    * ``numato``  -- line-based ASCII over CDC-ACM: ``relay on 0``. Testable
      from a shell with ``echo``, which is most of why it is the default.
    * ``lcus``    -- four raw bytes, CH340 based: ``A0 01 01 A2``.

    The port is opened lazily and reopened after a failure, so a module that
    is unplugged and plugged back in recovers without a restart.
    """

    def __init__(
        self,
        port: str,
        protocol: str = NUMATO,
        channel: int = 0,
        baudrate: int = 9600,
        timeout: float = 0.5,
    ) -> None:
        if not SERIAL_AVAILABLE:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "pyserial is required for the relay transport: "
                "pip install -r requirements.txt"
            )
        if protocol not in RELAY_PROTOCOLS:
            raise ValueError(
                f"protocol must be one of {RELAY_PROTOCOLS}, got {protocol!r}"
            )
        if channel < 0 or channel >= len(_NUMATO_CHANNELS):
            raise ValueError(f"channel out of range: {channel}")
        self.port = port
        self.protocol = protocol
        self.channel = channel
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None

    def _frame(self, permitted: bool) -> bytes:
        if self.protocol == NUMATO:
            verb = "on" if permitted else "off"
            return f"relay {verb} {_NUMATO_CHANNELS[self.channel]}\r".encode("ascii")
        # LCUS: 0xA0, 1-based channel, state, then the low byte of their sum.
        body = bytes((0xA0, self.channel + 1, 1 if permitted else 0))
        return body + bytes((sum(body) & 0xFF,))

    def _open(self):
        if self._serial is None or not self._serial.is_open:
            # write_timeout matters more than it looks: without it a wedged
            # USB device blocks the renewal thread forever, which stops the
            # permit without ever counting an error.
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        return self._serial

    def write(self, permitted: bool) -> None:
        try:
            handle = self._open()
            handle.write(self._frame(permitted))
            handle.flush()
            # Numato echoes the command and a prompt. Nothing reads them, so
            # drop them rather than let the buffer fill over a long shift.
            handle.reset_input_buffer()
        except Exception:
            # Drop the handle so the next renewal opens a fresh one; a stale
            # file descriptor after a replug never recovers on its own.
            self.close()
            raise

    def close(self) -> None:
        handle, self._serial = self._serial, None
        if handle is not None:
            try:
                handle.close()
            except Exception:  # pragma: no cover - closing must not raise
                pass

    def describe(self) -> str:
        return f"{self.protocol} relay channel {self.channel} at {self.port}"


def destination(config) -> str:
    """The configured far end, or empty if there is none.

    Which field counts depends on the transport, and this is the one place
    that knows -- the config validator names the same field in its error.
    """
    if getattr(config, "safety_transport", TCP) == TCP:
        return getattr(config, "safety_host", "") or ""
    return getattr(config, "safety_serial_port", "") or ""


def create_backend(config) -> Backend:
    """Build the backend ``safety_transport`` asks for."""
    transport = getattr(config, "safety_transport", TCP)
    if transport == RELAY:
        return SerialRelayBackend(
            port=getattr(config, "safety_serial_port", ""),
            protocol=getattr(config, "safety_relay_protocol", NUMATO),
            channel=int(getattr(config, "safety_relay_channel", 0)),
            baudrate=int(getattr(config, "safety_baud", 9600)),
        )
    if transport == RTU:
        return ModbusSerialBackend(
            port=getattr(config, "safety_serial_port", ""),
            baudrate=int(getattr(config, "safety_baud", 9600)),
            parity=getattr(config, "safety_parity", "N"),
            stopbits=int(getattr(config, "safety_stopbits", 1)),
            bytesize=int(getattr(config, "safety_bytesize", 8)),
            unit_id=int(getattr(config, "safety_unit_id", 1)),
            kind=getattr(config, "safety_kind", COIL),
            address=int(getattr(config, "safety_address", 0)),
        )
    if transport != TCP:
        raise ValueError(f"unknown safety_transport {transport!r}")
    return ModbusBackend(
        host=getattr(config, "safety_host", ""),
        port=int(getattr(config, "safety_port", 502)),
        unit_id=int(getattr(config, "safety_unit_id", 1)),
        kind=getattr(config, "safety_kind", COIL),
        address=int(getattr(config, "safety_address", 0)),
    )


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
        self._destination = destination(config)

        # Enabled when there is somewhere to write: a configured far end, or
        # an injected backend, which is how the tests reach it with no
        # hardware and no network.
        self.enabled = backend is not None or bool(self._destination)

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
            self._backend = create_backend(self._config)
        log.info(
            "Safety output: %s",
            getattr(self._backend, "describe", lambda: type(self._backend).__name__)(),
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
