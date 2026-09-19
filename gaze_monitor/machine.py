"""Read-only machine state, polled from a variable-frequency drive.

This module can only read. It has no write path at all, and that is not an
oversight -- the permit output in :mod:`gaze_monitor.safety` is a separate
object on separate hardware, and the two are kept apart so that a bug here
cannot reach the machine. The worst this module can do is stop reporting.

What it is for is context. Knowing that the saw is actually running, drawing
current, and has not tripped turns a bare attention state into something that
can be reasoned about afterwards: an operator looking away from a stopped
machine is not the same event as one looking away mid-cut, and only the drive
knows which it was.

Addressing is the part that bites. Several drives, LS Electric among them,
document a communication address one higher than the number that goes on the
wire, so ``machine_address_offset`` is -1 for those. Get it wrong and every
read succeeds and returns the neighbouring register, which is far worse than
failing: the numbers look plausible.

Like the safety module this is pure Python -- no OpenCV, no numpy -- so the
tests can drive it against a loopback Modbus server with no hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whether the import lands
    from pymodbus.client import ModbusSerialClient

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

# Documented communication addresses, contiguous so one transaction covers
# the lot. Anything the reader needs has to live in this span or it costs
# another round trip on a 9600 baud bus.
ADDR_CURRENT = 0x0009  # 0.1 A
ADDR_OUTPUT_HZ = 0x000A  # 0.01 Hz
ADDR_STATUS = 0x000E  # bitfield, see below
ADDR_FAULT = 0x000F  # bitfield
BLOCK_START = ADDR_CURRENT
BLOCK_COUNT = ADDR_FAULT - ADDR_CURRENT + 1  # 0x0009..0x000F inclusive

# Operation status bits (0x000E).
BIT_STOPPED = 0
BIT_RUN_FORWARD = 1
BIT_RUN_REVERSE = 2
BIT_FAULT = 3
BIT_BRAKE_RELEASED = 10


@dataclass(frozen=True)
class MachineStatus:
    """One good read. ``raw_*`` are kept so an unexpected bit can be chased
    without a code change and another trip to site."""

    current_a: float
    output_hz: float
    running: bool
    stopped: bool
    faulted: bool
    brake_released: bool
    raw_status: int
    raw_fault: int
    read_at: float

    @classmethod
    def decode(cls, registers: list[int], read_at: float) -> MachineStatus:
        def at(address: int) -> int:
            return registers[address - BLOCK_START]

        status = at(ADDR_STATUS)
        return cls(
            current_a=at(ADDR_CURRENT) * 0.1,
            output_hz=at(ADDR_OUTPUT_HZ) * 0.01,
            running=bool(status & (1 << BIT_RUN_FORWARD))
            or bool(status & (1 << BIT_RUN_REVERSE)),
            stopped=bool(status & (1 << BIT_STOPPED)),
            faulted=bool(status & (1 << BIT_FAULT)),
            brake_released=bool(status & (1 << BIT_BRAKE_RELEASED)),
            raw_status=status,
            raw_fault=at(ADDR_FAULT),
            read_at=read_at,
        )


class MachineReader:
    """Polls the drive on a background thread. Disabled unless configured.

    ``poll`` is the whole of it and takes the time explicitly, so the tests
    can step it without threads or sleeps -- the same shape as the safety
    output, for the same reason.
    """

    def __init__(self, config, client=None) -> None:
        self.port = getattr(config, "machine_port", "") or ""
        self.unit_id = int(getattr(config, "machine_unit_id", 1))
        self.offset = int(getattr(config, "machine_address_offset", -1))
        self.interval = float(getattr(config, "machine_interval", 1.0))
        self.enabled = client is not None or bool(self.port)

        self._config = config
        self._client = client
        self._lock = threading.Lock()
        self._latest: MachineStatus | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.reads = 0
        self.read_errors = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Open the port and begin polling. False means it stayed disabled."""
        if not self.enabled:
            return False
        if self._client is None:
            if not AVAILABLE:  # pragma: no cover - depends on the install
                raise RuntimeError(
                    "pymodbus is required to read machine state: "
                    "pip install -r requirements.txt"
                )
            self._client = ModbusSerialClient(
                self.port,
                baudrate=int(getattr(self._config, "machine_baud", 9600)),
                parity=getattr(self._config, "machine_parity", "N"),
                stopbits=int(getattr(self._config, "machine_stopbits", 1)),
                bytesize=int(getattr(self._config, "machine_bytesize", 8)),
                timeout=1.0,
            )
        log.info(
            "Machine reader: %s unit %d, registers 0x%04X+%d, address offset %+d",
            self.port or type(self._client).__name__,
            self.unit_id,
            BLOCK_START,
            BLOCK_COUNT,
            self.offset,
        )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="gaze-machine", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - closing must not raise
                pass

    # -- reading -----------------------------------------------------------

    def poll(self, now: float) -> MachineStatus | None:
        """One read. Returns the decoded status, or None if it failed."""
        if not self.enabled or self._client is None:
            return None
        try:
            if not self._client.connected:
                self._client.connect()
            result = self._client.read_holding_registers(
                BLOCK_START + self.offset,
                count=BLOCK_COUNT,
                device_id=self.unit_id,
            )
            if result is None or (hasattr(result, "isError") and result.isError()):
                raise OSError(f"Modbus read rejected: {result!r}")
            status = MachineStatus.decode(list(result.registers), now)
        except Exception as exc:
            self.read_errors += 1
            if self.read_errors == 1 or self.read_errors % 100 == 0:
                log.warning(
                    "Machine read failed (%d so far): %s", self.read_errors, exc
                )
            return None
        self.reads += 1
        with self._lock:
            self._latest = status
        return status

    def latest(self) -> MachineStatus | None:
        """The last good read, or None if there has never been one."""
        with self._lock:
            return self._latest

    def age(self, now: float) -> float:
        """Seconds since the last good read. 0 if there has never been one."""
        with self._lock:
            latest = self._latest
        return 0.0 if latest is None else max(0.0, now - latest.read_at)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll(time.monotonic())
            self._stop.wait(self.interval)
