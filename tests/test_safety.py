"""The safety output.

Most of this drives the state machine through a recording backend, so it can
be stepped without threads, sockets or sleeps -- ``tick`` takes the time
explicitly for exactly that reason.

The last section talks to the real Modbus/TCP backend over a loopback server.
Two rounds of dependency trouble in this project have established that an
install which resolves is not an install that works, and the same goes for a
protocol: the only way to know the writes land is to watch them land.
"""

import socket
import threading
import time
from types import SimpleNamespace

import pytest

from gaze_monitor.attention import AttentionState, AttentionStatus
from gaze_monitor.safety import Backend, SafetyOutput


def _cfg(**overrides):
    values = dict(
        brake_after=3.0,
        brake_on_fault=False,
        safety_host="",
        safety_port=502,
        safety_unit_id=1,
        safety_kind="coil",
        safety_address=0,
        safety_interval=0.1,
        safety_stale_after=0.5,
        safety_auto_arm=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _status(away=0.0, state=AttentionState.ATTENTIVE):
    return AttentionStatus(state=state, away_seconds=away, tracked=True)


class _Recorder(Backend):
    """A backend that remembers, and optionally refuses."""

    def __init__(self, fail: bool = False) -> None:
        self.writes: list[bool] = []
        self.fail = fail
        self.closed = False

    def write(self, permitted: bool) -> None:
        if self.fail:
            raise OSError("no route to host")
        self.writes.append(permitted)

    def close(self) -> None:
        self.closed = True


def _armed(**overrides):
    rec = _Recorder()
    out = SafetyOutput(_cfg(**overrides), backend=rec)
    out.arm()
    return out, rec


# -- staying out of the way ------------------------------------------------


def test_no_host_and_no_backend_means_disabled():
    out = SafetyOutput(_cfg())
    assert out.enabled is False
    assert out.start() is False


def test_a_disabled_output_ignores_everything():
    """The run loop calls these unconditionally, and the default config has
    no safety output at all."""
    out = SafetyOutput(_cfg())
    out.observe(_status(), 0.0)
    out.tick(0.0)
    out.arm()
    out.stop()
    assert out.last_written is None


# -- the permit ------------------------------------------------------------


def test_refuses_a_permit_until_armed():
    """Coming up permissive after a crash is the thing this must never do."""
    rec = _Recorder()
    out = SafetyOutput(_cfg(), backend=rec)
    out.observe(_status(away=0.0), 10.0)
    out.tick(10.0)
    assert rec.writes == [False]
    assert out.armed is False


def test_permits_once_armed_and_attentive():
    out, rec = _armed()
    out.observe(_status(away=0.0), 10.0)
    out.tick(10.0)
    assert rec.writes == [True]


def test_demands_stop_past_brake_after():
    out, rec = _armed(brake_after=3.0)
    out.observe(_status(away=2.9), 10.0)
    out.tick(10.0)
    out.observe(_status(away=3.0), 10.1)
    out.tick(10.1)
    assert rec.writes == [True, False]


def test_brake_after_zero_never_demands_a_stop():
    """0 disables braking, so the output becomes a link test only."""
    out, rec = _armed(brake_after=0.0)
    out.observe(_status(away=999.0), 10.0)
    out.tick(10.0)
    assert rec.writes == [True]


def test_a_camera_fault_does_not_brake_by_default():
    """A fouled lens should be loud, not stopping: the machine's own guards
    are unaffected by this system failing."""
    out, rec = _armed(brake_on_fault=False)
    out.observe(_status(away=0.0, state=AttentionState.FAULT), 10.0)
    out.tick(10.0)
    assert rec.writes == [True]


def test_a_camera_fault_brakes_when_asked():
    out, rec = _armed(brake_on_fault=True)
    out.observe(_status(away=0.0, state=AttentionState.FAULT), 10.0)
    out.tick(10.0)
    assert rec.writes == [False]


# -- silence, which is different from refusal ------------------------------


def test_nothing_observed_yet_is_silence_not_a_permit():
    out, rec = _armed()
    out.tick(10.0)
    assert rec.writes == []
    assert out.silent is True


def test_a_stale_vision_result_stops_renewals():
    """The far end's own timeout is left to act, because at this point the
    monitor does not know what is happening and should not claim otherwise."""
    out, rec = _armed(safety_stale_after=0.5)
    out.observe(_status(away=0.0), 10.0)
    out.tick(10.0)
    assert rec.writes == [True]

    out.tick(10.6)  # nothing new observed for 0.6s
    assert rec.writes == [True]
    assert out.silent is True


def test_renewals_resume_when_the_vision_loop_recovers():
    out, rec = _armed()
    out.observe(_status(), 10.0)
    out.tick(10.6)
    assert out.silent is True
    out.observe(_status(), 11.0)
    out.tick(11.0)
    assert out.silent is False
    assert rec.writes == [True]


def test_staleness_reports_the_age_of_the_last_observation():
    out, _ = _armed()
    assert out.staleness(10.0) == 0.0
    out.observe(_status(), 10.0)
    assert out.staleness(10.25) == pytest.approx(0.25)


# -- failures --------------------------------------------------------------


def test_a_write_failure_is_counted_and_does_not_propagate():
    """Losing the link must not take down the monitor: the operator display
    and the log are still worth having."""
    out = SafetyOutput(_cfg(), backend=_Recorder(fail=True))
    out.arm()
    out.observe(_status(), 10.0)
    out.tick(10.0)
    assert out.write_errors == 1
    assert out.writes == 0


def test_stop_refuses_the_permit_and_closes_the_backend():
    out, rec = _armed()
    out.observe(_status(), 10.0)
    out.tick(10.0)
    out.stop()
    assert rec.writes[-1] is False
    assert rec.closed is True
    assert out.last_written is False


def test_auto_arm_skips_the_start_up_check():
    rec = _Recorder()
    out = SafetyOutput(_cfg(safety_auto_arm=True), backend=rec)
    assert out.armed is True
    out.observe(_status(), 10.0)
    out.tick(10.0)
    assert rec.writes == [True]


# -- the real thing, against a loopback server -----------------------------

pymodbus = pytest.importorskip("pymodbus")

from pymodbus.client import ModbusTcpClient  # noqa: E402
from pymodbus.datastore import (  # noqa: E402
    ModbusDeviceContext,
    ModbusSequentialDataBlock,
    ModbusServerContext,
)
from pymodbus.server import StartTcpServer  # noqa: E402

from gaze_monitor.safety import ModbusBackend  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def loopback():
    """A Modbus/TCP server on localhost.

    This is the same arrangement as pointing safety_host at 127.0.0.1 to
    rehearse an integration before any machine is available.
    """
    port = _free_port()
    # 1, not 0: the ModbusSequentialDataBlock shim subtracts one internally,
    # so a starting address of 0 underflows and is rejected.
    context = ModbusServerContext(
        devices=ModbusDeviceContext(
            co=ModbusSequentialDataBlock(1, [0] * 16),
            hr=ModbusSequentialDataBlock(1, [0] * 16),
        ),
        single=True,
    )
    threading.Thread(
        target=StartTcpServer,
        kwargs={"context": context, "address": ("127.0.0.1", port)},
        daemon=True,
    ).start()

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.05)
    else:  # pragma: no cover - the server never came up
        pytest.fail("loopback Modbus server did not start")
    return port


def _read_coil(port: int, address: int = 0) -> bool:
    client = ModbusTcpClient("127.0.0.1", port=port)
    try:
        assert client.connect()
        result = client.read_coils(address, count=1, device_id=1)
        assert not result.isError(), result
        return bool(result.bits[0])
    finally:
        client.close()


def test_a_permit_reaches_a_coil_over_tcp(loopback):
    backend = ModbusBackend("127.0.0.1", port=loopback, kind="coil", address=0)
    try:
        backend.write(True)
        assert _read_coil(loopback) is True
        backend.write(False)
        assert _read_coil(loopback) is False
    finally:
        backend.close()


def test_a_permit_reaches_a_holding_register_over_tcp(loopback):
    backend = ModbusBackend("127.0.0.1", port=loopback, kind="register", address=2)
    client = ModbusTcpClient("127.0.0.1", port=loopback)
    try:
        backend.write(True)
        assert client.connect()
        assert client.read_holding_registers(2, count=1, device_id=1).registers == [1]
        backend.write(False)
        assert client.read_holding_registers(2, count=1, device_id=1).registers == [0]
    finally:
        client.close()
        backend.close()


def test_the_whole_output_drives_a_real_coil(loopback):
    """End to end: observe, tick, and watch the wire."""
    backend = ModbusBackend("127.0.0.1", port=loopback, kind="coil", address=4)
    out = SafetyOutput(_cfg(brake_after=3.0), backend=backend)
    try:
        out.observe(_status(away=0.0), 10.0)
        out.tick(10.0)
        assert _read_coil(loopback, 4) is False, "unarmed must not permit"

        out.arm()
        out.observe(_status(away=0.0), 11.0)
        out.tick(11.0)
        assert _read_coil(loopback, 4) is True

        out.observe(_status(away=5.0), 12.0)
        out.tick(12.0)
        assert _read_coil(loopback, 4) is False
    finally:
        out.stop()


def test_an_unreachable_far_end_is_counted_not_raised():
    backend = ModbusBackend("127.0.0.1", port=_free_port(), kind="coil", address=0)
    out = SafetyOutput(_cfg(), backend=backend)
    out.arm()
    out.observe(_status(), 10.0)
    out.tick(10.0)
    assert out.write_errors == 1
    assert out.writes == 0


# -- shutdown --------------------------------------------------------------


def test_sigterm_is_routed_through_cleanup():
    """Without this the permit survives `docker stop`.

    Python leaves SIGTERM at its default disposition, which kills the process
    outright and skips every cleanup path, so the last thing written to the
    machine controller stays written.
    """
    import signal

    from gaze_monitor.safety import install_signal_handlers

    previous = signal.getsignal(signal.SIGTERM)
    try:
        installed = install_signal_handlers(SafetyOutput(_cfg()))
        assert "SIGTERM" in installed
        handler = signal.getsignal(signal.SIGTERM)
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_sigusr1_arms_only_when_an_output_is_configured():
    import signal

    from gaze_monitor.safety import install_signal_handlers

    prev_term = signal.getsignal(signal.SIGTERM)
    prev_usr1 = signal.getsignal(signal.SIGUSR1)
    try:
        assert "SIGUSR1" not in install_signal_handlers(SafetyOutput(_cfg()))

        out = SafetyOutput(_cfg(), backend=_Recorder())
        assert "SIGUSR1" in install_signal_handlers(out)
        assert out.armed is False
        signal.getsignal(signal.SIGUSR1)(signal.SIGUSR1, None)
        assert out.armed is True
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGUSR1, prev_usr1)
