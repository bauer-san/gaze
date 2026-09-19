"""The read-only machine reader.

Two halves, for two different kinds of mistake. Decoding is exercised against
hand-built register blocks, because a bit index off by one produces a status
that is wrong and entirely plausible. The reading itself runs against a real
loopback Modbus server, because the address offset is the other way to be
confidently wrong: every read succeeds and returns the neighbouring register.
"""

import socket
import threading
import time
from types import SimpleNamespace

import pytest

from gaze_monitor.machine import (
    ADDR_CURRENT,
    ADDR_FAULT,
    ADDR_OUTPUT_HZ,
    ADDR_STATUS,
    BLOCK_COUNT,
    BLOCK_START,
    MachineReader,
    MachineStatus,
)


def _cfg(**overrides):
    values = dict(
        machine_port="",
        machine_baud=9600,
        machine_parity="N",
        machine_stopbits=1,
        machine_bytesize=8,
        machine_unit_id=1,
        machine_address_offset=-1,
        machine_interval=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _block(current=0, hz=0, status=0, fault=0):
    """A register block as the drive would return it."""
    regs = [0] * BLOCK_COUNT
    regs[ADDR_CURRENT - BLOCK_START] = current
    regs[ADDR_OUTPUT_HZ - BLOCK_START] = hz
    regs[ADDR_STATUS - BLOCK_START] = status
    regs[ADDR_FAULT - BLOCK_START] = fault
    return regs


# -- decoding --------------------------------------------------------------


def test_scaling_matches_the_documented_units():
    """Current is tenths of an amp, frequency is hundredths of a hertz.

    Both are plausible as raw integers, which is exactly why they are worth
    pinning: 3000 reads fine as a frequency until you notice it is 30 Hz.
    """
    status = MachineStatus.decode(_block(current=123, hz=3000), 1.0)
    assert status.current_a == pytest.approx(12.3)
    assert status.output_hz == pytest.approx(30.0)


@pytest.mark.parametrize(
    "bit, attribute",
    [
        (0, "stopped"),
        (1, "running"),
        (2, "running"),
        (3, "faulted"),
        (10, "brake_released"),
    ],
)
def test_each_status_bit_lands_on_its_own_flag(bit, attribute):
    status = MachineStatus.decode(_block(status=1 << bit), 1.0)
    assert getattr(status, attribute) is True


def test_running_covers_both_directions_and_nothing_else():
    assert MachineStatus.decode(_block(status=1 << 1), 1.0).running is True
    assert MachineStatus.decode(_block(status=1 << 2), 1.0).running is True
    # Bit 0 is "stopped"; a drive asserting it is not running.
    assert MachineStatus.decode(_block(status=1 << 0), 1.0).running is False


def test_the_raw_words_are_kept():
    """So an unexpected bit can be chased from a dashboard rather than from
    another trip to site."""
    status = MachineStatus.decode(_block(status=0xABCD, fault=0x1234), 1.0)
    assert status.raw_status == 0xABCD
    assert status.raw_fault == 0x1234


# -- staying out of the way ------------------------------------------------


def test_no_port_means_disabled():
    reader = MachineReader(_cfg())
    assert reader.enabled is False
    assert reader.start() is False
    assert reader.latest() is None
    assert reader.poll(1.0) is None
    reader.stop()  # must not raise


def test_age_is_zero_before_the_first_read():
    assert MachineReader(_cfg()).age(100.0) == 0.0


# -- against a real server -------------------------------------------------

from pymodbus.client import ModbusTcpClient  # noqa: E402
from pymodbus.datastore import (  # noqa: E402
    ModbusDeviceContext,
    ModbusSequentialDataBlock,
    ModbusServerContext,
)
from pymodbus.server import StartTcpServer  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def drive():
    """A Modbus server standing in for the drive.

    Holding registers are seeded so that each one holds its own address. That
    makes an address-offset mistake fail loudly instead of returning zeros:
    the decoded value names the register it actually came from.
    """
    port = _free_port()
    # The data block shim subtracts one internally, so it starts at 1.
    context = ModbusServerContext(
        devices=ModbusDeviceContext(
            hr=ModbusSequentialDataBlock(1, list(range(64))),
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


def _reader(port, **overrides):
    client = ModbusTcpClient("127.0.0.1", port=port)
    return MachineReader(_cfg(**overrides), client=client)


def test_an_injected_client_enables_the_reader(drive):
    reader = _reader(drive)
    try:
        assert reader.enabled is True
    finally:
        reader.stop()


def test_the_address_offset_moves_the_whole_block(drive):
    """Each register holds its own address, so the offset is visible in the
    data rather than having to be taken on trust."""
    for offset in (0, -1):
        reader = _reader(drive, machine_address_offset=offset)
        try:
            status = reader.poll(5.0)
            assert status is not None
            # current is the first register of the block, scaled by 0.1
            assert status.current_a == pytest.approx((BLOCK_START + offset) * 0.1)
        finally:
            reader.stop()


def test_a_good_read_is_counted_and_retained(drive):
    reader = _reader(drive)
    try:
        assert reader.latest() is None
        status = reader.poll(5.0)
        assert status is not None
        assert reader.reads == 1
        assert reader.read_errors == 0
        assert reader.latest() == status
        assert reader.age(7.0) == pytest.approx(2.0)
    finally:
        reader.stop()


def test_a_failed_read_is_counted_and_keeps_the_last_good_one():
    """A drive that stops answering must not erase what it last said.

    The age is what tells a reader the value is old; blanking it would look
    identical to a machine that is genuinely stopped and idle.
    """
    reader = _reader(_free_port())
    try:
        reader._latest = MachineStatus.decode(_block(current=50), 1.0)
        assert reader.poll(2.0) is None
        assert reader.read_errors == 1
        assert reader.reads == 0
        assert reader.latest().current_a == pytest.approx(5.0)
    finally:
        reader.stop()


def test_the_reader_has_no_write_path():
    """Structural, and deliberate: this module must not be able to command
    the machine, whatever anyone later decides would be convenient."""
    forbidden = [
        name
        for name in dir(MachineReader)
        if "write" in name.lower() or "command" in name.lower()
    ]
    assert forbidden == []
