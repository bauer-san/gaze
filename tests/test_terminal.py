"""The terminal UI: prompt text, status line, and key handling.

This module must work on a box with no vision stack, so nothing here imports
OpenCV -- which is also what lets these run on the x86_64 CI runners.
"""

import io
import logging
import os
import pty
import select
import time

import pytest

from gaze_monitor.attention import AttentionState, AttentionStatus
from gaze_monitor.calibration import GazeCalibrator
from gaze_monitor.terminal import (
    COMMAND_KEYS,
    TerminalUI,
    calibration_line,
    monitoring_line,
)

# -- prompt text -----------------------------------------------------------


def test_calibration_line_names_the_corner_and_the_count():
    line = calibration_line(GazeCalibrator())
    assert "Top-Left" in line
    assert "1/4" in line
    # The operator marks the machine, not the screen: say so in the prompt,
    # because a terminal has no circle to point at.
    assert "machine" in line


def test_calibration_line_advances_with_collected_corners():
    calibrator = GazeCalibrator(sample_duration=0.01, min_samples=1)
    calibrator.start_collection(0.0)
    for _ in range(5):
        calibrator.collect(0.0, 0.1, 0.1, 1.0)
    accepted, _ = calibrator.validate_and_save()
    assert accepted
    line = calibration_line(calibrator)
    assert "2/4" in line
    assert "Top-Right" in line


def test_calibration_line_says_hold_still_while_collecting():
    calibrator = GazeCalibrator()
    calibrator.start_collection(0.0)
    assert "hold still" in calibration_line(calibrator)
    assert "press" not in calibration_line(calibrator)


def test_calibration_line_appends_a_rejection_message():
    line = calibration_line(GazeCalibrator(), "Too much movement")
    assert "Too much movement" in line


# -- status line -----------------------------------------------------------


def _status(**kwargs):
    defaults = dict(
        state=AttentionState.ATTENTIVE,
        away_seconds=0.0,
        in_zone=True,
        tracked=True,
        position=(0.0, 0.0),
    )
    defaults.update(kwargs)
    return AttentionStatus(**defaults)


@pytest.mark.parametrize(
    "state,label",
    [
        (AttentionState.ATTENTIVE, "WATCHING BLADE"),
        (AttentionState.WARNING, "EYES OFF BLADE"),
        (AttentionState.ALERT, "ATTENTION LOST"),
        (AttentionState.FAULT, "SENSOR FAULT"),
    ],
)
def test_monitoring_line_names_every_state(state, label):
    """Colour is a secondary cue; the words have to carry the meaning."""
    assert label in monitoring_line(_status(state=state), 30.0)


def test_monitoring_line_reports_the_numbers():
    line = monitoring_line(
        _status(
            state=AttentionState.ALERT,
            away_seconds=3.25,
            in_zone=False,
            tracked=True,
            position=(0.125, -0.5),
        ),
        24.5,
    )
    assert "3.2s" in line
    assert "24.5 fps" in line
    assert "+0.125" in line and "-0.500" in line
    assert "zone=out" in line
    assert "tracked=yes" in line


def test_monitoring_line_handles_an_unmeasured_gaze():
    """position is None whenever the face is not tracked; do not crash."""
    line = monitoring_line(_status(tracked=False, position=None), 0.0)
    assert "tracked=no" in line
    assert "--" in line


# -- terminal handling -----------------------------------------------------


def test_start_refuses_a_non_tty():
    ui = TerminalUI(stream=io.StringIO(), stdin=io.StringIO())
    with pytest.raises(RuntimeError, match="not a tty"):
        ui.start()


@pytest.mark.parametrize(
    "key,command",
    [
        ("c", "collect"),
        ("\r", "collect"),
        (" ", "collect"),
        ("r", "recalibrate"),
        ("q", "quit"),
        ("\x1b", "quit"),
    ],
)
def test_command_keys_cover_the_documented_bindings(key, command):
    assert COMMAND_KEYS[key] == command


class _Pty:
    """A real pty, so isatty() and termios behave as they do over ssh."""

    def __enter__(self):
        self.master, slave = pty.openpty()
        self.stdin = os.fdopen(slave, "r")
        self.out = io.StringIO()
        self.ui = TerminalUI(stream=self.out, stdin=self.stdin, use_color=False)
        return self

    def send(self, text: str) -> None:
        os.write(self.master, text.encode())
        # The write has to land before a zero-timeout select will see it.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if select.select([self.stdin.fileno()], [], [], 0.01)[0]:
                return
        raise AssertionError("pty write never became readable")

    def __exit__(self, *exc):
        try:
            self.ui.stop()
        finally:
            self.stdin.close()
            os.close(self.master)


def test_poll_returns_the_pressed_command():
    with _Pty() as t:
        t.ui.start()
        t.send("c")
        assert t.ui.poll() == "collect"


def test_poll_is_non_blocking_when_nothing_is_pressed():
    with _Pty() as t:
        t.ui.start()
        assert t.ui.poll() is None


def test_poll_collapses_a_held_key():
    """A repeating key must not queue up four corner marks in one press."""
    with _Pty() as t:
        t.ui.start()
        t.send("ccccc")
        assert t.ui.poll() == "collect"
        assert t.ui.poll() is None


def test_render_paints_a_status_line():
    with _Pty() as t:
        t.ui.start()
        t.ui.render(
            calibrating=False,
            calibrator=GazeCalibrator(),
            status=_status(state=AttentionState.WARNING),
            fps=12.0,
        )
        assert "EYES OFF BLADE" in t.out.getvalue()


def test_render_prompts_during_calibration():
    with _Pty() as t:
        t.ui.start()
        t.ui.render(
            calibrating=True,
            calibrator=GazeCalibrator(),
            status=_status(),
            fps=12.0,
        )
        assert "Top-Left" in t.out.getvalue()


def test_log_handlers_are_restored_on_stop():
    """The UI takes over the root logger so records do not tear the status
    line in half. It has to give it back."""
    root = logging.getLogger()
    before = list(root.handlers)
    with _Pty() as t:
        t.ui.start()
        assert root.handlers != before
    assert root.handlers == before


def test_terminal_modes_are_restored_on_stop():
    import termios

    with _Pty() as t:
        fd = t.stdin.fileno()
        before = termios.tcgetattr(fd)
        t.ui.start()
        assert termios.tcgetattr(fd) != before
        # stop() is also called again by __exit__; it must tolerate that.
        t.ui.stop()
        assert termios.tcgetattr(fd) == before
