"""Starting a camera that may have wedged.

The sequencing lives apart from the hardware precisely so it can be tested
here, with no camera, no librealsense and no OpenCV. What is worth pinning is
not that a reset happens but that it happens once, only when it is needed, and
that each way of failing is reported as itself -- the caller raises different
advice for "it never came back" than for "it came back and is still dead".
"""

import logging

import pytest

from gaze_monitor.recovery import (
    OK,
    RECOVERED,
    RESET_FAILED,
    STILL_DEAD,
    start_with_recovery,
)


class _Camera:
    """Records the calls, and produces frames after ``frames_after`` opens."""

    def __init__(self, frames_after=1, reset_works=True):
        self.frames_after = frames_after
        self.reset_works = reset_works
        self.opens = 0
        self.probes = 0
        self.resets = 0

    def open(self):
        self.opens += 1

    def first_frame(self):
        self.probes += 1
        return self.opens >= self.frames_after

    def reset(self):
        self.resets += 1
        return self.reset_works

    def run(self):
        return start_with_recovery(self.open, self.first_frame, self.reset)


# -- the happy path costs nothing ------------------------------------------


def test_a_healthy_camera_is_never_reset():
    """The reason this is on failure rather than on every start. Nobody should
    pay several seconds of USB re-enumeration for a camera that is fine."""
    cam = _Camera(frames_after=1)
    assert cam.run() == OK
    assert cam.opens == 1
    assert cam.resets == 0


# -- the wedge, and getting out of it ---------------------------------------


def test_a_wedged_camera_is_reset_and_reopened():
    """Enumerates, accepts a pipeline, produces nothing: reset and try again."""
    cam = _Camera(frames_after=2)
    assert cam.run() == RECOVERED
    assert cam.resets == 1
    assert cam.opens == 2, "the pipeline must be rebuilt after the reset"


def test_the_reset_is_attempted_once_and_not_in_a_loop():
    """A camera that is genuinely dead must fail, not retry forever. The
    services restart unless stopped, so a loop here would re-enumerate a USB
    device every few seconds indefinitely."""
    cam = _Camera(frames_after=99)
    assert cam.run() == STILL_DEAD
    assert cam.resets == 1
    assert cam.opens == 2


def test_a_reset_that_fails_stops_immediately():
    """No point reopening a pipeline on a device that never came back."""
    cam = _Camera(frames_after=99, reset_works=False)
    assert cam.run() == RESET_FAILED
    assert cam.resets == 1
    assert cam.opens == 1, "the pipeline was reopened despite the failed reset"


def test_the_four_outcomes_are_distinct():
    """The caller raises different advice for each, so they must not collide."""
    assert len({OK, RECOVERED, RESET_FAILED, STILL_DEAD}) == 4


# -- what it says -----------------------------------------------------------


def test_a_reset_is_announced_and_a_recovery_is_too(caplog):
    """A camera needing a reset is a fact worth surfacing. Recovering silently
    would bury it, and the pattern over weeks is the useful signal."""
    caplog.set_level(logging.INFO)
    assert _Camera(frames_after=2).run() == RECOVERED
    assert "Resetting it" in caplog.text
    assert "recovered after a hardware reset" in caplog.text


def test_a_healthy_start_says_nothing(caplog):
    caplog.set_level(logging.INFO)
    assert _Camera(frames_after=1).run() == OK
    assert caplog.text == ""


# -- failures that are not this module's business ---------------------------


def test_an_open_that_raises_is_left_to_the_caller():
    """A pipeline that will not start at all is a different fault -- no
    camera, wrong mode, device held elsewhere -- and resetting would neither
    diagnose nor fix it."""

    def boom():
        raise RuntimeError("no device")

    with pytest.raises(RuntimeError, match="no device"):
        start_with_recovery(boom, lambda: True, lambda: True)


def test_a_reopen_that_raises_after_a_reset_also_propagates():
    calls = {"n": 0}

    def open_once_then_fail():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("vanished mid-reset")

    with pytest.raises(RuntimeError, match="vanished"):
        start_with_recovery(open_once_then_fail, lambda: False, lambda: True)
