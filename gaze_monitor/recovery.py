"""Starting a camera that may have wedged, and unwedging it if so.

A RealSense that was killed without ``pipeline.stop()`` lands in a state that
looks entirely healthy from outside: it enumerates at USB 3.2, librealsense
lists it, ``pipeline.start()`` succeeds, and then ``wait_for_frames`` times
out forever. Nothing in the device's own reporting says anything is wrong.
The remedy is a ``hardware_reset()``, which drops it off the bus and brings
it back clean.

This module holds only the sequencing, not the hardware. The three steps are
passed in as callables, which keeps the decision -- when to reset, how many
times, what counts as failure -- testable with no camera, no librealsense and
no OpenCV, on a runner that has none of them.

The reset fires on failure rather than on every start, deliberately:

* Re-enumeration costs several seconds, and a healthy camera delivers its
  first frame in about thirty milliseconds. Paying that on every start to
  cover a case that should now be rare is the wrong trade.
* The compose services restart unless stopped. A camera that is genuinely
  absent would otherwise re-enumerate a USB device every few seconds forever,
  which is a good way to turn "unplugged" into something worse.
* A camera that needs resetting regularly is a fact worth surfacing. Resetting
  unconditionally would bury it in a startup delay nobody reads.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Outcomes. Strings rather than an enum because the caller maps them onto its
# own error type, and the pure module deliberately raises nothing itself.
OK = "ok"
RECOVERED = "recovered"
RESET_FAILED = "reset-failed"
STILL_DEAD = "still-dead"


def start_with_recovery(open_pipeline, first_frame_arrives, hardware_reset) -> str:
    """Open the camera, and reset it once if no frame appears.

    ``open_pipeline`` may raise; that is a real failure and is left to
    propagate. The other two return booleans. Returns one of the outcome
    constants above -- the caller decides which of them are fatal, because
    only it knows what to say about its own device.
    """
    open_pipeline()
    if first_frame_arrives():
        return OK

    log.warning(
        "The camera started but delivered no frame. That is what a device "
        "wedged by an unclean shutdown looks like: it enumerates, accepts a "
        "pipeline, and never produces anything. Resetting it and retrying "
        "once."
    )

    if not hardware_reset():
        return RESET_FAILED

    open_pipeline()
    if first_frame_arrives():
        log.warning("Camera recovered after a hardware reset.")
        return RECOVERED
    return STILL_DEAD
