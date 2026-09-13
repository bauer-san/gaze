"""Terminal operator UI, for commissioning and running over ssh.

The window UI in :mod:`gaze_monitor.ui` needs a display the operator can see,
which an installed unit reached over ssh does not have. This module runs the
same two phases -- calibration prompts, then a live status line -- through a
terminal instead.

The four corners the operator marks are the corners of the **physical** area
to watch: the machine, not any screen. The window UI draws circles at the
screen corners, but those are a prompt and a progress indicator, not a target
-- so cueing the same procedure from a terminal loses nothing.

Nothing here imports OpenCV. This is the half of the UI that has to work on a
box with no vision stack, and the tests exercise it that way.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import tty

from .attention import STATE_LABELS, AttentionState

log = logging.getLogger(__name__)

# Erase the whole line and return to column 0. Writing the status line is
# always a full repaint: the previous one may have been longer.
CLEAR_LINE = "\x1b[2K\r"
RESET = "\x1b[0m"

# Colour is a secondary cue only. Every state prints its name, so the display
# still reads correctly when colour is off, piped, or invisible to the reader.
STATE_COLORS = {
    AttentionState.ATTENTIVE: "\x1b[32m",
    AttentionState.WARNING: "\x1b[33m",
    AttentionState.ALERT: "\x1b[31;1m",
    AttentionState.FAULT: "\x1b[35m",
}

# Enter is accepted alongside 'c' because it is what a person reaches for when
# a terminal asks them to confirm something.
COMMAND_KEYS = {
    "c": "collect",
    "\r": "collect",
    "\n": "collect",
    " ": "collect",
    "r": "recalibrate",
    "q": "quit",
    "\x1b": "quit",  # Esc
    "\x03": "quit",  # Ctrl-C, which cbreak mode leaves to us
}


def calibration_line(calibrator, message: str = "") -> str:
    """The prompt for the corner the operator should be fixating now."""
    index = min(calibrator.corner_index + 1, 4)
    name = calibrator.current_corner_name

    if calibrator.is_collecting:
        body = f"corner {index}/4 {name}: hold still"
    else:
        body = f"corner {index}/4 {name}: look at it on the machine, then press 'c'"

    line = f"CALIBRATION  {body}"
    if message:
        line = f"{line}   [{message}]"
    return line


def monitoring_line(status, fps: float) -> str:
    """One line describing the monitor's verdict for this frame."""
    label = STATE_LABELS[status.state]
    position = status.position
    where = (
        f"gaze=({position[0]:+.3f},{position[1]:+.3f})"
        if position is not None
        else "gaze=(  --  ,  --  )"
    )
    distance = f"{status.z_m:5.2f}m" if status.z_m > 0.0 else "  --  "
    return (
        f"[{label:<14}] away {status.away_seconds:5.1f}s  "
        f"tracked={'yes' if status.tracked else 'no':<3}  "
        f"zone={'in' if status.in_zone else 'out':<3}  "
        f"dist={distance}  {where}  {fps:5.1f} fps"
    )


class _StatusAwareHandler(logging.StreamHandler):
    """A log handler that does not tear the status line in half.

    The status line is repainted in place on the same terminal the log writes
    to. Without this, every log record lands in the middle of it.
    """

    def __init__(self, ui: TerminalUI, stream) -> None:
        super().__init__(stream)
        self._ui = ui

    def emit(self, record: logging.LogRecord) -> None:
        self._ui.clear()
        super().emit(record)
        self._ui.repaint()


class TerminalUI:
    """Calibration prompts and a live status line, driven from a terminal.

    ``poll`` is non-blocking: the capture loop must keep running while the
    operator decides, or the camera backs up and the fixation it collects is
    stale.
    """

    can_calibrate = True

    def __init__(self, stream=None, stdin=None, use_color: bool | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._stdin = stdin if stdin is not None else sys.stdin
        self._use_color = use_color
        self._last_line = ""
        self._saved_termios = None
        self._fd = None
        self._saved_handlers: list[logging.Handler] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not self._stdin.isatty():
            raise RuntimeError(
                "Terminal UI needs a terminal: stdin is not a tty. Run it from "
                "an interactive shell, or with 'docker run -it' / compose's "
                "tty: true and stdin_open: true."
            )
        if self._use_color is None:
            self._use_color = bool(
                self._stream.isatty() and not os.environ.get("NO_COLOR")
            )

        self._fd = self._stdin.fileno()
        self._saved_termios = termios.tcgetattr(self._fd)
        # cbreak, not raw: it leaves ISIG alone, so Ctrl-C still signals if
        # the loop is wedged somewhere that never reaches poll().
        tty.setcbreak(self._fd)
        self._install_log_handler()

        self._write(
            "Keys: 'c' or Enter to mark a corner, 'r' to recalibrate, 'q' to quit.\n"
        )

    def stop(self) -> None:
        self.clear()
        self._restore_log_handler()
        if self._saved_termios is not None and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_termios)
            self._saved_termios = None
        self._write("\n")

    # -- input -------------------------------------------------------------

    def poll(self) -> str | None:
        """Return a pending command, or None. Never blocks."""
        if self._fd is None:
            return None
        command = None
        # Drain the buffer: a held key must not queue up four corner marks.
        while select.select([self._fd], [], [], 0)[0]:
            try:
                data = os.read(self._fd, 1024)
            except (BlockingIOError, InterruptedError):
                break
            if not data:
                break
            for char in data.decode("utf-8", "ignore"):
                command = COMMAND_KEYS.get(char, command)
        return command

    def closed(self) -> bool:
        return False

    # -- output ------------------------------------------------------------

    def render(
        self,
        *,
        calibrating: bool,
        calibrator,
        status,
        fps: float,
        message: str = "",
        message_visible: bool = False,
        frame=None,
    ) -> None:
        del frame  # no preview in a terminal
        if calibrating:
            line = calibration_line(calibrator, message if message_visible else "")
        else:
            line = monitoring_line(status, fps)
            if self._use_color:
                colour = STATE_COLORS.get(status.state)
                if colour:
                    line = f"{colour}{line}{RESET}"
            if message and message_visible:
                line = f"{line}   [{message}]"
        self._paint(line)

    def clear(self) -> None:
        self._write(CLEAR_LINE)

    def repaint(self) -> None:
        if self._last_line:
            self._write(self._last_line)

    # -- internals ---------------------------------------------------------

    def _paint(self, line: str) -> None:
        self._last_line = line
        self._write(CLEAR_LINE + line)

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except (BrokenPipeError, ValueError):
            # The far end of an ssh session went away. The monitor itself is
            # still doing its job; losing the display is not a reason to die.
            pass

    def _install_log_handler(self) -> None:
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        formatter = self._saved_handlers[0].formatter if self._saved_handlers else None
        handler = _StatusAwareHandler(self, self._stream)
        if formatter is not None:
            handler.setFormatter(formatter)
        root.handlers = [handler]

    def _restore_log_handler(self) -> None:
        if self._saved_handlers is not None:
            logging.getLogger().handlers = self._saved_handlers
            self._saved_handlers = None
