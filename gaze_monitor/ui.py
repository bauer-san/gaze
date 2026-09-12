"""The monitor loop: capture -> landmarks -> attention state -> annunciation.

Two phases:

* **Calibration** -- only on first run, or after a factory reset. The operator
  fixates each corner of the attention area in turn and the result is written
  to disk.
* **Monitoring** -- the steady state. Gaze is compared against the stored zone
  and escalated by :class:`~gaze_monitor.attention.AttentionMonitor`.

All drawing happens into an off-screen canvas which is shown exactly once per
frame, at the bottom of the loop. The previous version called imshow in the
middle and then kept drawing, so calibration rejection messages were written
to a buffer that was never displayed -- the operator pressed 'c', saw nothing
happen, and had no idea the fixation had been refused.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import time

import cv2
import numpy as np

from .attention import STATE_LABELS, AttentionMonitor, AttentionState
from .calibration import (
    GazeCalibrator,
    load_calibration,
    save_calibration,
)
from .capture import CameraError, create_source
from .config import MonitorConfig
from .gaze import GazeFilter, gaze_from_landmarks
from .quality import depth_is_blind

log = logging.getLogger(__name__)

WINDOW_NAME = "Operator Attention Monitor"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR. Deliberately distinguishable for the most common colour-vision
# deficiencies: the states differ in brightness and in the text, not by hue
# alone, which is why every state also prints its name.
COLORS = {
    AttentionState.ATTENTIVE: (80, 200, 80),
    AttentionState.WARNING: (0, 190, 255),
    AttentionState.ALERT: (60, 60, 255),
    AttentionState.FAULT: (255, 80, 255),
}
WHITE = (255, 255, 255)
GREY = (140, 140, 140)


# How long a calibration accept/reject message stays on screen.
MESSAGE_SECONDS = 2.0

# Consecutive unusable reads before the camera is declared unhealthy. The
# monitor's own fault_after timer then decides when that becomes a FAULT.
FAILURES_BEFORE_UNHEALTHY = 3

# Pause after a failed read. A backend that fails fast (rather than
# blocking until a timeout, as RealSense does) would otherwise spin the
# loop at 100% CPU for as long as the camera is down -- which is exactly
# when there is other work, like annunciating the fault, to get done.
FAILURE_BACKOFF_SECONDS = 0.02


def _log_transition(old: AttentionState, new: AttentionState, status) -> None:
    level = logging.WARNING if new.is_alarming else logging.INFO
    log.log(
        level,
        "attention %s -> %s (away %.1fs, tracked=%s)",
        old.value,
        new.value,
        status.away_seconds,
        status.tracked,
    )


def _smooth(sample, filter_x: GazeFilter, filter_y: GazeFilter):
    """Apply the EMA to a raw sample, or return None if it has no state yet."""
    dx = filter_x.apply(sample.dx)
    dy = filter_y.apply(sample.dy)
    if dx is None or dy is None:
        return None
    return dataclasses.replace(sample, dx=dx, dy=dy)


# Chrome reserved at the top (status bar) and bottom (zone annotations), so
# nothing the operator needs to read ends up under the bar or off the canvas.
STATUS_BAR_H = 42
TOP_MARGIN = STATUS_BAR_H + 26
BOTTOM_MARGIN = 76

# Half-extent of the out-of-zone crosshair, in pixels.
MARKER_REACH = 28


def _zone_rect(w: int, h: int):
    """The on-screen rectangle standing in for the calibrated attention area."""
    side = int(w * 0.10)
    return side, TOP_MARGIN, w - side, h - BOTTOM_MARGIN


def _corner_targets(w: int, h: int):
    """Calibration target positions, in CORNER_NAMES order.

    Deliberately the corners of _zone_rect: the operator marks the area in the
    same place the monitor later draws it, so the two screens agree.
    """
    x0, y0, x1, y1 = _zone_rect(w, h)
    return [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]


def _draw_calibration(display, calibrator, message, message_visible) -> None:
    h, w = display.shape[:2]
    targets = _corner_targets(w, h)
    idx = min(calibrator.corner_index, len(targets) - 1)

    # Show corners already captured, so the operator can see progress.
    for done in range(calibrator.corner_index):
        cv2.circle(display, targets[done], 14, (80, 200, 80), -1)

    if calibrator.is_collecting:
        cv2.circle(display, targets[idx], 30, (0, 190, 255), 3)
        cv2.circle(display, targets[idx], 12, (0, 190, 255), -1)
        prompt = "Hold still..."
    else:
        cv2.circle(display, targets[idx], 20, (0, 255, 0), -1)
        prompt = f"Look at the {calibrator.current_corner_name} corner, press 'c'"

    cv2.putText(
        display,
        "CALIBRATION - mark the corners of the area to watch",
        (int(w * 0.12), int(h * 0.38)),
        FONT,
        0.8,
        WHITE,
        2,
    )
    cv2.putText(display, prompt, (int(w * 0.12), int(h * 0.46)), FONT, 0.9, WHITE, 2)
    cv2.putText(
        display,
        f"Corner {min(calibrator.corner_index + 1, 4)} of 4",
        (int(w * 0.12), int(h * 0.53)),
        FONT,
        0.7,
        GREY,
        2,
    )

    if message and message_visible:
        colour = (60, 60, 255) if "accepted" not in message else (80, 200, 80)
        cv2.putText(
            display, message, (int(w * 0.12), int(h * 0.62)), FONT, 0.75, colour, 2
        )


def _draw_monitoring(display, status, config: MonitorConfig) -> None:
    h, w = display.shape[:2]
    colour = COLORS[status.state]
    x0, y0, x1, y1 = _zone_rect(w, h)

    cv2.rectangle(display, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(display, "ATTENTION AREA", (x0 + 8, y0 - 10), FONT, 0.5, GREY, 1)

    if status.position is not None:
        u, v = status.position
        if np.isfinite(u) and np.isfinite(v):
            # Clamp into the visible band, not the raw canvas: a gaze far
            # outside the zone would otherwise pin to a screen corner and
            # disappear under the status bar or the debug inset.
            # MARKER_REACH keeps the crosshair arms inside the canvas
            # rather than clipped against the edge.
            sx = int(np.clip(x0 + u * (x1 - x0), MARKER_REACH, w - MARKER_REACH - 1))
            sy = int(
                np.clip(
                    y0 + v * (y1 - y0),
                    TOP_MARGIN - 20,
                    h - BOTTOM_MARGIN + 20,
                )
            )
            if status.in_zone:
                cv2.circle(display, (sx, sy), 18, colour, -1)
                cv2.circle(display, (sx, sy), 18, WHITE, 1)
            else:
                # Hollow crosshair, so "outside the area" is legible as a shape
                # and not only as a colour.
                cv2.circle(display, (sx, sy), 18, colour, 3)
                cv2.line(
                    display, (sx - MARKER_REACH, sy), (sx + MARKER_REACH, sy), colour, 2
                )
                cv2.line(
                    display, (sx, sy - MARKER_REACH), (sx, sy + MARKER_REACH), colour, 2
                )

    if status.state is AttentionState.ALERT:
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), colour, 18)
        text = STATE_LABELS[status.state]
        (tw, _), _ = cv2.getTextSize(text, FONT, 1.8, 4)
        cv2.putText(display, text, ((w - tw) // 2, int(h * 0.5)), FONT, 1.8, colour, 4)
    elif status.state is AttentionState.FAULT:
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), colour, 18)
        text = STATE_LABELS[status.state]
        (tw, _), _ = cv2.getTextSize(text, FONT, 1.4, 3)
        cv2.putText(display, text, ((w - tw) // 2, int(h * 0.5)), FONT, 1.4, colour, 3)
        cv2.putText(
            display,
            "No usable frames from the camera",
            (int(w * 0.2), int(h * 0.58)),
            FONT,
            0.7,
            WHITE,
            2,
        )

    if not status.tracked and status.state is not AttentionState.FAULT:
        cv2.putText(display, "operator not detected", (x0, y1 + 28), FONT, 0.6, GREY, 1)
    if config.debug and status.position is not None:
        u, v = status.position
        cv2.putText(
            display, f"u={u:+.2f} v={v:+.2f}", (x0, y1 + 52), FONT, 0.5, GREY, 1
        )


def _draw_status_bar(display, status, fps: float, calibrating: bool = False) -> None:
    """Always-on health line. A monitor that looks identical when it has
    stopped working is worse than no monitor, so state is shown continuously
    rather than only when alarming."""
    h, w = display.shape[:2]
    if calibrating:
        # There is no zone yet, so claiming the blade is being watched would
        # be a lie on the one screen the operator is looking at.
        colour, label = WHITE, "CALIBRATING"
    else:
        colour, label = COLORS[status.state], STATE_LABELS[status.state]

    cv2.rectangle(display, (0, 0), (w, STATUS_BAR_H), (28, 28, 28), -1)
    cv2.circle(display, (24, 21), 10, colour, -1)
    cv2.putText(display, label, (44, 28), FONT, 0.7, colour, 2)

    right = (
        f"{fps:4.1f} fps"
        if calibrating
        else (f"away {status.away_seconds:4.1f}s   {fps:4.1f} fps")
    )
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.55, 1)
    cv2.putText(display, right, (w - tw - 14, 27), FONT, 0.55, GREY, 1)


def _draw_preview(display, frame) -> None:
    """Small camera inset, debug only."""
    h, w = display.shape[:2]
    pw = max(140, w // 6)
    ph = int(pw * frame.shape[0] / frame.shape[1])
    if ph >= h - STATUS_BAR_H - 20 or pw >= w:
        return
    y0, x0 = h - ph - 10, w - pw - 10
    display[y0 : y0 + ph, x0 : x0 + pw] = cv2.resize(frame, (pw, ph))
    cv2.rectangle(display, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), GREY, 1)


class _WindowUI:
    """The cv2 window: the original annunciator, and the only one with a
    camera preview. Needs a display the operator can actually see."""

    can_calibrate = True

    def __init__(self, config: MonitorConfig) -> None:
        self.config = config

    def start(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.config.screen_w, self.config.screen_h)
        if self.config.fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

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
        config = self.config
        display = np.zeros((config.screen_h, config.screen_w, 3), dtype=np.uint8)
        # Preview first: it is background context, and drawing it last
        # would cover the gaze marker exactly when the gaze is furthest
        # out of the zone and most worth seeing.
        if config.debug and frame is not None:
            _draw_preview(display, frame)
        if calibrating:
            _draw_calibration(display, calibrator, message, message_visible)
        else:
            _draw_monitoring(display, status, config)
            if message and message_visible:
                cv2.putText(
                    display,
                    message,
                    (int(config.screen_w * 0.12), int(config.screen_h * 0.9)),
                    FONT,
                    0.7,
                    (80, 200, 80),
                    2,
                )
        _draw_status_bar(display, status, fps, calibrating=calibrating)

        # The single point at which anything reaches the screen.
        cv2.imshow(WINDOW_NAME, display)

    def poll(self) -> str | None:
        # waitKey is also OpenCV's event pump, so this must be called every
        # frame whether or not anyone is pressing anything.
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            return "quit"
        if key == ord("c"):
            return "collect"
        if key == ord("r"):
            return "recalibrate"
        return None

    def closed(self) -> bool:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1

    def stop(self) -> None:
        cv2.destroyAllWindows()


class _LogUI:
    """No display at all: the installed unit annunciates through the log.

    It cannot calibrate, because calibration needs the operator to be told
    which corner to look at and to say when they are on it.
    """

    can_calibrate = False

    def start(self) -> None:
        pass

    def render(self, **_kwargs) -> None:
        pass

    def poll(self) -> str | None:
        return None

    def closed(self) -> bool:
        return False

    def stop(self) -> None:
        pass


def create_ui(config: MonitorConfig):
    """Pick the annunciator. The loop does not care which it gets."""
    if config.tui:
        from .terminal import TerminalUI

        return TerminalUI()
    if config.headless:
        return _LogUI()
    return _WindowUI(config)


def run_monitor(config: MonitorConfig, force_calibration: bool = False) -> int:
    """Run until the operator quits. Returns a process exit code."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required: pip install -r requirements.txt"
        ) from exc

    ui = create_ui(config)

    record = None if force_calibration else load_calibration(config.calibration_file)
    if record is not None:
        log.info(
            "Loaded calibration from %s (captured %s from %s)",
            config.calibration_file,
            record.created_utc or "unknown date",
            record.source,
        )
    elif not ui.can_calibrate:
        raise RuntimeError(
            f"No stored calibration at {config.calibration_file} and headless mode "
            "cannot run the calibration UI. Calibrate once with --tui over ssh, or "
            "with a display attached, then reuse the stored file."
        )
    else:
        log.info("No stored calibration; starting calibration.")

    monitor = AttentionMonitor(
        zone=record.zone if record else None,
        warn_after=config.warn_after,
        alert_after=config.alert_after,
        clear_after=config.clear_after,
        fault_after=config.fault_after,
        on_transition=_log_transition,
    )
    calibrator = GazeCalibrator(
        sample_duration=config.sample_duration,
        sigma_threshold=config.sigma_threshold,
    )
    calibrating = record is None
    filter_x = GazeFilter(alpha=config.filter_alpha)
    filter_y = GazeFilter(alpha=config.filter_alpha)

    cam = create_source(
        config.source,
        width=config.camera_width,
        height=config.camera_height,
        fps=config.fps,
        device=config.device,
        serial=config.serial,
    )

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # required for the iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    message = ""
    message_until = 0.0
    failures = 0
    frame_times: collections.deque[float] = collections.deque(maxlen=30)
    last_tick = time.monotonic()

    ui.start()

    try:
        cam.start()
    except CameraError as exc:
        log.error("%s", exc)
        ui.stop()
        return 2

    try:
        while True:
            now = time.monotonic()
            frame_times.append(now - last_tick)
            last_tick = now
            elapsed = sum(frame_times)
            fps = len(frame_times) / elapsed if elapsed > 0 else 0.0

            try:
                captured = cam.read()
            except CameraError as exc:
                log.error("Camera failed terminally: %s", exc)
                return 2

            if captured is None:
                failures += 1
                time.sleep(FAILURE_BACKOFF_SECONDS)
            else:
                failures = 0

            sample = None
            blinded = False
            if captured is not None:
                frame = captured.color
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                face_seen = bool(results.multi_face_landmarks)
                if face_seen:
                    raw = gaze_from_landmarks(
                        results.multi_face_landmarks[0], w, h, captured.depth_m
                    )
                    if raw is not None:
                        sample = _smooth(raw, filter_x, filter_y)

                # A covered lens still delivers frames at full rate, so read()
                # cannot see it and the health timer never starts. No depth
                # returns *and* no face is what being blinded looks like.
                # Either alone is normal: a depth sensor can fail while colour
                # still tracks a face, and an empty room has no face in it.
                blinded = not face_seen and depth_is_blind(
                    captured.depth_m, config.min_depth_fraction
                )

            camera_ok = failures < FAILURES_BEFORE_UNHEALTHY and not blinded

            if calibrating:
                if calibrator.is_collecting:
                    if sample is not None:
                        calibrator.collect(now, sample.dx, sample.dy, sample.z_m)
                    if calibrator.should_finish(now):
                        ok, message = calibrator.validate_and_save()
                        message_until = now + MESSAGE_SECONDS
                        log.info("Calibration: %s", message)
                        if calibrator.is_finished:
                            record = calibrator.build_record(
                                source=cam.name,
                                margin=config.zone_margin,
                                frame_size=(
                                    captured.size if captured is not None else None
                                ),
                            )
                            save_calibration(record, config.calibration_file)
                            monitor.zone = record.zone
                            monitor.reset()
                            calibrating = False
                            message = "Calibration saved"
                            message_until = now + MESSAGE_SECONDS
                status = monitor.update(now, None, camera_ok=camera_ok)
            else:
                status = monitor.update(now, sample, camera_ok=camera_ok)

            ui.render(
                calibrating=calibrating,
                calibrator=calibrator,
                status=status,
                fps=fps,
                message=message,
                message_visible=now < message_until,
                frame=captured.color if captured is not None else None,
            )

            if ui.closed():
                break

            command = ui.poll()
            if command == "quit":
                break
            if command == "collect" and calibrating and not calibrator.is_collecting:
                calibrator.start_collection(now)
            if command == "recalibrate":
                log.info("Recalibration requested")
                calibrator.reset()
                monitor.zone = None
                monitor.reset()
                filter_x.reset()
                filter_y.reset()
                calibrating = True
                message = ""
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        cam.stop()
        face_mesh.close()
        ui.stop()

    return 0
