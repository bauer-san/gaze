"""Runtime configuration: defaults, YAML file, and CLI overrides.

Precedence is CLI > config file > built-in defaults. Keeping the merge here
rather than inline in demo.py means it can be tested, and means an unknown key
in someone's config.yaml produces a warning instead of being silently ignored.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field, fields
from typing import Any

import yaml

from .calibration import default_calibration_path
from .quality import MIN_DEPTH_FRACTION

log = logging.getLogger(__name__)

# Camera backends demo.py can offer and capture.create_source can build.
# Kept here rather than in capture.py so that --help and --factory-reset
# work on a machine with no OpenCV installed.
SOURCE_NAMES = ("realsense", "webcam", "kinect", "jetson")
GST_PREFIX = "gst:"

# Where a safety permit can be written, and how a USB relay is spoken to.
# Duplicated as literals in gaze_monitor.safety rather than imported, so that
# --help and --factory-reset still work with no pymodbus and no pyserial.
SAFETY_TRANSPORTS = ("tcp", "rtu", "relay")
RELAY_PROTOCOLS = ("numato", "lcus")
PARITIES = ("N", "E", "O")

# Overlays the preview can draw, cheapest first. Mirrors DRAW_MODES in
# gaze_monitor.preview, as a literal for the same reason as the tuples above.
PREVIEW_DRAW_MODES = ("none", "irises", "contours", "mesh")
PREVIEW_BACKGROUNDS = ("black", "camera")


class ConfigError(ValueError):
    """The supplied configuration cannot produce a working monitor."""


@dataclass
class MonitorConfig:
    # -- camera --
    source: str = "realsense"
    device: int = 0
    serial: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    fps: int = 30

    # -- display --
    screen_w: int = 1280
    screen_h: int = 720
    fullscreen: bool = False
    headless: bool = False
    # Calibration prompts and a live status line in the terminal, for
    # commissioning and running a unit over ssh. See gaze_monitor.terminal.
    tui: bool = False
    debug: bool = False

    # -- gaze --
    filter_alpha: float = 0.12

    # -- attention thresholds (seconds) --
    warn_after: float = 1.0
    alert_after: float = 3.0
    clear_after: float = 0.4
    fault_after: float = 1.0
    zone_margin: float = 0.1
    # Fraction of depth pixels below which a depth camera is treated as
    # blinded -- a covered lens keeps delivering frames, so nothing else
    # notices. 0 disables the check. Only applies to cameras with depth.
    min_depth_fraction: float = MIN_DEPTH_FRACTION

    # -- safety output --
    # Optional, and off unless brake_after is set together with somewhere to
    # write. The output is a permit that is continuously renewed, so loss of
    # contact means stop; see gaze_monitor.safety.
    brake_after: float = 0.0
    brake_on_fault: bool = False

    # Where the permit goes.
    #   tcp    Modbus/TCP           -- safety_host, safety_port
    #   rtu    Modbus RTU, serial   -- safety_serial_port and the line settings
    #   relay  USB relay module     -- safety_serial_port, no Modbus at all
    safety_transport: str = "tcp"
    safety_host: str = ""
    safety_port: int = 502
    safety_serial_port: str = ""
    safety_baud: int = 9600
    safety_parity: str = "N"
    safety_stopbits: int = 1
    safety_bytesize: int = 8
    safety_unit_id: int = 1
    safety_kind: str = "coil"
    safety_address: int = 0
    # USB relay modules are all slightly different. "numato" is line-based
    # ASCII over CDC-ACM, "lcus" is the four-byte CH340 dialect.
    safety_relay_protocol: str = "numato"
    safety_relay_channel: int = 0
    safety_interval: float = 0.1
    safety_stale_after: float = 0.5
    # Skips the start-up check that the machine cannot run before permits are
    # issued. Off by default, because arming automatically after a crash is
    # exactly what a safety output must not do.
    safety_auto_arm: bool = False

    # -- machine state, read only --
    # Polls a variable-frequency drive over Modbus RTU for context: running or
    # stopped, what it is drawing, whether it has tripped. Nothing here can
    # command anything. The permit is a separate output on separate hardware,
    # and keeping the two apart is the point -- a bug in the reader cannot
    # reach the machine. Empty machine_port disables it.
    machine_port: str = ""
    machine_baud: int = 9600
    machine_parity: str = "N"
    machine_stopbits: int = 1
    machine_bytesize: int = 8
    machine_unit_id: int = 1
    # Frame address = documented address + offset. Several drives, LS Electric
    # among them, document a communication address one higher than the number
    # that actually goes on the wire, so -1 is right for those and 0 for a
    # drive that does not play that game. Get it wrong and every read returns
    # the neighbouring register, plausibly and silently.
    machine_address_offset: int = -1
    machine_interval: float = 1.0

    # -- live landmark preview, a demonstration aid --
    # Serves the annotated camera image over HTTP so the face mesh can be
    # watched from a browser. 0 disables it, which is the default: this is
    # live video of whoever is in front of the camera, served to anyone who
    # can reach the port, with no authentication. Commissioning and demos
    # only, not something to leave running on an installed machine.
    #
    # It cannot slow the monitor down. The capture loop only copies a frame,
    # and only when somebody is watching; drawing and encoding happen on the
    # HTTP threads. See gaze_monitor.preview.
    preview_port: int = 0
    preview_bind: str = "0.0.0.0"
    preview_quality: int = 80
    preview_max_fps: float = 10.0
    preview_draw: str = "contours"
    # "black" draws the landmarks on nothing, and is the default: the camera
    # image is then never copied out of the capture loop, so the endpoint
    # serves geometry and no picture of anyone. "camera" shows the live image
    # underneath, which is what answers "why is tracking poor here".
    preview_background: str = "black"

    # -- metrics --
    # TCP port for the Prometheus exporter; 0 disables it. Off by default
    # because opening a listening socket should be asked for, not assumed.
    metrics_port: int = 0

    # -- calibration --
    sample_duration: float = 1.0
    sigma_threshold: float = 0.05
    calibration_file: pathlib.Path = field(default_factory=default_calibration_path)

    def __post_init__(self) -> None:
        self.calibration_file = pathlib.Path(self.calibration_file).expanduser()

        if self.source not in SOURCE_NAMES and not self.source.startswith(GST_PREFIX):
            raise ConfigError(
                f"Unknown source {self.source!r}. Expected one of "
                f"{', '.join(SOURCE_NAMES)}, or a '{GST_PREFIX}<pipeline>' string."
            )

        if not 0.0 < self.filter_alpha <= 1.0:
            raise ConfigError(
                f"filter_alpha must be in (0, 1], got {self.filter_alpha}"
            )
        if self.warn_after < 0 or self.alert_after < 0:
            raise ConfigError("warn_after and alert_after must be >= 0")
        if self.warn_after > self.alert_after:
            raise ConfigError(
                f"warn_after ({self.warn_after}) must not exceed "
                f"alert_after ({self.alert_after})"
            )
        if self.zone_margin < 0:
            raise ConfigError("zone_margin must be >= 0")
        if self.sample_duration <= 0:
            raise ConfigError("sample_duration must be > 0")
        if self.brake_after < 0:
            raise ConfigError("brake_after must be >= 0")
        if self.safety_transport not in SAFETY_TRANSPORTS:
            raise ConfigError(
                f"safety_transport must be one of {', '.join(SAFETY_TRANSPORTS)}, "
                f"got {self.safety_transport!r}"
            )
        # Which field has to be filled in depends on the transport, and naming
        # the wrong one in the error is how people end up setting both.
        needs = (
            "safety_host" if self.safety_transport == "tcp" else "safety_serial_port"
        )
        if self.brake_after > 0 and not getattr(self, needs):
            # A brake threshold with nowhere to send it looks armed and does
            # nothing, which is the worst state a safety feature can be in.
            raise ConfigError(
                f"brake_after is set but {needs} is empty, so a stop could "
                f"never be demanded. Set {needs}, or set brake_after to 0."
            )
        if self.safety_kind not in ("coil", "register"):
            raise ConfigError(
                f"safety_kind must be 'coil' or 'register', got {self.safety_kind!r}"
            )
        if self.safety_relay_protocol not in RELAY_PROTOCOLS:
            raise ConfigError(
                f"safety_relay_protocol must be one of {', '.join(RELAY_PROTOCOLS)}, "
                f"got {self.safety_relay_protocol!r}"
            )
        if self.safety_relay_channel < 0:
            raise ConfigError("safety_relay_channel must be >= 0")
        if not 1 <= self.safety_port <= 65535:
            raise ConfigError(
                f"safety_port must be in [1, 65535], got {self.safety_port}"
            )
        if not 0 <= self.safety_unit_id <= 247:
            raise ConfigError(
                f"safety_unit_id must be in [0, 247], got {self.safety_unit_id}"
            )
        if self.safety_address < 0:
            raise ConfigError("safety_address must be >= 0")
        _check_serial_line(
            "safety",
            self.safety_baud,
            self.safety_parity,
            self.safety_stopbits,
            self.safety_bytesize,
        )
        if self.safety_interval <= 0:
            raise ConfigError("safety_interval must be > 0")
        if self.safety_stale_after <= self.safety_interval:
            # Otherwise the staleness check fires before the first renewal
            # lands, and the output is permanently silent.
            raise ConfigError(
                f"safety_stale_after ({self.safety_stale_after}) must exceed "
                f"safety_interval ({self.safety_interval})"
            )
        if not 0 <= self.machine_unit_id <= 247:
            raise ConfigError(
                f"machine_unit_id must be in [0, 247], got {self.machine_unit_id}"
            )
        if self.machine_interval <= 0:
            raise ConfigError("machine_interval must be > 0")
        _check_serial_line(
            "machine",
            self.machine_baud,
            self.machine_parity,
            self.machine_stopbits,
            self.machine_bytesize,
        )
        if self.machine_port and self.machine_port == self.safety_serial_port:
            # Two devices can share an RS-485 bus, but not a USB relay, and
            # this mistake reads as "the permit stopped working" rather than
            # as a configuration error.
            raise ConfigError(
                "machine_port and safety_serial_port are the same device "
                f"({self.machine_port!r}). The permit output and the drive "
                "reader must not share a port."
            )
        if not 0 <= self.preview_port <= 65535:
            raise ConfigError(
                f"preview_port must be in [0, 65535], got {self.preview_port}"
            )
        if self.preview_draw not in PREVIEW_DRAW_MODES:
            raise ConfigError(
                f"preview_draw must be one of {', '.join(PREVIEW_DRAW_MODES)}, "
                f"got {self.preview_draw!r}"
            )
        if self.preview_background not in PREVIEW_BACKGROUNDS:
            raise ConfigError(
                f"preview_background must be one of "
                f"{', '.join(PREVIEW_BACKGROUNDS)}, "
                f"got {self.preview_background!r}"
            )
        if not 1 <= self.preview_quality <= 100:
            raise ConfigError(
                f"preview_quality must be in [1, 100], got {self.preview_quality}"
            )
        if self.preview_max_fps < 0:
            raise ConfigError("preview_max_fps must be >= 0")
        if self.preview_port and self.preview_port == self.metrics_port:
            # Both want a listening socket, and the collision surfaces as
            # whichever started second silently not being there.
            raise ConfigError(
                f"preview_port and metrics_port are both {self.preview_port}"
            )
        if not 0 <= self.metrics_port <= 65535:
            raise ConfigError(
                f"metrics_port must be in [0, 65535], got {self.metrics_port}"
            )
        if not 0.0 <= self.min_depth_fraction < 1.0:
            raise ConfigError(
                "min_depth_fraction must be in [0, 1), got "
                f"{self.min_depth_fraction}"
            )
        if min(self.screen_w, self.screen_h) <= 0:
            raise ConfigError("screen_w and screen_h must be positive")
        if self.headless and self.fullscreen:
            raise ConfigError("fullscreen makes no sense with headless")
        if self.tui and self.fullscreen:
            raise ConfigError("fullscreen makes no sense with tui")
        if self.tui and self.headless:
            raise ConfigError(
                "tui and headless are different annunciators; pick one. "
                "headless logs and cannot calibrate; tui draws a status line "
                "in the terminal and can."
            )


def _check_serial_line(
    prefix: str, baud: int, parity: str, stopbits: int, bytesize: int
) -> None:
    """Validate one set of serial line settings.

    Shared because the safety output and the drive reader each have their own
    port, and a mismatched line setting fails as silence rather than as an
    error -- the hardest kind of fault to find on a two-wire bus.
    """
    if baud <= 0:
        raise ConfigError(f"{prefix}_baud must be > 0, got {baud}")
    if parity not in PARITIES:
        raise ConfigError(
            f"{prefix}_parity must be one of {', '.join(PARITIES)}, got {parity!r}"
        )
    if stopbits not in (1, 2):
        raise ConfigError(f"{prefix}_stopbits must be 1 or 2, got {stopbits}")
    if bytesize not in (7, 8):
        raise ConfigError(f"{prefix}_bytesize must be 7 or 8, got {bytesize}")


def load_config_file(path: pathlib.Path) -> dict[str, Any]:
    """Read a YAML config. A missing file is not an error."""
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a YAML mapping, got {type(data).__name__}"
        )
    return data


def build_config(
    file_values: dict[str, Any] | None = None,
    cli_values: dict[str, Any] | None = None,
) -> MonitorConfig:
    """Merge file and CLI values over the defaults.

    ``None`` in ``cli_values`` means "flag not supplied", so it falls through
    to the file value and then the default.
    """
    known = {f.name for f in fields(MonitorConfig)}
    merged: dict[str, Any] = {}

    for key, value in (file_values or {}).items():
        if key not in known:
            log.warning("Ignoring unknown config key %r", key)
            continue
        merged[key] = value

    for key, value in (cli_values or {}).items():
        if value is None:
            continue
        if key not in known:
            log.warning("Ignoring unknown CLI key %r", key)
            continue
        merged[key] = value

    return MonitorConfig(**merged)
