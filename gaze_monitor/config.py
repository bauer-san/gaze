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
    # Optional, and off unless both brake_after and safety_host are set. The
    # output is a permit that is continuously renewed, so loss of contact
    # means stop; see gaze_monitor.safety.
    brake_after: float = 0.0
    brake_on_fault: bool = False
    safety_host: str = ""
    safety_port: int = 502
    safety_unit_id: int = 1
    safety_kind: str = "coil"
    safety_address: int = 0
    safety_interval: float = 0.1
    safety_stale_after: float = 0.5
    # Skips the start-up check that the machine cannot run before permits are
    # issued. Off by default, because arming automatically after a crash is
    # exactly what a safety output must not do.
    safety_auto_arm: bool = False

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
        if self.brake_after > 0 and not self.safety_host:
            # A brake threshold with nowhere to send it looks armed and does
            # nothing, which is the worst state a safety feature can be in.
            raise ConfigError(
                "brake_after is set but safety_host is empty, so a stop could "
                "never be demanded. Set safety_host, or set brake_after to 0."
            )
        if self.safety_kind not in ("coil", "register"):
            raise ConfigError(
                f"safety_kind must be 'coil' or 'register', got {self.safety_kind!r}"
            )
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
        if self.safety_interval <= 0:
            raise ConfigError("safety_interval must be > 0")
        if self.safety_stale_after <= self.safety_interval:
            # Otherwise the staleness check fires before the first renewal
            # lands, and the output is permanently silent.
            raise ConfigError(
                f"safety_stale_after ({self.safety_stale_after}) must exceed "
                f"safety_interval ({self.safety_interval})"
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
