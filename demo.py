"""Entry point for the operator attention monitor.

    python3 demo.py --config config.yaml

Calibration happens automatically on first run and is then reused on every
start. `--recalibrate` redoes it; `--factory-reset` discards it.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from gaze_monitor.calibration import default_calibration_path, factory_reset
from gaze_monitor.config import (
    SOURCE_NAMES,
    ConfigError,
    build_config,
    load_config_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alert an operator when their attention leaves the blade area.",
        epilog=(
            "Unset flags fall back to the config file, then to built-in "
            "defaults; see config.example.yaml for the values and what they mean."
        ),
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("config.yaml"),
        help="YAML config file; CLI flags take precedence over its values",
    )

    camera = parser.add_argument_group("camera")
    camera.add_argument(
        "--source",
        default=None,
        help=(
            "Camera backend: "
            + ", ".join(SOURCE_NAMES)
            + ", or 'gst:<pipeline>' for a raw GStreamer pipeline"
        ),
    )
    camera.add_argument(
        "--device", type=int, default=None, help="V4L2/CSI device index"
    )
    camera.add_argument("--serial", default=None, help="RealSense serial number")
    camera.add_argument("--camera-width", type=int, default=None, help="Capture width")
    camera.add_argument(
        "--camera-height", type=int, default=None, help="Capture height"
    )
    camera.add_argument("--fps", type=int, default=None, help="Capture frame rate")

    display = parser.add_argument_group("display")
    display.add_argument("--width", dest="screen_w", type=int, default=None)
    display.add_argument("--height", dest="screen_h", type=int, default=None)
    display.add_argument(
        "--fullscreen", action="store_true", default=None, help="Start fullscreen"
    )
    display.add_argument(
        "--headless",
        action="store_true",
        default=None,
        help="No window; annunciate through the log. Needs a stored calibration.",
    )
    display.add_argument(
        "--tui",
        action="store_true",
        default=None,
        help=(
            "No window; calibrate and monitor from the terminal. Works over "
            "ssh, and unlike --headless it can run the calibration."
        ),
    )
    display.add_argument(
        "--debug", action="store_true", default=None, help="Verbose logs and overlays"
    )

    attention = parser.add_argument_group("attention thresholds")
    attention.add_argument(
        "--warn-after", type=float, default=None, help="Seconds before warning"
    )
    attention.add_argument(
        "--alert-after", type=float, default=None, help="Seconds before alerting"
    )
    attention.add_argument(
        "--clear-after",
        type=float,
        default=None,
        help="Seconds back on target before an alarm clears",
    )
    attention.add_argument(
        "--fault-after",
        type=float,
        default=None,
        help="Seconds of unusable camera frames before declaring a fault",
    )
    attention.add_argument(
        "--min-depth-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of depth pixels below which the camera is treated as "
            "blinded (a covered lens still delivers frames); 0 disables"
        ),
    )
    attention.add_argument(
        "--zone-margin",
        type=float,
        default=None,
        help="Tolerance around the calibrated area, as a fraction of its size",
    )
    attention.add_argument(
        "--alpha",
        dest="filter_alpha",
        type=float,
        default=None,
        help="Gaze smoothing factor (0-1]; lower is smoother but slower",
    )

    metrics = parser.add_argument_group("metrics")
    metrics.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help=(
            "Serve Prometheus metrics on this port; 0 disables. Counters here "
            "survive slow scraping, which state-change logs do not."
        ),
    )

    calib = parser.add_argument_group("calibration")
    calib.add_argument(
        "--calibration-file",
        type=pathlib.Path,
        default=None,
        help=f"Calibration store (default: {default_calibration_path()})",
    )
    calib.add_argument(
        "--recalibrate",
        action="store_true",
        help="Redefine the attention area, overwriting the stored calibration",
    )
    calib.add_argument(
        "--factory-reset",
        action="store_true",
        help="Delete the stored calibration and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        file_values = load_config_file(args.config)
        cli_values = {
            key: getattr(args, key)
            for key in (
                "source",
                "device",
                "serial",
                "camera_width",
                "camera_height",
                "fps",
                "screen_w",
                "screen_h",
                "fullscreen",
                "headless",
                "tui",
                "debug",
                "filter_alpha",
                "warn_after",
                "alert_after",
                "clear_after",
                "fault_after",
                "zone_margin",
                "min_depth_fraction",
                "metrics_port",
                "calibration_file",
            )
        }
        config = build_config(file_values, cli_values)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.factory_reset:
        if factory_reset(config.calibration_file):
            print(f"Calibration cleared: {config.calibration_file}")
        else:
            print(f"No calibration to clear at {config.calibration_file}")
        return 0

    # Imported here, not at module scope: --help and --factory-reset are
    # useful on a box that has no OpenCV/MediaPipe stack installed.
    from gaze_monitor import ui

    try:
        return ui.run_monitor(config, force_calibration=args.recalibrate)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
