"""Entry point to run the canonical gaze demo using the kinect_gaze package."""

import argparse
import pathlib
from typing import Any, Dict

import yaml

from kinect_gaze import ui


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Run gaze demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--source", choices=["webcam", "kinect"], default=None, help="Input source")
    parser.add_argument("--width", type=int, default=None, help="Virtual screen width for calibration mapping")
    parser.add_argument("--height", type=int, default=None, help="Virtual screen height for calibration mapping")
    parser.add_argument("--alpha", type=float, default=None, help="Smoothing alpha for gaze filter (0-1)")
    parser.add_argument("--fullscreen", action="store_true", default=None, help="Start display in fullscreen mode")
    parser.add_argument("--debug", action="store_true", default=None, help="Enable debug logging/overlay")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    cfg = load_config(cfg_path)

    # Merge precedence: CLI args (if provided) > config file > defaults
    source = args.source if args.source is not None else cfg.get("source", "webcam")
    screen_w = args.width if args.width is not None else cfg.get("screen_w", 1280)
    screen_h = args.height if args.height is not None else cfg.get("screen_h", 720)
    filter_alpha = args.alpha if args.alpha is not None else cfg.get("filter_alpha", 0.12)

    # For booleans, argparse defaults to None above so we can detect omission
    fullscreen = args.fullscreen if args.fullscreen is not None else cfg.get("fullscreen", False)
    debug = args.debug if args.debug is not None else cfg.get("debug", False)

    ui.run_monitor(
        source=source,
        screen_w=screen_w,
        screen_h=screen_h,
        filter_alpha=filter_alpha,
        fullscreen=fullscreen,
        debug=debug,
    )


if __name__ == "__main__":
    main()
