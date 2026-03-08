"""Entry point to run the canonical gaze demo using the kinect_gaze package."""

import argparse
from kinect_gaze import ui


def main():
    parser = argparse.ArgumentParser(description="Run gaze demo")
    parser.add_argument("--source", choices=["webcam", "kinect"], default="webcam", help="Input source")
    parser.add_argument("--width", type=int, default=1280, help="Virtual screen width for calibration mapping")
    parser.add_argument("--height", type=int, default=720, help="Virtual screen height for calibration mapping")
    parser.add_argument("--alpha", type=float, default=0.12, help="Smoothing alpha for gaze filter (0-1)")
    parser.add_argument("--fullscreen", action="store_true", help="Start display in fullscreen mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging/overlay")
    args = parser.parse_args()

    ui.run_monitor(
        source=args.source,
        screen_w=args.width,
        screen_h=args.height,
        filter_alpha=args.alpha,
        fullscreen=args.fullscreen,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
