"""Entry point to run the canonical gaze demo using the kinect_gaze package."""

import argparse
from kinect_gaze import ui


def main():
    parser = argparse.ArgumentParser(description="Run gaze demo")
    parser.add_argument("--source", choices=["webcam", "kinect"], default="webcam")
    args = parser.parse_args()

    ui.run_monitor(source=args.source)


if __name__ == "__main__":
    main()
