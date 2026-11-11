"""The simulated operator as a camera the monitor can open.

:class:`SimulatedCamera` implements the same ``start``/``read``/``stop``
contract as :class:`gaze_monitor.capture.CameraSource` and yields the same
:class:`~gaze_monitor.capture.Frame` (BGR colour, depth in metres), so any
code that consumes a camera can consume the simulator instead.

It works in two ways:

* **Playback** of a run made with ``generate --save-frames``, paced to the
  wall clock. This is the one to point the real monitor at: rendering costs
  hundreds of milliseconds a frame, far too slow to drive a monitor whose
  timers are in wall-clock seconds and whose calibration needs several
  samples a second.
* **Rendering on demand**, as fast as frames can be produced, with
  simulated time advancing by exactly one frame per ``read()``. For harnesses
  that keep their own clock, such as :mod:`.evaluate`.

Either way ``last_truth`` holds the ground truth for the frame just read.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class _Frame:
    """Stand-in for gaze_monitor.capture.Frame when OpenCV is unavailable."""

    color: np.ndarray
    depth_m: np.ndarray | None = None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.color.shape[:2]
        return w, h


def _frame_class():
    try:
        from gaze_monitor.capture import Frame

        return Frame
    except ImportError:  # pragma: no cover - no OpenCV in this environment
        return _Frame


class SimulatedCamera:
    name = "sim"
    has_depth = True

    def __init__(
        self,
        run_dir: str | pathlib.Path | None = None,
        loop: bool = True,
        **sim_kwargs,
    ):
        self.run_dir = pathlib.Path(run_dir) if run_dir else None
        self.loop = loop
        self.sim_kwargs = sim_kwargs
        self.last_truth: dict | None = None
        self._sim = None
        self._truth = None
        self._t0 = None
        self._fps = float(sim_kwargs.get("fps", 30.0))
        self._n = 0
        self._Frame = _frame_class()

    def start(self) -> None:
        if self.run_dir is not None:
            frames = self.run_dir / "frames"
            if not frames.is_dir():
                raise RuntimeError(
                    f"{self.run_dir} has no frames/. Generate it with --save-frames "
                    "to play it back."
                )
            meta = json.loads((self.run_dir / "meta.json").read_text())
            self._fps = float(meta["config"]["fps"])
            self._n = int(meta["frames"])
            self._truth = (self.run_dir / "truth.jsonl").read_text().splitlines()
            self._t0 = time.monotonic()
        else:
            from .sim import Simulation

            self._sim = Simulation(**self.sim_kwargs)
            self._n = self._sim.frame_count

    def read(self):
        if self.run_dir is not None:
            return self._playback()
        if self._sim is None:
            raise RuntimeError("read() before start()")
        if self._sim._index >= self._n:
            return None
        frame = self._sim.next_frame()
        self.last_truth = frame.truth
        return self._Frame(color=frame.render.color, depth_m=frame.render.depth_m)

    def _playback(self):
        import cv2

        if self._t0 is None:
            raise RuntimeError("read() before start()")
        elapsed = time.monotonic() - self._t0
        idx = int(elapsed * self._fps)
        if idx >= self._n:
            if not self.loop:
                return None
            self._t0 = time.monotonic()
            idx = 0
        # Pace to the frame clock, as a real camera blocks until its next frame.
        due = self._t0 + (idx + 1) / self._fps
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        stem = self.run_dir / "frames" / f"{idx:06d}"
        color = cv2.imread(str(stem) + ".jpg", cv2.IMREAD_COLOR)
        depth = cv2.imread(str(stem) + "_depth.png", cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            return None
        self.last_truth = json.loads(self._truth[idx])
        return self._Frame(color=color, depth_m=depth.astype(np.float32) / 1000.0)

    def stop(self) -> None:
        self._sim = None
        self._t0 = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def run_live(
    demo_args: list[str], run_dir: str | pathlib.Path, loop: bool = True
) -> int:
    """Run ``demo.py`` with its camera replaced by a recorded simulated run.

    Everything else is the real application: calibration UI or ``--tui``,
    the state machine, metrics, the preview server, the safety output.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import demo
    from gaze_monitor import ui

    camera = SimulatedCamera(run_dir=run_dir, loop=loop)
    ui.create_source = lambda *a, **k: camera
    args = list(demo_args)
    if "--source" not in args:
        # Satisfies the config's source validation; the backend is replaced.
        args = ["--source", "webcam", *args]
    return demo.main(args)
