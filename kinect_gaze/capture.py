"""Camera backends.

Every backend yields a :class:`Frame` with:

* ``color``   -- BGR uint8, HxWx3.
* ``depth_m`` -- float32 HxW in **metres**, aligned pixel-for-pixel with
  ``color``, or ``None`` for cameras without depth. 0.0 means "no reading".

Unit conversion and depth/colour alignment happen here, once, so that the
rest of the package never has to ask what kind of camera it is talking to.

``read()`` returns ``None`` for a *transient* failure (a dropped frame, a
timed-out wait) and raises for a *terminal* one (device disconnected). Callers
are expected to tolerate a run of ``None`` and then declare a fault -- see
:class:`kinect_gaze.attention.AttentionMonitor`.
"""

from __future__ import annotations

import abc
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

from .config import GST_PREFIX, SOURCE_NAMES

log = logging.getLogger(__name__)

# Kinect v1 DEPTH_REGISTERED is millimetres aligned to the RGB camera.
_KINECT_MM_TO_M = 1.0 / 1000.0


class CameraError(RuntimeError):
    """The camera cannot be used at all."""


class UnknownSourceError(CameraError):
    """The requested source name is not a backend we implement."""


@dataclass
class Frame:
    color: np.ndarray
    depth_m: np.ndarray | None = None

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the colour image."""
        h, w = self.color.shape[:2]
        return w, h


class CameraSource(abc.ABC):
    """Base class for camera backends."""

    name = "camera"
    has_depth = False

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> Frame | None: ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Release the device. Must tolerate being called twice."""

    def __enter__(self) -> CameraSource:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class WebcamSource(CameraSource):
    """Any V4L2/AVFoundation device OpenCV can open. No depth."""

    name = "webcam"

    def __init__(
        self, device: int = 0, width: int = 640, height: int = 480, fps: int = 30
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise CameraError(f"Unable to open webcam device {self.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep latency down: without this the driver hands us stale frames
        # from a queue, which reads as lag in the attention timer.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Frame | None:
        if self.cap is None:
            raise CameraError("read() before start()")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return Frame(color=frame)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class RealSenseSource(CameraSource):
    """Intel RealSense (D435i and friends), depth aligned to colour.

    The depth frame is multiplied by the device's own depth scale, so
    ``depth_m`` is metres regardless of how the unit is configured.
    """

    name = "realsense"
    has_depth = True

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
        timeout_ms: int = 2000,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.timeout_ms = timeout_ms
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # pragma: no cover - depends on hardware SDK
            raise CameraError(
                "pyrealsense2 is not installed. Install it with "
                "`pip install -r requirements-realsense.txt`, or on Jetson "
                "build librealsense with -DBUILD_PYTHON_BINDINGS=ON "
                "(see jetson/README.md)."
            ) from exc

        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )

        try:
            profile = self.pipeline.start(config)
        except RuntimeError as exc:  # pragma: no cover - depends on hardware
            raise CameraError(
                f"Could not start the RealSense pipeline at {self.width}x"
                f"{self.height}@{self.fps}: {exc}. Check the camera is on USB 3 "
                "(a USB 2 link cannot sustain this mode) and that no other "
                "process holds the device."
            ) from exc

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        # Align depth into the colour frame: the landmark pixels we probe come
        # from the colour image, and the two sensors are physically offset.
        self.align = rs.align(rs.stream.color)
        log.info(
            "RealSense started: %dx%d@%d, depth scale %.6f m/unit",
            self.width,
            self.height,
            self.fps,
            self.depth_scale,
        )

    def read(self) -> Frame | None:
        if self.pipeline is None:
            raise CameraError("read() before start()")
        try:
            frames = self.pipeline.wait_for_frames(self.timeout_ms)
        except RuntimeError:
            # Timeouts are routine under USB contention; treat as a dropped frame.
            return None

        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        return Frame(color=color, depth_m=depth_m)

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None


class KinectSource(CameraSource):
    """Kinect v1 via libfreenect. Legacy; the supported camera is RealSense."""

    name = "kinect"
    has_depth = True

    def __init__(self, device: int = 0):
        self.device = device
        self._freenect = None

    def start(self) -> None:
        try:
            import freenect
        except ImportError as exc:  # pragma: no cover - depends on hardware SDK
            raise CameraError(
                "freenect is not available. It ships with libfreenect's Python "
                "bindings and is not installable from PyPI; build libfreenect "
                "with BUILD_PYTHON3=ON, or use --source realsense."
            ) from exc
        self._freenect = freenect

    def read(self) -> Frame | None:
        fn = self._freenect
        if fn is None:
            raise CameraError("read() before start()")

        # DEPTH_REGISTERED is millimetres aligned to the RGB camera -- NOT the
        # raw 11-bit disparity, so the 0.1236*tan(...) conversion that used to
        # live here was decoding the wrong quantity entirely.
        depth_result = fn.sync_get_depth(self.device, fn.DEPTH_REGISTERED)
        video_result = fn.sync_get_video(self.device, fn.VIDEO_RGB)
        if depth_result is None or video_result is None:
            return None

        depth_mm, _ = depth_result
        rgb, _ = video_result
        depth_m = depth_mm.astype(np.float32) * _KINECT_MM_TO_M
        # libfreenect reports unknown depth as 0; keep that convention.
        depth_m[depth_mm == 0] = 0.0
        return Frame(color=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_m=depth_m)

    def stop(self) -> None:
        if self._freenect is not None:
            try:
                self._freenect.sync_stop()
            except Exception:  # pragma: no cover - best effort on shutdown
                log.debug("freenect.sync_stop() failed", exc_info=True)
            self._freenect = None


def has_gstreamer_support() -> bool:
    """Whether this OpenCV build can open GStreamer pipelines.

    getBuildInformation() pads its columns, so match the value rather than a
    fixed-width substring -- the spacing differs between builds.
    """
    match = re.search(
        r"^\s*GStreamer:\s*(\S+)", cv2.getBuildInformation(), re.MULTILINE
    )
    return bool(match) and match.group(1).upper() == "YES"


class GStreamerSource(CameraSource):
    """A GStreamer pipeline read through OpenCV's appsink. No depth."""

    name = "gstreamer"

    def __init__(self, pipeline: str):
        self.pipeline = pipeline
        self.cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            hint = ""
            if not has_gstreamer_support():
                hint = (
                    " This OpenCV build has no GStreamer support -- the PyPI "
                    "wheels are built without it. Use the distribution package "
                    "(apt install python3-opencv) or the container image."
                )
            raise CameraError(f"Could not open GStreamer pipeline.{hint}")

    def read(self) -> Frame | None:
        if self.cap is None:
            raise CameraError("read() before start()")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return Frame(color=frame)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def jetson_csi_pipeline(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """GStreamer pipeline for a CSI camera on Jetson via nvarguscamerasrc."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def create_source(
    spec: str = "realsense",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    device: int = 0,
    serial: str | None = None,
) -> CameraSource:
    """Build a camera backend from a source string.

    Accepted values: ``realsense``, ``webcam``, ``kinect``, ``jetson``, or a
    raw GStreamer pipeline prefixed with ``gst:``.

    An unrecognised name raises. It used to fall through to the webcam, so
    ``--source jetson`` silently monitored the wrong camera.
    """
    spec = (spec or "").strip()

    if spec.startswith(GST_PREFIX):
        return GStreamerSource(spec[len(GST_PREFIX) :])
    if spec == "realsense":
        return RealSenseSource(width=width, height=height, fps=fps, serial=serial)
    if spec == "webcam":
        return WebcamSource(device=device, width=width, height=height, fps=fps)
    if spec == "kinect":
        return KinectSource(device=device)
    if spec == "jetson":
        return GStreamerSource(
            jetson_csi_pipeline(sensor_id=device, width=width, height=height, fps=fps)
        )

    raise UnknownSourceError(
        f"Unknown camera source {spec!r}. Expected one of "
        f"{', '.join(SOURCE_NAMES)}, or a '{GST_PREFIX}<pipeline>' string."
    )
