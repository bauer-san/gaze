# syntax=docker/dockerfile:1
#
# Two targets:
#
#   desktop -- x86_64 development. Everything installs from PyPI.
#   jetson  -- Jetson Orin Nano Super and friends (JetPack 6 / L4T r36).
#              Everything installs from PyPI here too, except OpenCV, which
#              comes from apt because the PyPI wheels have no GStreamer.
#
#   docker compose build gaze
#   docker compose --profile jetson build gaze-jetson
#
# The Jetson image is built and published by CI; see
# .github/workflows/ci.yml and jetson/README.md. Building it on the board
# itself works but is the slow path.

# Declared before the first FROM so it is in the global scope and usable in
# the jetson stage's FROM. An ARG after a FROM belongs to that stage only.
ARG L4T_TAG=r36.2.0

# ---------------------------------------------------------------- desktop --
FROM python:3.11-slim AS desktop

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GAZE_CALIBRATION_FILE=/var/lib/gaze_monitor/calibration.json

# OpenCV's runtime libraries. libgl1/libglib2.0-0 are needed even to import
# cv2; the rest are needed for imshow during calibration.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so editing the code does not reinstall them.
COPY requirements.txt requirements-realsense.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt -r requirements-realsense.txt

COPY . .

RUN mkdir -p /var/lib/gaze_monitor
VOLUME ["/var/lib/gaze_monitor"]

CMD ["python3", "demo.py", "--config", "config.example.yaml"]

# ----------------------------------------------------------------- jetson --
# l4t-base, not l4t-jetpack. Nothing in this pipeline touches CUDA: MediaPipe
# runs on the CPU through XNNPACK, OpenCV comes from apt, and the D435i is a
# USB device. That is a 276 MB base instead of a 5.6 GB one, which is what
# makes the CI build possible at all -- a standard GitHub runner has nowhere
# near the disk for l4t-jetpack. The NVIDIA container runtime mounts the
# host's Tegra userspace into the container, so nvarguscamerasrc still works
# for the CSI source.
#
# If you take the TensorRT route in jetson/optimizations.md you will need CUDA
# in the image: switch this back to nvcr.io/nvidia/l4t-jetpack and set
# L4T_TAG=r36.4.0 (the newest tag NGC publishes for that image).
FROM nvcr.io/nvidia/l4t-base:${L4T_TAG} AS jetson

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    GAZE_CALIBRATION_FILE=/var/lib/gaze_monitor/calibration.json

# python3-opencv from apt, never the PyPI wheel: the distribution build is the
# one with GStreamer, which the jetson/CSI and gst: camera sources need. The
# pip step below removes the wheel that MediaPipe drags in, and the check at
# the end of this stage fails the build if the wrong one wins anyway.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-opencv \
        libusb-1.0-0 \
        ca-certificates \
        v4l-utils \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so editing the code does not reinstall them.
#
# Both of the dependencies that used to need special handling here now ship
# aarch64 wheels: pyrealsense2 at the pinned version, and mediapipe at 0.10.18
# (see the notes in requirements.txt). librealsense no longer has to be built
# from source, and no externally supplied MediaPipe wheel is needed.
#
# MediaPipe depends on opencv-contrib-python, and pip installs it under
# /usr/local, which precedes the apt build on sys.path. Removing it leaves the
# apt cv2 -- 'pip uninstall' of something absent is a no-op, not an error.
COPY requirements.txt requirements-realsense.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt -r requirements-realsense.txt \
    && python3 -m pip uninstall -y opencv-contrib-python opencv-python

# Fail the build here rather than on the board. Every one of these has been a
# real failure mode: the wrong cv2 winning, an OpenCV without GStreamer, a
# MediaPipe that imports but cannot build a graph, a pyrealsense2 that cannot
# reach libusb.
RUN python3 - <<'PY'
import cv2
import mediapipe as mp
import pyrealsense2 as rs

gst = next(
    (ln for ln in cv2.getBuildInformation().splitlines()
     if ln.strip().startswith("GStreamer:")),
    "",
)
assert cv2.__file__.startswith("/usr/lib/"), (
    f"a pip OpenCV is shadowing the apt build: {cv2.__file__}"
)
assert "YES" in gst, f"this OpenCV has no GStreamer support: {gst.strip()!r}"

mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1).close()
rs.context()

print(f"cv2 {cv2.__version__} ({cv2.__file__})")
print(f"mediapipe {mp.__version__}")
print(gst.strip())
PY

COPY . .

RUN mkdir -p /var/lib/gaze_monitor
VOLUME ["/var/lib/gaze_monitor"]

# Headless by default: an installed unit has no monitor. Calibration is a
# separate, one-off run with a display attached (see docker-compose.yml).
CMD ["python3", "demo.py", "--config", "config.example.yaml", "--headless"]
