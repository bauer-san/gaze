# syntax=docker/dockerfile:1
#
# Two targets:
#
#   desktop -- x86_64 development. Everything installs from PyPI. x86_64 only:
#              it is built on Python 3.11, which is the one version with no
#              aarch64 pyrealsense2 wheel. On arm64 use the pi target.
#   jetson  -- Jetson Orin Nano Super and friends (JetPack 6 / L4T r36).
#              Everything installs from PyPI here too, except OpenCV, which
#              comes from apt because the PyPI wheels have no GStreamer.
#   pi      -- Raspberry Pi 4 on 64-bit Raspberry Pi OS. Pins Python 3.10:
#              the version Pi OS ships has no RealSense wheel, and the next
#              one up has a wheel that will not load there.
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

# --------------------------------------------------------------------- pi --
# Raspberry Pi 4 (64-bit Raspberry Pi OS). No NVIDIA anything: MediaPipe
# already runs on the CPU through XNNPACK, so nothing is lost relative to the
# Jetson except speed.
#
# Python 3.10 is not arbitrary, and it took two goes to get right.
#
# pyrealsense2 publishes aarch64 wheels for cp39, cp310 and cp312, but not
# cp311 -- which is exactly what Raspberry Pi OS Bookworm ships, so a native
# `pip install` on the stock OS fails outright. Pinning the interpreter is the
# main thing this image buys you.
#
# 3.12 looks like the obvious pin and does not work. Every one of those wheels
# is tagged manylinux2014, which claims glibc 2.17, and the tags understate
# what the binaries actually need:
#
#     cp39   GLIBC_2.28      cp310  GLIBC_2.34      cp312  GLIBC_2.38
#
# Bookworm -- Debian's and the Pi's -- has 2.36, so the cp312 wheel resolves,
# installs, and then fails at import with a missing GLIBC_2.38. Only cp310 is
# both present and loadable here, and it is the version already proven on the
# Jetson. The check at the end of this stage exists because pip resolution
# cannot catch this: the wheel is selectable, it just does not load.
#
# Consequence worth knowing: because the interpreter comes from the image
# rather than from apt, OpenCV has to come from pip too, and the pip wheels
# have no GStreamer. The D435i and USB webcams work; the CSI Pi Camera, which
# needs libcamerasrc through GStreamer, does not. Mixing in Debian's
# python3-opencv is not a way out -- it is built for the system Python 3.11,
# the version with no RealSense wheel at all.
FROM python:3.10-slim-bookworm AS pi

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GAZE_CALIBRATION_FILE=/var/lib/gaze_monitor/calibration.json

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libusb-1.0-0 \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-realsense.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt -r requirements-realsense.txt

# Fail here rather than on someone's desk. The Python/wheel matrix above is
# the whole reason this stage exists, so prove it rather than assume it.
RUN python3 - <<'PY'
import cv2
import mediapipe as mp
import pyrealsense2 as rs

mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1).close()
rs.context()
print(f"cv2 {cv2.__version__} | mediapipe {mp.__version__}")
PY

COPY . .

RUN mkdir -p /var/lib/gaze_monitor
VOLUME ["/var/lib/gaze_monitor"]

# A Pi 4 is slower than the Jetson, and filter_alpha is per frame while every
# threshold is in seconds, so the default smoothing is too slow at a low frame
# rate -- see README.md. Raise it here rather than shipping a demo that warns
# before the gaze estimate has caught up.
CMD ["python3", "demo.py", "--config", "config.example.yaml", \
     "--source", "realsense", "--alpha", "0.35", "--headless"]
