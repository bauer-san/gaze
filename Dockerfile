# syntax=docker/dockerfile:1
#
# Two targets:
#
#   desktop -- x86_64 development. Everything installs from PyPI, including
#              pyrealsense2, so this builds unattended.
#   jetson  -- Jetson Orin Nano Super and friends (JetPack 6 / L4T r36).
#              OpenCV comes from apt (the PyPI wheels have no CUDA and no
#              GStreamer) and librealsense is built from source, because Intel
#              publishes no aarch64 wheels.
#
#   docker compose build gaze
#   docker compose --profile jetson build gaze-jetson

# Declared before the first FROM so it is in the global scope and usable in
# the jetson stage's FROM. An ARG after a FROM belongs to that stage only.
ARG L4T_TAG=r36.4.0

# ---------------------------------------------------------------- desktop --
FROM python:3.11-slim AS desktop

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GAZE_CALIBRATION_FILE=/var/lib/kinect_gaze/calibration.json

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

RUN mkdir -p /var/lib/kinect_gaze
VOLUME ["/var/lib/kinect_gaze"]

CMD ["python3", "demo.py", "--config", "config.example.yaml"]

# ----------------------------------------------------------------- jetson --
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG} AS jetson

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    GAZE_CALIBRATION_FILE=/var/lib/kinect_gaze/calibration.json

# python3-opencv from apt, never the PyPI wheel: the distribution build is the
# one with GStreamer (so the jetson/CSI camera sources work) and is built for
# this architecture.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-opencv \
        python3-numpy \
        libusb-1.0-0-dev \
        libssl-dev \
        libudev-dev \
        pkg-config \
        cmake \
        git \
        ca-certificates \
        v4l-utils \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

# --- librealsense with Python bindings -------------------------------------
# FORCE_RSUSB_BACKEND avoids the patched kernel modules, which a container
# cannot load anyway; the RSUSB backend talks to the D435i over libusb.
ARG LIBREALSENSE_VERSION=v2.56.5
RUN git clone --depth 1 --branch ${LIBREALSENSE_VERSION} \
        https://github.com/IntelRealSense/librealsense.git /tmp/librealsense \
    && cmake -S /tmp/librealsense -B /tmp/librealsense/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DFORCE_RSUSB_BACKEND=true \
        -DBUILD_PYTHON_BINDINGS=true \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false \
        -DBUILD_UNIT_TESTS=false \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    && cmake --build /tmp/librealsense/build --parallel "$(nproc)" \
    && cmake --install /tmp/librealsense/build \
    && ldconfig \
    && rm -rf /tmp/librealsense

ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages:/usr/local/lib

# --- mediapipe -------------------------------------------------------------
# MediaPipe publishes no aarch64 wheels. Supply one built for JetPack:
#     docker compose --profile jetson build \
#         --build-arg MEDIAPIPE_WHEEL=https://.../mediapipe-0.10.x-...aarch64.whl
# A local file works too if you COPY it in first. This deliberately fails the
# build rather than producing an image that looks fine and dies at runtime.
ARG MEDIAPIPE_WHEEL=""
RUN if [ -n "${MEDIAPIPE_WHEEL}" ]; then \
        python3 -m pip install --no-cache-dir "${MEDIAPIPE_WHEEL}"; \
    else \
        python3 -m pip install --no-cache-dir mediapipe || { \
            echo "" >&2; \
            echo "ERROR: no aarch64 mediapipe wheel is available from PyPI." >&2; \
            echo "Rebuild with --build-arg MEDIAPIPE_WHEEL=<url-or-path>." >&2; \
            echo "See jetson/README.md for where to get one." >&2; \
            exit 1; \
        }; \
    fi

RUN python3 -m pip install --no-cache-dir "PyYAML>=6.0"

WORKDIR /app
COPY . .

RUN mkdir -p /var/lib/kinect_gaze
VOLUME ["/var/lib/kinect_gaze"]

# Headless by default: an installed unit has no monitor. Calibration is a
# separate, one-off run with a display attached (see docker-compose.yml).
CMD ["python3", "demo.py", "--config", "config.example.yaml", "--headless"]
