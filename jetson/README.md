# Jetson Orin Nano (Jetson) setup guide

This document collects pragmatic steps and notes to run the Kinect Gaze demo on NVIDIA Jetson devices (Orin Nano, Xavier, etc.). It is not exhaustive — JetPack, L4T, and driver versions matter. Follow the high-level steps below and adapt versions for your board.

Prerequisites

- JetPack installed (match your board): check `nvcc --version` and `jetson_release` output.
- Sufficient swap or build space for compiling some packages (MediaPipe / Open3D may need source builds).
- NVIDIA Container Toolkit if you prefer containers.

Recommended approach

1. Use the JetPack-provided Python and apt packages where possible — many python wheels on PyPI are x86-only.

2. Install system packages before pip:

```bash
sudo apt update
sudo apt install -y python3-opencv libopencv-dev libblas-dev liblapack-dev libjpeg-dev libtiff-dev gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good v4l-utils
```

3. Camera & capture

- For CSI cameras on Jetson, use `nvarguscamerasrc` (we provide a `jetson` pipeline in `kinect_gaze.capture`).
- For USB webcams, prefer V4L2/GStreamer pipelines (`v4l2src`) or the distro `python3-opencv` package which includes GStreamer support.

4. Python dependencies

- Use `requirements-jetson.txt` for guidance (project root). Some packages must be installed from apt or built from source:
  - `opencv-python`: DO NOT pip install the standard wheel — instead use `sudo apt install python3-opencv` or build OpenCV with CUDA.
  - `mediapipe`: MediaPipe wheels are not always available for Jetson; consider building from source or using an optimized ONNX/TensorRT model.
  - `pyrealsense2`: follow Intel RealSense Jetson instructions or use pip wheels provided by librealsense where available.

5. Inference optimization

- Convert heavy models to ONNX and then TensorRT (FP16) for the best performance.
- Use OpenCV DNN with the TensorRT backend, or use TensorRT Python bindings directly.
- See `jetson/optimizations.md` for concrete tips.

6. Container approach

- If you prefer reproducible setups, use an NVIDIA L4T/JetPack base image and install apt packages there. See `Dockerfile.jetson` for a starting example.

7. Manual testing & CI notes

- CI cannot run Jetson-specific tests in standard GitHub Actions. See `jetson/CI_NOTES.md` for recommended manual checks and how to capture logs for CI artifacts.

If you want, I can: (1) add a helper script to probe available camera backends at runtime, (2) add a simple ONNX->TensorRT conversion example, or (3) create a Jetson-tailored Dockerfile that installs and builds MediaPipe. Tell me which you prefer next.
