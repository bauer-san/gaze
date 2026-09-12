# Jetson Orin Nano Super deployment

Notes for running the attention monitor on the Jetson Orin Nano Super dev kit
with an Intel RealSense D435i. JetPack 6 / L4T r36 is assumed; check yours with
`cat /etc/nv_tegra_release` or `jetson_release`. The notes below were verified
on L4T r36.4.4 (JetPack 6.2), Ubuntu 22.04, Python 3.10.12.

The container is the recommended path, and it is built and published for you:
CI builds the `jetson` target on an arm64 runner after the test matrix passes
and pushes it to GHCR (see [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml)).
The board pulls it:

```bash
docker compose --profile jetson pull gaze-jetson
```

Tags are `jetson` (moving, follows `main`) and `jetson-<sha>` (immutable — pin
a commissioned unit to one of these). Set `GAZE_JETSON_IMAGE` to override.

The package inherits the repository's visibility, so for this public
repository it pulls anonymously — verified against `ghcr.io` with no
credentials. If you make the repository private, or GHCR ever publishes the
package private, log the board in once:

```bash
echo <a-personal-access-token-with-read:packages> | \
    docker login ghcr.io -u <your-github-username> --password-stdin
```

What follows explains what the image is doing, and how to do it by hand if you
would rather not use Docker.

## Dependencies

Everything installs from PyPI, including the two that used to need special
handling:

```bash
pip install -r requirements.txt -r requirements-realsense.txt
```

Two version constraints are doing real work there, and both are about aarch64
wheel availability rather than about the code:

* **pyrealsense2** — the pinned `2.58.4.10922` publishes a
  `cp310-cp310-manylinux2014_aarch64` wheel, so librealsense does **not** have
  to be built from source. Not every release does; check
  <https://pypi.org/pypi/pyrealsense2/json> before bumping. **2.57 is a hard
  floor on JetPack** — see [the IMU section](#the-imu-and-why-the-version-floor-exists)
  below, which is not an obvious failure to diagnose. The wheels use the V4L2
  backend, so they need the `/dev/video*` nodes as well as `/dev/bus/usb`.
* **mediapipe** — `0.10.18` is the newest of the 0.10 series with a cp310
  aarch64 wheel. `0.10.20` and `0.10.21` are x86_64/macOS only, so a bump to
  either silently breaks the Jetson. Check
  <https://pypi.org/pypi/mediapipe/json> before bumping. MediaPipe runs its
  graph on the CPU here, through the XNNPACK delegate.

Kinect v1 is still the exception: `libfreenect` has no usable wheel and must be
built from source with its Python bindings.

Check the result:

```bash
python3 -c "import pyrealsense2 as rs; rs.context(); print('pyrealsense2 ok')"
python3 -c "import mediapipe as mp; print('mediapipe', mp.__version__)"
```

`pyrealsense2` exposes no `__version__`, so constructing a context is the real
check. Note that constructing a context is a weaker check than it looks: the
IMU failure below happens *after* it, when the device is opened.

## Camera permissions, outside the container

The container runs as root and needs none of this. Running on the host as an
ordinary user does: the D435i's USB node comes up `root:root 0664`, and
librealsense needs to write to it. The symptom is an enumeration that finds the
camera and then cannot open it.

Install librealsense's own rules, matching the pinned version:

```bash
sudo curl -fsSL -o /etc/udev/rules.d/99-realsense-libusb.rules \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.58.4/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

They set `MODE:="0666", GROUP:="plugdev"` on Intel RealSense product IDs; your
user must be in `plugdev`. Be aware that the same file also applies
`chmod -R 0777` to the IMU's sysfs and device nodes — upstream's choice, and
moot on a kernel without HID sensor support.

## The IMU, and why the version floor exists

The D435i has a BMI085 IMU, exposed over HID and surfaced through the kernel's
`hid-sensor-*` drivers as an iio device. The L4T kernel is built with
`CONFIG_HID_SENSOR_HUB` unset, so there is nothing to load and the IMU never
appears:

```bash
zcat /proc/config.gz | grep HID_SENSOR_HUB   # '# CONFIG_HID_SENSOR_HUB is not set'
ls /sys/bus/iio/devices/                     # empty
ls /sys/bus/hid/devices/                     # 0003:8086:0B3A.0001 -- the HID interface is there
```

Up to and including pyrealsense2 2.56.5, librealsense reacts to the missing HID
interface by throwing out of device creation, so the camera is unusable:

```
DEBUG   (d400-device.cpp:463) The IMU sensor is for PID b3a has been identified as BMI085
WARNING (ds-motion-common.cpp:452) No HID info provided, IMU is disabled
ERROR   (rs.cpp:256) [rs2_create_device( ..., index:0 ) UNKNOWN] bad optional access
RuntimeError: bad optional access
```

2.57 and later log the same warning and carry on with the depth and colour
sensors, which is all this project uses — it reads no motion data. On the
board, a working device reports
`sensors = ['Stereo Module', 'RGB Camera']`, with no Motion Module. That
absence is expected here, not a fault.

Getting the IMU back would mean rebuilding the L4T kernel with
`CONFIG_HID_SENSOR_HUB` and the `hid-sensor-accel-3d` / `hid-sensor-gyro-3d`
drivers. Nothing in this project needs it.

## OpenCV

This is the one dependency to take from the distribution, never from PyPI:

```bash
sudo apt install -y python3-opencv
```

The apt build has GStreamer support, which the PyPI wheels do not. Without it
the `jetson` and `gst:` camera sources cannot open a pipeline, and
[`capture.py`](../gaze_monitor/capture.py) will say so explicitly rather than
failing obscurely:

```
CameraError: Could not open GStreamer pipeline. This OpenCV build has no
GStreamer support -- the PyPI wheels are built without it. Use the
distribution package (apt install python3-opencv) or the container image.
```

The catch: MediaPipe depends on `opencv-contrib-python`, and pip installs it
somewhere that precedes the apt build on `sys.path`. Two OpenCV distributions
in one environment install competing copies of `cv2` and shadow each other, and
the pip one wins. The container resolves this by installing the requirements
and then removing the pip OpenCV, leaving the apt build; a build-time check
fails the image if the wrong `cv2` wins anyway.

Outside the container, check which one you have — a GStreamer-less `cv2` is the
symptom:

```bash
python3 -c "import cv2; print(cv2.__version__, cv2.__file__)"
python3 -c "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if 'GStreamer' in l])"
```

An answer under `~/.local/lib` or `/usr/local/lib` is the pip wheel shadowing
apt's, which lives in `/usr/lib/python3.10/dist-packages`.

## Cameras

The D435i is the supported camera and needs nothing beyond the wheel above:

```bash
python3 demo.py --source realsense --config config.example.yaml
```

For a CSI camera, `--source jetson` builds an `nvarguscamerasrc` pipeline
(see `jetson_csi_pipeline` in [`capture.py`](../gaze_monitor/capture.py)).
`--device N` selects the sensor id. There is no depth from a CSI camera, so
distance compensation is off. For anything else, pass a pipeline directly:

```bash
python3 demo.py --source 'gst:v4l2src device=/dev/video0 ! videoconvert ! appsink'
```

## The image

[`../Dockerfile`](../Dockerfile)'s `jetson` target is built on
`nvcr.io/nvidia/l4t-base`, not `l4t-jetpack`. Nothing in this pipeline touches
CUDA — MediaPipe is on the CPU, OpenCV comes from apt, the D435i is a USB
device — and `l4t-base` is a 276 MB base against `l4t-jetpack`'s 5.6 GB, which
is what lets CI build it at all. `runtime: nvidia` in the compose file is still
required: the NVIDIA container runtime mounts the host's Tegra userspace into
the container, which is where `nvarguscamerasrc` comes from.

If you take the TensorRT route in [`optimizations.md`](optimizations.md) you
will need CUDA in the image, and the base has to go back to
`nvcr.io/nvidia/l4t-jetpack` with `L4T_TAG=r36.4.0` — the newest tag NGC
publishes for that image.

## Running as an appliance

Commission once, then run headless. Over ssh, with no display anywhere:

```bash
docker compose --profile jetson pull gaze-jetson
docker compose --profile calibrate-tui run --rm calibrate-tui
docker compose --profile jetson up -d gaze-jetson
```

The calibration prompts appear in your terminal, one corner at a time; press
`c` or Enter on each. The corners are the machine's, not the screen's, so
nothing is lost by not having one. With the dev kit's HDMI output attached,
`--profile calibrate` runs the same procedure in a window instead.

To watch a running unit over ssh without a display, `--tui` gives a live
status line instead of log lines:

```bash
docker compose run --rm -it gaze-jetson \
    python3 demo.py --config config.example.yaml --source realsense --tui
```

Calibration persists in the `gaze-calibration` volume. Headless mode refuses to
start without it rather than silently monitoring an undefined area.

The service is `restart: unless-stopped` on purpose: an attention monitor that
has exited looks exactly like one that is quiet because nothing is wrong.

## Performance

Measured on the board (L4T r36.4.4, MAXN_SUPER power mode, D435i at
640x480x30, one face in frame): **22.8 fps** end to end — capture, depth/colour
alignment and FaceMesh with `refine_landmarks=True`. The camera alone, with no
inference, runs at 26.1 fps, so MediaPipe is most of the gap and the capture
path is not the bottleneck.

MediaPipe FaceMesh with `refine_landmarks=True` is the cost in this pipeline.
If the frame rate is not adequate:

* Drop the capture resolution first (`camera_width`/`camera_height`). Iris
  landmarks do not need 1080p; 640×480 is the default for that reason.
* Put the board in its highest power mode: `sudo nvpmodel -m 0` then
  `sudo jetson_clocks`.
* Profile before optimising — `tegrastats` shows whether you are CPU-bound
  (MediaPipe runs on CPU by default) or elsewhere.

See [`optimizations.md`](optimizations.md) for the TensorRT route if you decide
to replace MediaPipe with your own landmark model.

Frame rate is visible in the status bar and in `--debug` logs, so you can see
what the board is actually managing before changing anything.
