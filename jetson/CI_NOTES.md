# Jetson manual verification

GitHub Actions runners are x86_64 for the test matrix, so CI covers the pure
logic only — which is deliberately most of it: `attention`, `calibration`,
`config` and `gaze` need nothing but numpy. The `jetson-image` job does build
the aarch64 dependency set on a native arm64 runner and asserts, at build time,
that the right `cv2` wins, that it has GStreamer, that MediaPipe can build a
FaceMesh graph and that pyrealsense2 can construct a context. Everything below
is what still needs a human and a board.

## Status

Checked on a Jetson Orin Nano Super dev kit, L4T r36.4.4 (JetPack 6.2),
Ubuntu 22.04, Python 3.10.12, on 2026-09-12:

| Section | Result |
| --- | --- |
| 1. Imports and backends | **pass** |
| 2. Camera | **pass** — D435i, fw 5.17.0.10, USB 3.2, 26.1 fps through `RealSenseSource` |
| 3. Test suite | **pass** — 103 passed, 0 skipped |
| 4. Calibration round trip | **pass** — four corners accepted over ssh, `calib_z` 0.649 m |
| 5. Escalation, by hand | **partial** — warn/alert timings confirmed; 3 rows outstanding |
| 6. Performance capture | **partial** — 22.8 fps end to end measured; no `tegrastats` capture |

Two findings came out of §2 and are now pinned in the dependency files. Both
were silent failures rather than obvious ones:

* **pyrealsense2 must be >= 2.57.** The L4T kernel has `CONFIG_HID_SENSOR_HUB`
  unset, so the D435i's IMU never appears, and up to 2.56.5 librealsense throws
  `bad optional access` out of device creation rather than degrading to
  "no IMU". The camera enumerates and then cannot be opened. See
  [`README.md`](README.md#the-imu-and-why-the-version-floor-exists).
* **The wheels use the V4L2 backend, not RSUSB.** They need `/dev/video*`, not
  just `/dev/bus/usb`, which is why the compose file bind-mounts the whole of
  `/dev`.

§4 was done through the published image over ssh, with no display anywhere:

```
docker compose --profile calibrate-tui run --rm calibrate-tui
Calibration: Top-Left accepted ... Bottom-Right accepted
Calibration saved to /var/lib/gaze_monitor/calibration.json
```

The stored `calib_z` is 0.649 m, against ~0.67 m measured independently from
the iris landmark depth — so depth is being read correctly and distance
compensation has a sane anchor.

Note when reading a stored zone that `dx_min` > `dx_max` is normal, not a
corrupt record: the gaze axis runs opposite to the screen axis, and
`AttentionZone.project` divides by the signed span rather than special-casing
it.

Of §5, the escalation timings are confirmed from the transition log: warning
at `away 1.0s` against `warn_after: 1.0`, alert at `away 3.0s` against
`alert_after: 3.0`. Three rows are still outstanding and all need a person:
the brief-glance-back hysteresis, `SENSOR FAULT` on a covered lens, and
stepping back a metre.

The image itself has not been run on the board — CI builds and publishes it,
and nothing here has exercised it. Everything in §2 was verified against the
host install, so the image inherits the dependency findings but not the proof.

## 1. Imports and backends

```bash
python3 -c "import cv2, numpy, yaml; print('cv2', cv2.__version__)"
# pyrealsense2 exposes no __version__; constructing a context is the real check.
python3 -c "import pyrealsense2 as rs; rs.context(); print('pyrealsense2 ok')"
python3 -c "import mediapipe; print('mediapipe', mediapipe.__version__)"
python3 -c "from gaze_monitor import capture; print('capture ok')"
```

Both hardware wheels install from PyPI on aarch64 at the pinned versions — no
librealsense source build, no externally supplied MediaPipe wheel. See
[`README.md`](README.md) for the version constraints that make that true, and
check them before bumping either pin.

The `cv2` that answers must be the apt one, and it must have GStreamer (needed
for the `jetson` and `gst:` sources):

```bash
python3 -c "import cv2; print(cv2.__file__)"   # want /usr/lib/python3.10/dist-packages
python3 -c "from gaze_monitor.capture import has_gstreamer_support as g; print(g())"
```

A path under `~/.local` or `/usr/local` means a pip OpenCV is shadowing the apt
build, and `has_gstreamer_support()` will say `False`. Verified both ways round
on the board: apt OpenCV 4.8.0 reports `GStreamer: YES (1.20.3)` and `True`;
`opencv-python` 4.13 and `opencv-contrib-python` 4.11 from PyPI report `NO` and
`False`.

Both failure messages were confirmed on the board, with no camera attached and
with a GStreamer-less OpenCV respectively:

```
CameraError: Could not start the RealSense pipeline at 640x480@30: No device
connected. Check the camera is on USB 3 (a USB 2 link cannot sustain this
mode) and that no other process holds the device.

CameraError: Could not open GStreamer pipeline. This OpenCV build has no
GStreamer support -- the PyPI wheels are built without it. Use the
distribution package (apt install python3-opencv) or the container image.
```

## 2. Camera

RealSense enumeration, before involving this project at all:

```bash
rs-enumerate-devices -s
python3 -c "
import pyrealsense2 as rs
d = rs.context().query_devices()[0]
print(d.get_info(rs.camera_info.name),
      d.get_info(rs.camera_info.firmware_version),
      d.get_info(rs.camera_info.usb_type_descriptor))
print([d.sensors[i].get_info(rs.camera_info.name) for i in range(len(d.sensors))])
"
```

Expect `usb_type_descriptor` to be `3.2`; a `2.1` here means the camera has
negotiated USB 2 and cannot sustain 640x480x30. Expect exactly two sensors,
`['Stereo Module', 'RGB Camera']` — the Motion Module is absent on this kernel
and that is not a fault.

Then through the project's own backend, which is what proves the metres
convention in [`capture.py`](../gaze_monitor/capture.py):

```bash
python3 -c "
from gaze_monitor.capture import create_source
src = create_source('realsense', width=640, height=480, fps=30); src.start()
f = [src.read() for _ in range(30)][-1]
d = f.depth_m
print(f.color.shape, d.dtype, 'valid %.0f%%' % (100 * (d > 0).mean()))
src.stop()
"
```

Measured on the board: 26.1 fps, no transient `None`s over 90 frames, 94% of
depth pixels valid, depth already `float32` metres.

CSI pipeline, if you are using one:

```bash
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1' ! \
  nvvidconv ! videoconvert ! autovideosink
```

## 3. The test suite

It runs on the board as-is, and the OpenCV-dependent tests will no longer skip:

```bash
python -m pytest -q
```

## 4. Calibration round trip

The part CI cannot reach at all.

```bash
python3 demo.py --tui --recalibrate --config config.example.yaml  # four corners
python3 demo.py --headless --config config.example.yaml           # must reuse it
cat "${GAZE_CALIBRATION_FILE:-$HOME/.local/share/gaze_monitor/calibration.json}"
```

Drop `--tui` if you have a display attached. Either way the corners are the
machine's, not the screen's.

Check that the stored `calib_z` is roughly the operator's real distance from
the camera in metres. A wildly wrong value means depth is not being read
correctly, and distance compensation will then work against you.

## 5. Escalation, by hand

With the monitor running, confirm each transition and roughly its timing.
Over ssh, run it with `--tui`: the status line names the state and shows
`away_seconds` live, which is easier to time against than log lines.

```bash
python3 demo.py --tui --source realsense --config config.example.yaml
```


| Action | Expected |
| --- | --- |
| Look at the blade area | `WATCHING BLADE`, green |
| Look away, hold | amber at ~`warn_after`, red at ~`alert_after` |
| Look back briefly, away again | stays red — hysteresis, does not flicker clear |
| Look back and hold | clears after ~`clear_after` |
| Cover the lens | `SENSOR FAULT` after ~`fault_after` |
| Step back a metre and repeat | zone edges hold (depth compensation) |

The last row is the one worth being fussy about. The arithmetic has been
checked against a real stored calibration: with compensation on, a sample
representing the same physical point from 1 m further back projects to
identical zone coordinates (centre stays u=0.500 v=0.500, edge stays 0.000);
with it off the same sample reads u=-0.216 v=+1.726 and falls outside. So the
sign and the magnitude are right.

What that does **not** check is the physical model behind it, which assumes
the camera sits at the area being watched, so that stepping back increases the
eye-to-camera and eye-to-target distances together. If the camera is mounted
well away from the danger zone that assumption breaks, and only the physical
test will show it.

## 6. Performance capture

```bash
sudo tegrastats --interval 1000 > tegrastats.log &
python3 demo.py --source realsense --config config.example.yaml --debug
```

Attach `tegrastats.log`, the demo's stdout/stderr, and your JetPack/L4T version
and board model to any issue or PR.
