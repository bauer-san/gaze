# Jetson manual verification

GitHub Actions runners are x86_64 and cannot exercise Jetson hardware, so CI
covers the pure logic only — which is deliberately most of it: `attention`,
`calibration`, `config` and `gaze` need nothing but numpy. Everything below is
what a human has to check on the board.

## 1. Imports and backends

```bash
python3 -c "import cv2, numpy, yaml; print('cv2', cv2.__version__)"
# pyrealsense2 exposes no __version__; constructing a context is the real check.
python3 -c "import pyrealsense2 as rs; rs.context(); print('pyrealsense2 ok')"
python3 -c "import mediapipe; print('mediapipe', mediapipe.__version__)"
python3 -c "from gaze_monitor import capture; print('capture ok')"
```

GStreamer support in the OpenCV build (needed for the `jetson` and `gst:`
sources):

```bash
python3 -c "import cv2; print('GStreamer:' in cv2.getBuildInformation())"
python3 -c "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if 'GStreamer' in l])"
```

## 2. Camera

RealSense enumeration, before involving this project at all:

```bash
rs-enumerate-devices -s
```

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
python3 demo.py --recalibrate --config config.example.yaml   # mark four corners
python3 demo.py --headless --config config.example.yaml      # must reuse it
cat "${GAZE_CALIBRATION_FILE:-$HOME/.local/share/gaze_monitor/calibration.json}"
```

Check that the stored `calib_z` is roughly the operator's real distance from
the camera in metres. A wildly wrong value means depth is not being read
correctly, and distance compensation will then work against you.

## 5. Escalation, by hand

With the monitor running, confirm each transition and roughly its timing:

| Action | Expected |
| --- | --- |
| Look at the blade area | `WATCHING BLADE`, green |
| Look away, hold | amber at ~`warn_after`, red at ~`alert_after` |
| Look back briefly, away again | stays red — hysteresis, does not flicker clear |
| Look back and hold | clears after ~`clear_after` |
| Cover the lens | `SENSOR FAULT` after ~`fault_after` |
| Step back a metre and repeat | zone edges hold (depth compensation) |

The last row is the one worth being fussy about: it is the only check that
depth is being applied in the right direction.

## 6. Performance capture

```bash
sudo tegrastats --interval 1000 > tegrastats.log &
python3 demo.py --source realsense --config config.example.yaml --debug
```

Attach `tegrastats.log`, the demo's stdout/stderr, and your JetPack/L4T version
and board model to any issue or PR.
