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
Ubuntu 22.04, Python 3.10.12, on 2026-09-12 and 2026-09-13:

| Section | Result |
| --- | --- |
| 1. Imports and backends | **pass** |
| 2. Camera | **pass** — D435i, fw 5.17.0.10, USB 3.2, 26.1 fps through `RealSenseSource` |
| 3. Test suite | **pass** — 144 passed, 0 skipped |
| 4. Calibration round trip | **pass** — four corners accepted over ssh, `calib_z` 0.649 m |
| 5. Escalation, by hand | **partial** — timings and sensor fault confirmed; 2 rows outstanding |
| 6. Performance capture | **pass** — 22.8 fps end to end; ~10.5% CPU, GR3D idle, 55 degC |

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
`alert_after: 3.0`. Two rows are still outstanding and both need a person:
the brief-glance-back hysteresis, and stepping back a metre.

### The covered-lens row, and what it used to get wrong

This row asked for `SENSOR FAULT` and the monitor gave `ATTENTION LOST`. The
monitor was right and the row was wrong: `fault_after` counts *unusable
frames*, and a covered lens produces perfectly good frames that merely have
nothing in them. `read()` returned `None` zero times in 60 obstructed frames.
Not being able to see the operator is treated as looking away, deliberately
and pessimistically -- so the alarm did sound, just under the wrong name.

A blocked-sensor check now closes that gap: no depth returns **and** no face
counts as unusable and feeds the same `fault_after` timer. The conjunction is
load-bearing. Depth alone would fault a camera whose depth sensor died while
colour still tracked faces perfectly well, which is a working monitor, not a
broken one. Cameras with no depth at all skip the check.

Verified so far:

* the two populations are far apart -- 88.3-98.1% valid depth with the lens
  clear against 0.0-1.2% covered, measured over 90 and 60 frames;
* the fault path fires on hardware, forced by running with the threshold
  inverted (`--min-depth-fraction 0.99`), including recovery the instant a
  face appears, which is the conjunction doing its job;
* the thresholds are pinned in `tests/test_quality.py` against those numbers.

Confirmed on the board at the shipping threshold, against the published
image (`MIN_DEPTH_FRACTION` 0.05, not an inverted test value), with paper
over the lens:

```
12:57:16 WARNING attention alert -> fault (away 19.4s, tracked=False)
12:57:24 WARNING attention fault -> alert (away 26.9s, tracked=True)
```

Note the recovery line. It returns to `alert` rather than to `attentive`
because the operator's gaze was still outside the zone at that instant --
uncovering the lens restores the camera, not the operator's attention. The
two are separate conditions and the monitor does not conflate them.

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

Board-level resource use is already collected continuously by the nv-monitor
scrape behind the Grafana "Orin Fleet Monitor" dashboard, per device. For
anything that runs longer than a few seconds, read it there rather than
capturing by hand: it has history, so a run can be compared against the
board's idle baseline instead of against nothing.

Container logs are in the board's local time (the compose services mount the
host's `/etc/localtime`), so a timestamp in the demo's output lines up with
the dashboard's axis directly, with no UTC arithmetic in between.

**The scrape is not fast.** It samples on an interval that will miss a short
spike entirely, so it answers "what does this cost while it runs" and not
"what was the worst instant". For anything brief -- start-up, a single
calibration fixation, a fault-and-recover -- take the fine-grained capture
instead, and correlate the two:

```bash
sudo tegrastats --interval 100 > tegrastats.log &
python3 demo.py --source realsense --config config.example.yaml --debug
```

Measured for a commissioning run plus a few minutes of monitoring
(640x480x30, one face in frame, MAXN_SUPER):

| | |
| --- | --- |
| CPU | ~10.5% peak, against a 1-2% idle baseline |
| GR3D (GPU) | flat at zero for the whole run |
| NVDEC / NVENC | flat at zero |
| Temperature | 54-55 degC, no rise over the run |

GR3D staying at zero is the useful one: it is the direct confirmation that
MediaPipe is running on the CPU through XNNPACK and that the GPU is entirely
idle. That is the headroom the TensorRT route in
[`optimizations.md`](optimizations.md) would be spending, and the reason there
is no case for spending it yet. NVDEC/NVENC at zero is expected -- the
RealSense delivers raw frames and no codec is in the path.

Two things not to be misled by when reading that dashboard. Its "GPU Percent"
panel disagreed with "GR3D Utilization" during this run (about 5% against
zero at the same instant), so at least one of them is not plotting GPU
utilisation; GR3D is the one that matches what the pipeline actually does.
And the temperature trace dithering between two adjacent values is sensor
quantisation, not thermal cycling -- nothing on a passively cooled board
oscillates at that rate.

Attach the demo's stdout/stderr, the dashboard window (or `tegrastats.log`),
and your JetPack/L4T version and board model to any issue or PR.
