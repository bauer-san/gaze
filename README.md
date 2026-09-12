# Operator Attention Monitor

A camera-based demonstrator that alerts the operator of a meat saw when their
attention leaves the area around the cutting blade.

A RealSense camera watches the operator's face. On commissioning, the operator
marks the four corners of the region they need to be watching — the blade and
the cut immediately around it. From then on the system tracks where they are
looking, and escalates when their gaze leaves that region:

| State | Meaning |
| --- | --- |
| `ATTENTIVE` | Gaze is inside the attention area |
| `WARNING` | Gaze has been outside it for `warn_after` seconds |
| `ALERT` | Gaze has been outside it for `alert_after` seconds |
| `FAULT` | The camera has stopped producing usable frames |

## What this is not

**This is an attention aid, not a machine guard.** It has no guaranteed
failure detection, no rated response time, and no interlock to the blade. Do
not use it as, or in place of, a guard.

Machine guarding for a band saw is a functional-safety design — ISO 13849 /
IEC 62061 — built from rated components with defined diagnostic coverage. A
webcam, a neural landmark model and a Python loop are none of those things.
This system is useful as a *supplementary* alert layer on top of proper
guarding, and the `FAULT` state exists so that it tells you when it has
stopped working rather than sitting there looking healthy.

## Hardware

| | |
| --- | --- |
| Camera | Intel RealSense D435i |
| Compute | NVIDIA Jetson Orin Nano Super dev kit (JetPack 6 / L4T r36) |

A plain webcam works for development, with no depth and so no distance
compensation. Kinect v1 is supported as legacy. See
[`jetson/README.md`](jetson/README.md) for the deployment notes.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-realsense.txt
python3 demo.py --config config.example.yaml
```

The first run has no stored calibration, so it starts the commissioning
routine. Without a RealSense to hand:

```bash
python3 demo.py --source webcam --config config.example.yaml
```

## Commissioning

Calibration happens **once**, not per shift. Look at each corner of the area
to be watched and press `c`; the fixation is rejected if you were not holding
still, or if too few frames saw your eyes. After the fourth corner the
calibration is written to disk and reused on every subsequent start.

| Key | |
| --- | --- |
| `c` | Capture the highlighted corner |
| `r` | Start over |
| `q` / `Esc` | Quit |

Calibration is stored at `~/.local/share/gaze_monitor/calibration.json`,
overridable with `--calibration-file` or the `GAZE_CALIBRATION_FILE`
environment variable (which is how the container reaches a mounted volume).

```bash
python3 demo.py --recalibrate      # redefine the area
python3 demo.py --factory-reset    # discard it and exit
```

The record is versioned. A file from a different version, or a corrupt one, is
refused and logged, and the system asks to be recalibrated — it will not
reinterpret old numbers, because that would move the danger zone silently.

## Docker

```bash
# x86_64 development, with the calibration window on the host display
xhost +local:docker
docker compose up gaze
```

On the Jetson there is nothing to build: CI builds the image on an arm64
runner after the tests pass and publishes it to GHCR, and the board pulls it.

```bash
docker compose --profile jetson pull gaze-jetson        # ghcr.io/bauer-san/gaze:jetson

docker compose --profile calibrate run --rm calibrate   # once, with a display
docker compose --profile jetson up -d gaze-jetson       # headless from then on
```

Calibration lives in the `gaze-calibration` volume shared by all three
services, so the unit is commissioned once and every later start reuses it.
Headless mode annunciates through the log and **requires** a stored
calibration — it cannot run the calibration UI.

## Configuration

[`config.example.yaml`](config.example.yaml) documents every setting and the
reasoning behind the defaults. Precedence is **CLI flag > config file >
built-in default**. `python3 demo.py --help` lists the flags.

The thresholds worth thinking about for a given installation:

| Setting | Default | |
| --- | --- | --- |
| `warn_after` | 1.0 s | Grace period — glancing at the feed table is normal |
| `alert_after` | 3.0 s | When the operator has genuinely lost the blade |
| `clear_after` | 0.4 s | Time back on target before an alarm clears |
| `fault_after` | 1.0 s | Unusable frames before declaring a sensor fault |
| `zone_margin` | 0.1 | Tolerance around the calibrated area |

## Layout

| Module | |
| --- | --- |
| [`attention.py`](gaze_monitor/attention.py) | The attention area and the escalation state machine. Pure logic, injected time. |
| [`calibration.py`](gaze_monitor/calibration.py) | Four-corner collection and the versioned on-disk record |
| [`capture.py`](gaze_monitor/capture.py) | Camera backends. Normalises all depth to **metres** at the source. |
| [`config.py`](gaze_monitor/config.py) | Defaults, YAML loading, CLI merge, validation |
| [`gaze.py`](gaze_monitor/gaze.py) | Iris geometry and smoothing |
| [`ui.py`](gaze_monitor/ui.py) | The run loop and rendering |

`attention`, `calibration`, `config` and `gaze` import nothing heavier than
numpy, so the logic that decides whether to alarm is tested without a camera,
OpenCV or MediaPipe.

Earlier experiments are in [`archive/`](archive/README.md); they are excluded
from linting and tests.

## Development

```bash
pip install -r dev-requirements.txt
python -m pytest -q
black . && ruff check . && flake8 .
```

CI runs the same three linters and the tests on Python 3.10 and 3.12. Tests
that need OpenCV skip themselves when it is absent.

## Extending

`AttentionMonitor` takes an `on_transition` hook, called once per state change.
That is the single place to attach a real annunciator — a beacon, a sounder, a
relay, an MQTT publish:

```python
def annunciate(old, new, status):
    if new.is_alarming:
        gpio.output(BEACON_PIN, 1)
    else:
        gpio.output(BEACON_PIN, 0)

monitor = AttentionMonitor(zone=zone, on_transition=annunciate)
```
