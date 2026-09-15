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

## Try it before you buy anything

The whole behaviour — calibration, tracking, escalation, metrics — runs on any
laptop with a webcam. No special hardware, about five minutes:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 demo.py --source webcam --config config.example.yaml
```

There is no stored calibration on a first run, so it starts commissioning:
look at each corner of the area you want watched and press `c`. Then look away
from it and watch the state escalate.

What you will *not* see is distance compensation, because a webcam has no
depth. That is the one thing worth buying hardware for, and the next section
explains why.

## Why a depth camera

Gaze is measured as the iris's displacement within the eye — an angle, not a
point in space. The same eye rotation sweeps a wider area the further away you
are, so a region calibrated as "looking at the blade" only means that at the
distance it was calibrated at. Step back a metre and the operator looking
straight at the blade reads as looking well outside it.

A depth camera fixes this: the measured distance scales the zone, so it holds
as the operator moves. Everything else in the system works without depth, and
the code treats a missing depth reading as "compensation off" rather than as
an error.

## Bill of materials

Four tiers. Each adds one capability to the one above it.

| Tier | Add | Roughly | Gets you |
| --- | --- | --- | --- |
| **0. Try it** | a laptop with a webcam | — | Everything except distance compensation |
| **1. Depth** | Intel RealSense D435i, on a **USB 3** port | $300–400 | The zone holds as the operator moves |
| **2. Appliance** | NVIDIA Jetson Orin Nano Super dev kit | $400 (2026 tariffs) | Headless 24/7 operation. **This is the verified path** |
| **2b. Small form factor PC** | Any x86_64 mini PC with USB 3 — an HP EliteDesk Mini, Intel NUC or similar | $100–200 used | Headless operation, and the easiest route to an industrial fanless box later |
| **2c. Cheapest** | Raspberry Pi 4, 4 GB, **64-bit** Pi OS, active cooling, powered USB 3 hub | $80–120 | Same software, lower power. **Frame rate unverified** |

Prices are rough and worth checking; the part names are the precise thing.

**Tier 2 is what has actually been run.** Measured on the Jetson: 22.8 fps end
to end at 640×480, ~10.5 % CPU, GPU idle, 55 °C. See
[`jetson/CI_NOTES.md`](jetson/CI_NOTES.md) for what has been verified on
hardware and what has not.

**Tier 2b needs no GPU.** MediaPipe runs on the CPU through XNNPACK — on the
Jetson the GPU was measured at flat zero utilisation for an entire run — so
what matters is CPU, and an x86 CPU with AVX2 outruns the Jetson here. A
second-hand mini PC is the cheapest way to a *fast* unit, and it is also the
smoothest path to a plant installation, because fanless IP-rated industrial
x86 box PCs are a commodity while sealed arm SBCs are not.

**Tier 2c builds and its dependencies are proven to import on arm64, but it
has never been run on a Pi.** A Cortex-A72 is materially slower than the
Jetson's A78AE, and nobody has measured the resulting frame rate. If you take
this path, read the `filter_alpha` note under [Configuration](#configuration)
first — it is not a detail.

Two Raspberry Pi traps worth knowing before you start:

* **The OS must be 64-bit.** 32-bit Raspberry Pi OS has no usable wheels for
  either MediaPipe or pyrealsense2, and there is no workaround.
* **Raspberry Pi OS Bookworm ships Python 3.11, which is the one version with
  no aarch64 pyrealsense2 wheel** (3.9, 3.10 and 3.12 all have one). A native
  `pip install` on the stock OS fails for that reason alone.
* **And 3.12 is a trap.** Its wheel exists, resolves and installs, then fails
  at import: every aarch64 pyrealsense2 wheel is tagged `manylinux2014`
  (glibc 2.17) but cp312 actually needs glibc 2.38, and Bookworm has 2.36.
  cp310 needs 2.34 and is fine. The container pins Python 3.10 for that
  reason, which is the main argument for using it rather than fighting the
  host.

A Kinect v1 is supported as legacy and needs `libfreenect` built from source.

## Running it

```bash
# any machine, from source
pip install -r requirements.txt -r requirements-realsense.txt
python3 demo.py --source realsense --config config.example.yaml
```

Images are built by CI and published to GHCR, so an appliance pulls rather
than builds:

```bash
# Jetson Orin Nano
docker compose --profile jetson pull gaze-jetson
docker compose --profile calibrate-tui run --rm calibrate-tui   # once, over ssh
docker compose --profile jetson up -d gaze-jetson

# Raspberry Pi 4
docker compose --profile pi pull gaze-pi
docker compose --profile calibrate-pi run --rm calibrate-pi     # once, over ssh
docker compose --profile pi up -d gaze-pi

# Small form factor PC (HP EliteDesk Mini, NUC, ...)
docker compose --profile x86 pull gaze-x86
docker compose --profile calibrate-x86 run --rm calibrate-x86   # once, over ssh
docker compose --profile x86 up -d gaze-x86

# x86_64 development, calibration window on the host display
xhost +local:docker
docker compose up gaze
```

Calibration lives in the `gaze-calibration` volume shared by every service, so
a unit is commissioned once and every later start reuses it. Headless mode
annunciates through the log and **requires** a stored calibration — it cannot
run the calibration UI.

The Pi image cannot use the CSI Pi Camera. Pinning Python 3.12 means OpenCV
comes from pip, and the pip wheels have no GStreamer, which `libcamerasrc`
needs. The D435i and USB webcams are unaffected. Debian's `python3-opencv`
is not a way out: it targets the system Python 3.11, the version with no
RealSense wheel.

## Commissioning

Calibration happens **once**, not per shift. Look at each corner of the area
to be watched and press `c`; the fixation is rejected if you were not holding
still, or if too few frames saw your eyes. After the fourth corner the
calibration is written to disk and reused on every subsequent start.

The corners are the corners of the **machine's danger area**, not of the
screen. The window is a prompt and a progress indicator, which is why the
whole procedure also works from a terminal.

| Key | |
| --- | --- |
| `c` / Enter | Capture the highlighted corner |
| `r` | Start over |
| `q` / `Esc` | Quit |

### Over ssh

An installed unit has no display, and neither does an ssh session. `--tui`
runs the same two phases — corner prompts, then a live status line — in the
terminal:

```bash
python3 demo.py --tui --recalibrate --config config.example.yaml
python3 demo.py --tui --config config.example.yaml
```

`--headless` is the third option and a different thing: it draws nothing at
all and annunciates purely through the log, which is right for an installed
unit but cannot calibrate, because calibration needs to tell the operator
which corner to look at and to hear back when they are on it. Pick one of the
three; the config rejects combinations that contradict each other.

`--tui` needs a real terminal. Under Docker that means `run -it`, or
`tty: true` plus `stdin_open: true` in compose — it refuses to start
otherwise rather than drawing a status line nobody can see.

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

## Metrics

```bash
python3 demo.py --metrics-port 9091 --config config.example.yaml
```

Prometheus metrics for a unit you cannot see: state, away time, operator
distance against the calibrated distance, frame rate, camera health, and
counters for state transitions and zone re-entries. Disabled unless a port is
given. See [`jetson/README.md`](jetson/README.md#metrics) for why the counters
matter more than the gauges.

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
| `filter_alpha` | 0.12 | Gaze smoothing — **per frame, not per second** |

### filter_alpha depends on your frame rate

Every threshold above is in seconds. `filter_alpha` is not: it is an
exponential moving average applied once per frame, so its time constant is
`1/alpha` frames divided by whatever frame rate you actually get.

| `filter_alpha` | 30 fps | 22.8 fps (Jetson, measured) | 6 fps |
| --- | --- | --- | --- |
| 0.12 (default) | 0.28 s | 0.37 s | **1.39 s** |
| 0.35 | 0.10 s | 0.13 s | 0.48 s |

At the Jetson's measured rate the default sits comfortably inside
`warn_after: 1.0`. On slower hardware it does not: at 6 fps the gaze estimate
is still catching up when the monitor has already decided to warn, which makes
the tuning incoherent rather than merely sluggish. The Pi image raises it to
0.35 for exactly this reason.

Check your actual frame rate before trusting the default — `--tui` shows it
live, or scrape `gaze_frame_rate_fps`.

## Layout

| Module | |
| --- | --- |
| [`attention.py`](gaze_monitor/attention.py) | The attention area and the escalation state machine. Pure logic, injected time. |
| [`calibration.py`](gaze_monitor/calibration.py) | Four-corner collection and the versioned on-disk record |
| [`capture.py`](gaze_monitor/capture.py) | Camera backends. Normalises all depth to **metres** at the source. |
| [`config.py`](gaze_monitor/config.py) | Defaults, YAML loading, CLI merge, validation |
| [`gaze.py`](gaze_monitor/gaze.py) | Iris geometry and smoothing |
| [`metrics.py`](gaze_monitor/metrics.py) | Prometheus exporter. Inert unless a port is set |
| [`quality.py`](gaze_monitor/quality.py) | Whether a frame is usable — catches a blocked lens |
| [`terminal.py`](gaze_monitor/terminal.py) | The `--tui` annunciator, for running over ssh |
| [`ui.py`](gaze_monitor/ui.py) | The run loop and rendering |

`attention`, `calibration`, `config`, `gaze`, `metrics`, `quality` and
`terminal` import nothing heavier than numpy, so the logic that decides
whether to alarm is tested without a camera, OpenCV or MediaPipe. That is why
CI can cover most of this on an x86_64 runner with no vision stack at all.

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
