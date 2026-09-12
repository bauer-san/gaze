# Archived experiments

Exploratory scripts kept for reference. **None of these run as part of the
project** and none are linted or tested — `pyproject.toml`, `.flake8` and CI
all exclude this directory.

The working code lives in [`kinect_gaze/`](../kinect_gaze/) and is run through
[`demo.py`](../demo.py).

## Why they are still here

They are the record of how the gaze estimate was arrived at, and a couple of
them contain behaviour that the package took a while to catch up with. If you
are wondering why something in the package works the way it does, the answer is
often in one of these.

| Script | What it explored |
| --- | --- |
| `test.py` … `test9.py` | Incremental Kinect capture and MediaPipe iris experiments |
| `10test.py` – `13test.py` | Four-corner calibration, depth compensation, multi-camera source selection |
| `legacy_14test.py` | The furthest-developed prototype: calibration class, RealSense + Kinect + webcam sources |
| `11test.py` | **The attention/alert logic**: `USER_TIMEOUT`, out-of-bounds detection, green/amber/red escalation |
| `kinect.py`, `new.py`, `newer.py`, `evennewer.py`, `ocd.py` | Small capture and display spikes |

`11test.py` is the notable one. The escalation behaviour it prototyped was
dropped in the refactor that created the package, and has since been
reimplemented — with hysteresis, a fault state and tests — in
[`kinect_gaze/attention.py`](../kinect_gaze/attention.py).

## If you want to salvage something

Copy the behaviour into the package with a test, rather than reviving the
script. These files assume module-level global state, a fixed 1920×1080
display, and hardware being present at import time; none of that survives
contact with the current structure.
