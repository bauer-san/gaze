# Kinect Gaze Demo

`demo.py` is the canonical demo for the gaze-detection use-case. The most recent experimental script has been archived as `archive/legacy_14test.py`; older experiments are in `archive/` for reference.

Quick start

1. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the demo (webcam by default):

```bash
python3 demo.py --source webcam
```

Use `--source kinect` to run with a Kinect (requires `libfreenect` and the `freenect` Python package), or add support for RealSense if desired.

Development

- Run tests:

```bash
/home/geoff/Documents/kinect/.venv/bin/python -m pytest -q
```

- Run formatters/linters locally (we exclude `archive/`):

```bash
/home/geoff/Documents/kinect/.venv/bin/black .
/home/geoff/Documents/kinect/.venv/bin/ruff check .
/home/geoff/Documents/kinect/.venv/bin/flake8 . --exclude=archive,.venv,venv,__pycache__ --max-line-length=88
```

CI

- A GitHub Actions workflow is included at `.github/workflows/ci.yml` to run linters and tests on push/PR.

Notes

- Some packages (e.g., `freenect`, `pyrealsense2`, `open3d`) require system libraries; follow their install docs if pip install fails.
- The `kinect_gaze/` package contains modular helpers: `capture`, `gaze`, `calibration`, and `ui`.