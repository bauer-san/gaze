# Kinect Gaze Demo

`demo.py` is the canonical demo for the gaze-detection use-case. Experimental scripts are archived in `archive/` and documented in `archive/README.md`.

Quick start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install runtime dependencies (pinned in `requirements.txt`):

```bash
pip install -r requirements.txt
```

3. Run the demo (webcam by default):

```bash
python3 demo.py --config config.example.yaml
```

CLI flags override values in the config file. Example flags:

```bash
python3 demo.py --config config.yaml --source kinect --width 1920 --height 1080 --alpha 0.15 --fullscreen
```

Configuration

- A sample config is provided as `config.example.yaml`. Create `config.yaml` to customize defaults; CLI options take precedence.

Development

- Install developer tools (linters & test runner):

```bash
pip install -r dev-requirements.txt
```

- Run tests:

```bash
python -m pytest -q
```

- Run formatters/linters (project excludes `archive/` via `pyproject.toml`):

```bash
black .
ruff check .
flake8 . --exclude=archive,.venv,venv,__pycache__ --max-line-length=88
```

Repository & CI

- CI workflow: `.github/workflows/ci.yml` runs linters and tests on push/PR.
- The project is hosted at: https://github.com/bauer-san/gaze.git

Notes

- Several runtime packages (e.g., `freenect`, `pyrealsense2`, `open3d`) require OS-level libraries — follow each project's install instructions if pip install fails.
- The `kinect_gaze/` package contains modular helpers (`capture`, `gaze`, `calibration`, `ui`) so the demo can be reused or extended.

Contributing

- If you plan to re-enable or salvage code from `archive/`, add a short note in that file explaining the intent before changing it.

Questions or changes you want copied from the archived `legacy_14test.py`? Open an issue or ask here.
