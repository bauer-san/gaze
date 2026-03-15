# Jetson manual testing and CI notes

Because GitHub Actions runners are x86 and cannot run Jetson-specific hardware tests, use these steps to perform manual validation and capture artifacts for CI artifacts or issue reports.

1. Sanity checks

- Verify Python imports and capture backends:

```bash
python3 -c "import cv2, numpy, yaml; print('cv2', cv2.__version__)"
python3 -c "python -c 'from kinect_gaze import capture; print("capture ok")'"
```

2. Camera pipeline test

- Test the Jetson pipeline (CSI):

```bash
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1' ! nvvidconv ! videoconvert ! autovideosink
```

3. Inference perf capture

- Run the demo and collect `tegrastats` output in the background:

```bash
sudo tegrastats --interval 1000 > tegrastats.log &
python3 demo.py --source jetson --config config.example.yaml
```

4. Attach logs to PRs or issues

- Include `tegrastats.log`, demo stdout/stderr, and a short description of JetPack/L4T and board model.
