# Jetson Orin Nano Super deployment

Notes for running the attention monitor on the Jetson Orin Nano Super dev kit
with an Intel RealSense D435i. JetPack 6 / L4T r36 is assumed; check yours with
`cat /etc/nv_tegra_release` or `jetson_release`.

The container route in [`../Dockerfile`](../Dockerfile) and
[`../docker-compose.yml`](../docker-compose.yml) is the recommended path — it
handles the two genuinely awkward dependencies for you. What follows explains
what it is doing, and how to do it by hand if you would rather not use Docker.

## The two hard dependencies

Everything else installs normally. These two do not:

### librealsense / pyrealsense2

Intel publishes no aarch64 wheels for most releases, so it must be built from
source with the Python bindings enabled:

```bash
git clone --depth 1 --branch v2.56.5 https://github.com/IntelRealSense/librealsense
cmake -S librealsense -B librealsense/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=true \
    -DBUILD_PYTHON_BINDINGS=true \
    -DBUILD_EXAMPLES=false \
    -DPYTHON_EXECUTABLE=/usr/bin/python3
cmake --build librealsense/build --parallel "$(nproc)"
sudo cmake --install librealsense/build && sudo ldconfig
```

`FORCE_RSUSB_BACKEND=true` matters. It makes librealsense talk to the camera
over libusb rather than through patched kernel modules, which is both easier on
JetPack and the only option inside a container, since a container cannot load
kernel modules.

Check it with `python3 -c "import pyrealsense2 as rs; rs.context(); print('ok')"`
(the module exposes no `__version__`). If the import fails, `PYTHONPATH`
probably needs `/usr/local/lib/python3.10/dist-packages`.

### MediaPipe

There is no aarch64 MediaPipe wheel on PyPI. You need one built for JetPack —
from the [jetson-containers](https://github.com/dusty-nv/jetson-containers)
project, another community build, or your own source build. Then:

```bash
docker compose --profile jetson build gaze-jetson \
    --build-arg MEDIAPIPE_WHEEL=<url or path>
```

The image build fails loudly if no wheel is supplied, rather than producing
something that looks fine and dies on first frame.

## OpenCV

Use the distribution package, never the PyPI wheel:

```bash
sudo apt install -y python3-opencv
```

The apt build is compiled for this architecture and, importantly, has
GStreamer support — which the PyPI wheels do not. Without it the `jetson` and
`gst:` camera sources cannot open a pipeline, and
[`capture.py`](../gaze_monitor/capture.py) will say so explicitly rather than
failing obscurely.

Do not install `opencv-python` or `opencv-contrib-python` alongside it: two
OpenCV distributions in one environment install competing copies of `cv2`.

## Cameras

The D435i is the supported camera and needs nothing beyond librealsense above:

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

## Running as an appliance

Commission once with a display attached, then run headless:

```bash
docker compose --profile calibrate run --rm calibrate
docker compose --profile jetson up -d gaze-jetson
```

Calibration persists in the `gaze-calibration` volume. Headless mode refuses to
start without it rather than silently monitoring an undefined area.

The service is `restart: unless-stopped` on purpose: an attention monitor that
has exited looks exactly like one that is quiet because nothing is wrong.

## Performance

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
