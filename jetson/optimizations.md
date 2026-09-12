# Jetson performance & optimization notes

This document collects practical optimizations for running inference and OpenCV pipelines on Jetson devices.

1) Use TensorRT (preferred)

- Export your model to ONNX, then convert to TensorRT with `trtexec` or the Python TensorRT API.
- Build with FP16 to reduce memory and increase throughput. Example:

```bash
# run on-host trtexec example
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=1024
```

2) Use OpenCV DNN with TensorRT backend

```python
net = cv2.dnn.readNetFromONNX('model.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
```

3) Prefer GStreamer pipelines for low-latency capture

- Use `nvarguscamerasrc` for CSI cameras and `v4l2src` for USB devices.
- Use `appsink` to feed frames into your OpenCV pipeline as in `gaze_monitor.capture`.

4) Reduce model input size and frame rate

- Downscale frames where possible, and run inference at lower FPS while using a tracker between inferences.

5) Profile

- Use `tegrastats` and `jetson_stats` to monitor CPU/GPU/memory usage while tuning.

6) Memory & threading

- Reuse preallocated buffers and avoid unnecessary copies.
- Pin worker threads to cores if you need deterministic latency.
