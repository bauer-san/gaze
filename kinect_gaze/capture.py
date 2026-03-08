"""Camera abstraction: supports webcam fallback and Kinect (freenect) when available."""

from typing import Tuple, Optional
import numpy as np

try:
    import freenect
except Exception:
    freenect = None

import cv2


class CameraSource:
    def __init__(self, source: str = "webcam"):
        self.source = source
        self.cap = None

    def start(self):
        if self.source == "kinect":
            if freenect is None:
                raise RuntimeError(
                    "freenect not available; install libfreenect or use --source webcam"
                )
            # Kinect uses freenect sync calls; nothing to open here
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Unable to open webcam")

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (bgr_frame, depth) where depth may be None for webcam."""
        if self.source == "kinect":
            depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
            rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return frame, depth
        else:
            ret, frame = self.cap.read()
            return (frame, None) if ret else (None, None)

    def stop(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
