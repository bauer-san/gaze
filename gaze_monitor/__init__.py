"""Camera-based operator attention monitoring for machinery danger zones.

Submodules are imported lazily by name rather than re-exported here, so that
the pure-logic modules (attention, calibration, config, gaze) can be imported
and tested without OpenCV or MediaPipe present.
"""

__version__ = "0.3.0"

__all__ = ["attention", "calibration", "capture", "config", "gaze", "ui"]
