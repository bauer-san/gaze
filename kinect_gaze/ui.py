"""Simple monitoring UI that ties capture, gaze math, and calibration together.

This UI is intentionally minimal and uses MediaPipe + OpenCV when available.
"""

import cv2
import numpy as np

from .calibration import GazeCalibrator
from .capture import create_source
from .gaze import GazeFilter, gaze_from_landmarks

try:
    import mediapipe as mp
except Exception:
    mp = None


def run_monitor(
    source: str = "webcam",
    screen_w: int = 1280,
    screen_h: int = 720,
    filter_alpha: float = 0.12,
    fullscreen: bool = False,
    debug: bool = False,
):
    if mp is None:
        raise RuntimeError(
            "mediapipe is required to run the monitor: pip install mediapipe"
        )

    cam = create_source(source)
    cam.start()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    filter_x = GazeFilter(alpha=filter_alpha)
    filter_y = GazeFilter(alpha=filter_alpha)
    calibrator = GazeCalibrator()

    dx_min = dx_max = dy_min = dy_max = 0.0
    is_calibrated = False

    win_name = "GazeDemo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            captured = cam.read()
            if captured is None:
                continue
            frame = captured.color
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            current_dx = current_dy = current_z = 0.0
            sample = None

            if results.multi_face_landmarks:
                sample = gaze_from_landmarks(
                    results.multi_face_landmarks[0], w, h, captured.depth_m
                )

            if sample is not None:
                current_z = sample.z_m
                current_dx = filter_x.apply(sample.dx) or 0.0
                current_dy = filter_y.apply(sample.dy) or 0.0

            display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

            if not is_calibrated:
                target_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
                idx = calibrator.calib_idx
                idx = idx if idx < 4 else 3
                cv2.putText(
                    display,
                    f"Look at {target_names[idx]} and press 'c'",
                    (int(screen_w * 0.15), int(screen_h * 0.48)),
                    1,
                    2,
                    (255, 255, 255),
                    2,
                )
            else:
                # apply z-compensation
                z_ratio = (
                    current_z / calibrator.calib_z
                    if (calibrator.calib_z > 0 and current_z > 0)
                    else 1.0
                )
                comp_dx = current_dx * z_ratio
                comp_dy = current_dy * z_ratio

                def map_val(v, cmin, cmax, limit):
                    if abs(cmax - cmin) < 1e-6:
                        return int(limit / 2)
                    perc = (v - cmin) / (cmax - cmin)
                    return int(np.clip(perc, 0, 1) * limit)

                sx = map_val(comp_dx, dx_min, dx_max, display.shape[1])
                sy = map_val(comp_dy, dy_min, dy_max, display.shape[0])
                cv2.circle(display, (sx, sy), 20, (0, 0, 255), -1)

            if debug:
                cv2.putText(
                    display, f"Z: {current_z:.2f}", (10, 30), 1, 1, (0, 255, 0), 1
                )

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and not is_calibrated and not calibrator.is_collecting:
                calibrator.start_collection()
            if calibrator.is_collecting and sample is not None:
                progress, status = calibrator.collect(current_dx, current_dy, current_z)
                if status == "FINISHED":
                    ok, msg = calibrator.validate_and_save()
                    if not ok:
                        cv2.putText(
                            display,
                            msg,
                            (int(screen_w * 0.15), int(screen_h * 0.55)),
                            1,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    if calibrator.is_finished():
                        bounds = calibrator.finalize_bounds()
                        dx_min = bounds["dx_min"]
                        dx_max = bounds["dx_max"]
                        dy_min = bounds["dy_min"]
                        dy_max = bounds["dy_max"]
                        is_calibrated = True
            if key == ord("r"):
                calibrator = GazeCalibrator()
                is_calibrated = False
            if key == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
