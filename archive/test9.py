import freenect
import cv2
import numpy as np
import mediapipe as mp


class GazeFilter:
    def __init__(self, alpha=0.12):
        self.alpha = alpha
        self.state = None

    def apply(self, value):
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state


filter_x = GazeFilter()
filter_y = GazeFilter()

# --- Calibration State ---
calib_points = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
calib_data = []
calib_z = 1.0  # Default calibration distance
calib_idx = 0
is_calibrated = False
dx_min, dx_max, dy_min, dy_max = 0, 0, 0, 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


def get_kinect_data():
    depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth


def get_eye_data(face_landmarks, depth_map, w, h):
    def single_eye(i, l, r):
        iris = face_landmarks.landmark[i]
        lc, rc = face_landmarks.landmark[l], face_landmarks.landmark[r]
        # Pixel coords for depth lookup
        px, py = int(iris.x * w), int(iris.y * h)
        z_raw = depth_map[py, px] if 0 <= py < h and 0 <= px < w else 2047
        z_m = 0.1236 * np.tan(z_raw / 2842.5 + 1.1863) if z_raw < 2047 else 0

        dx = (iris.x - (lc.x + rc.x) / 2) / ((rc.x - lc.x) / 2)
        dy = (iris.y - (lc.y + rc.y) / 2) / ((rc.x - lc.x) / 4)
        return dx, dy, z_m

    lx, ly, lz = single_eye(468, 33, 133)
    rx, ry, rz = single_eye(473, 362, 263)

    avg_z = (lz + rz) / 2 if (lz > 0 and rz > 0) else (lz + rz)
    return (lx + rx) / 2, (ly + ry) / 2, avg_z


cv2.namedWindow("GazeView", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("GazeView", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame, depth_map = get_kinect_data()
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    current_dx, current_dy, current_z = 0, 0, 0

    if results.multi_face_landmarks:
        raw_dx, raw_dy, current_z = get_eye_data(
            results.multi_face_landmarks[0], depth_map, w, h
        )
        current_dx = filter_x.apply(raw_dx)
        current_dy = filter_y.apply(raw_dy)

    display = np.zeros((1080, 1920, 3), dtype=np.uint8)

    if not is_calibrated:
        target = calib_points[calib_idx]
        cv2.putText(
            display,
            f"Look at {target} and press 'C'",
            (600, 540),
            1,
            2,
            (255, 255, 255),
            2,
        )
        pos = [(50, 50), (1870, 50), (50, 1030), (1870, 1030)][calib_idx]
        cv2.circle(display, pos, 20, (0, 255, 0), -1)
    else:
        # --- APPLY Z-COMPENSATION ---
        # If we are further away than calibration, current_z / calib_z > 1
        # This scales up the smaller dx/dy movement seen at a distance.
        z_ratio = current_z / calib_z if (calib_z > 0 and current_z > 0) else 1.0

        comp_dx = current_dx * z_ratio
        comp_dy = current_dy * z_ratio

        def map_value(val, c_min, c_max, limit):
            if abs(c_max - c_min) < 0.001:
                return 0
            perc = (val - c_min) / (c_max - c_min)
            return int(np.clip(perc, 0, 1) * limit)

        screen_x = map_value(comp_dx, dx_min, dx_max, 1920)
        screen_y = map_value(comp_dy, dy_min, dy_max, 1080)

        cv2.circle(display, (screen_x, screen_y), 20, (0, 0, 255), -1)
        cv2.putText(display, f"Dist: {current_z:.2f}m", (50, 50), 1, 1, (0, 255, 0), 1)

    cv2.imshow("GazeView", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c") and not is_calibrated:
        # Store Z-compensated values for calibration bounds
        # At calibration time, z_ratio is 1.0, so we just store raw dx/dy
        calib_data.append((current_dx, current_dy))
        if calib_idx == 0:
            calib_z = current_z  # Set baseline distance at first point

        calib_idx += 1
        if calib_idx >= 4:
            dx_min = (calib_data[0][0] + calib_data[2][0]) / 2
            dx_max = (calib_data[1][0] + calib_data[3][0]) / 2
            dy_min = (calib_data[0][1] + calib_data[1][1]) / 2
            dy_max = (calib_data[2][1] + calib_data[3][1]) / 2
            is_calibrated = True
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
