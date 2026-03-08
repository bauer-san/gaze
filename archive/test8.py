import freenect
import cv2
import numpy as np
import mediapipe as mp


# --- Filter and Helper Classes ---
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
# Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
calib_points = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
calib_data = []  # Will store (dx, dy) tuples
calib_idx = 0
is_calibrated = False

# Mapping ranges
dx_min, dx_max = 0, 0
dy_min, dy_max = 0, 0

# --- Mediapipe & Kinect Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


def get_kinect_data():
    depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth


def get_eye_displacement(face_landmarks, w, h):
    # Left Eye (468, 33, 133), Right Eye (473, 362, 263)
    def single_eye(i, l, r):
        iris = face_landmarks.landmark[i]
        lc, rc = face_landmarks.landmark[l], face_landmarks.landmark[r]
        dx = (iris.x - (lc.x + rc.x) / 2) / ((rc.x - lc.x) / 2)
        dy = (iris.y - (lc.y + rc.y) / 2) / ((rc.x - lc.x) / 4)
        return dx, dy

    lx, ly = single_eye(468, 33, 133)
    rx, ry = single_eye(473, 362, 263)
    return (lx + rx) / 2, (ly + ry) / 2


# Create a fullscreen window for calibration
cv2.namedWindow("GazeView", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("GazeView", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame, _ = get_kinect_data()
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    current_dx, current_dy = 0, 0

    if results.multi_face_landmarks:
        raw_dx, raw_dy = get_eye_displacement(results.multi_face_landmarks[0], w, h)
        current_dx = filter_x.apply(raw_dx)
        current_dy = filter_y.apply(raw_dy)

    # --- UI Logic ---
    display = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Virtual screen

    if is_calibrated:
        # 1. Normalize the current signal against the calibrated range
        # We use a helper function to ensure we handle 'inverted' axes
        def map_value(val, cal_min, cal_max, screen_limit):
            # Ensure min is actually the smaller number for the math
            actual_min = min(cal_min, cal_max)
            actual_max = max(cal_min, cal_max)

            # Calculate percentage of gaze across the calibrated range (0.0 to 1.0)
            if actual_max - actual_min == 0:
                return 0
            perc = (val - cal_min) / (cal_max - cal_min)

            # Clip to 0-1 range to prevent cursor from flying off screen
            perc = max(0, min(1, perc))
            return int(perc * screen_limit)

        screen_x = map_value(current_dx, dx_min, dx_max, 1920)
        screen_y = map_value(current_dy, dy_min, dy_max, 1080)

        # Draw the cursor
        cv2.circle(display, (screen_x, screen_y), 30, (0, 0, 255), 2)

    cv2.imshow("GazeView", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") and not is_calibrated:
        calib_data.append((current_dx, current_dy))
        calib_idx += 1
        if calib_idx >= 4:
            # Calculate bounds
            dx_min = (calib_data[0][0] + calib_data[2][0]) / 2  # Average Lefts
            dx_max = (calib_data[1][0] + calib_data[3][0]) / 2  # Average Rights
            dy_min = (calib_data[0][1] + calib_data[1][1]) / 2  # Average Tops
            dy_max = (calib_data[2][1] + calib_data[3][1]) / 2  # Average Bottoms
            is_calibrated = True

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
