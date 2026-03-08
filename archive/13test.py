import cv2
import numpy as np
import mediapipe as mp
import argparse
import time

# --- Settings ---
USER_TIMEOUT = 3.0
SCREEN_W, SCREEN_H = 1920, 1080

# Optional imports based on source
try:
    import freenect
except ImportError:
    freenect = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Multi-Camera Gaze Tracker")
parser.add_argument(
    "--source",
    type=str,
    default="webcam",
    choices=["kinect", "webcam", "realsense"],
    help="Select input: 'kinect', 'webcam', or 'realsense'",
)
args = parser.parse_args()


# --- Classes & Filters ---
class GazeFilter:
    def __init__(self, alpha=0.08):
        self.alpha = alpha
        self.state = None

    def apply(self, value):
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state


filter_x, filter_y = GazeFilter(), GazeFilter()
calib_data, calib_z, calib_idx = [], 1.0, 0
is_calibrated = False
dx_min, dx_max, dy_min, dy_max = 0, 0, 0, 0
last_seen_time = time.time()

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# --- Hardware Initialization ---
cap = None
pipeline = None
align = None

if args.source == "webcam":
    cap = cv2.VideoCapture(0)
elif args.source == "realsense":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)


def get_frame():
    if args.source == "kinect":
        depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
        rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth

    elif args.source == "realsense":
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame  # Return frame object for easy distance lookup

    else:  # Webcam
        ret, frame = cap.read()
        return (frame, None) if ret else (None, None)


def get_eye_data(face_landmarks, depth_src, w, h):
    def single_eye(i, l, r):
        iris = face_landmarks.landmark[i]
        lc, rc = face_landmarks.landmark[l], face_landmarks.landmark[r]
        px, py = int(iris.x * w), int(iris.y * h)

        # Depth logic per camera type
        z_m = 0
        if args.source == "kinect" and depth_src is not None:
            z_raw = depth_src[py, px] if 0 <= py < h and 0 <= px < w else 2047
            z_m = 0.1236 * np.tan(z_raw / 2842.5 + 1.1863) if z_raw < 2047 else 0
        elif args.source == "realsense" and depth_src is not None:
            if 0 <= px < w and 0 <= py < h:
                z_m = depth_src.get_distance(px, py)
        else:  # Fallback to MediaPipe synthetic Z
            z_m = abs(iris.z) * 10 + 0.6

        dx = (iris.x - (lc.x + rc.x) / 2) / ((rc.x - lc.x) / 2)
        dy = (iris.y - (lc.y + rc.y) / 2) / ((rc.x - lc.x) / 4)
        return dx, dy, z_m

    lx, ly, lz = single_eye(468, 33, 133)
    rx, ry, rz = single_eye(473, 362, 263)
    return (lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2


# --- Main Window Loop ---
cv2.namedWindow("Monitor", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame, depth_src = get_frame()
    if frame is None:
        break
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    display = np.zeros((1080, 1920, 3), dtype=np.uint8)
    raw_dx, raw_dy, current_z = 0, 0, 0
    user_present = False

    if results.multi_face_landmarks:
        user_present = True
        raw_dx, raw_dy, current_z = get_eye_data(
            results.multi_face_landmarks[0], depth_src, w, h
        )
        current_dx, current_dy = filter_x.apply(raw_dx), filter_y.apply(raw_dy)

    # --- CALIBRATION & MONITORING LOGIC ---
    if not is_calibrated:
        target_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        cv2.putText(
            display,
            f"Calibrate: {target_names[calib_idx]} ('C')",
            (SCREEN_W // 2 - 200, SCREEN_H // 2),
            1,
            2,
            (255, 255, 255),
            2,
        )
        pos = [
            (50, 50),
            (SCREEN_W - 50, 50),
            (50, SCREEN_H - 50),
            (SCREEN_W - 50, SCREEN_H - 50),
        ][calib_idx]
        cv2.circle(display, pos, 20, (0, 255, 0), -1)
    else:
        # Z-Compensation
        z_ratio = current_z / calib_z if (calib_z > 0 and current_z > 0) else 1.0
        comp_dx, comp_dy = current_dx * z_ratio, current_dy * z_ratio

        # Calculate normalized gaze
        perc_x = (
            (comp_dx - dx_min) / (dx_max - dx_min)
            if abs(dx_max - dx_min) > 0.001
            else 0.5
        )
        perc_y = (
            (comp_dy - dy_min) / (dy_max - dy_min)
            if abs(dy_max - dy_min) > 0.001
            else 0.5
        )

        is_looking_away = (
            not ((-0.1 <= perc_x <= 1.1) and (-0.1 <= perc_y <= 1.1))
            or not user_present
        )

        if not is_looking_away:
            last_seen_time = time.time()
            status_color = (0, 255, 0)
        else:
            duration = time.time() - last_seen_time
            status_color = (0, 0, 255) if duration > USER_TIMEOUT else (0, 165, 255)
            if duration > USER_TIMEOUT:
                cv2.putText(
                    display,
                    "LOOKING AWAY!",
                    (SCREEN_W // 2 - 250, SCREEN_H // 2),
                    1,
                    4,
                    (0, 0, 255),
                    4,
                )

        sx, sy = int(np.clip(perc_x, 0, 1) * SCREEN_W), int(
            np.clip(perc_y, 0, 1) * SCREEN_H
        )
        cv2.circle(display, (sx, sy), 20, status_color, -1)
        cv2.putText(
            display,
            f"Away: {max(0, time.time()-last_seen_time):.1f}s",
            (50, 50),
            1,
            2,
            status_color,
            2,
        )

    cv2.imshow("Monitor", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c") and not is_calibrated:
        calib_data.append((current_dx, current_dy))
        if calib_idx == 0:
            calib_z = current_z
        calib_idx += 1
        if calib_idx >= 4:
            dx_min, dx_max = (calib_data[0][0] + calib_data[2][0]) / 2, (
                calib_data[1][0] + calib_data[3][0]
            ) / 2
            dy_min, dy_max = (calib_data[0][1] + calib_data[1][1]) / 2, (
                calib_data[2][1] + calib_data[3][1]
            ) / 2
            is_calibrated = True
    elif key == ord("r"):  # Reset Calibration
        is_calibrated = False
        calib_idx = 0
        calib_data = []
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
