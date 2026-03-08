import cv2
import numpy as np
import mediapipe as mp
import argparse
import time



class GazeCalibrator:
    def __init__(self, sample_duration=1.0, sigma_threshold=0.05):
        self.sample_duration = sample_duration
        self.sigma_threshold = sigma_threshold

        self.is_collecting = False
        self.start_time = 0
        self.samples = []

        self.calib_idx = 0
        self.calib_data = []  # Stores (mean_dx, mean_dy)
        self.calib_z = 1.0
        self.points = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]

        # Final mapped bounds
        self.bounds = {"dx_min": 0, "dx_max": 0, "dy_min": 0, "dy_max": 0}

    def start_collection(self):
        self.is_collecting = True
        self.start_time = time.time()
        self.samples = []

    def collect_sample(self, dx, dy, z):
        """Adds a sample and returns progress (0-100) and status."""
        elapsed = time.time() - self.start_time
        progress = int((elapsed / self.sample_duration) * 100)

        if elapsed <= self.sample_duration:
            self.samples.append([dx, dy, z])
            return progress, "COLLECTING"
        else:
            return 100, "FINISHED"

    def validate_and_save(self):
        """Checks standard deviation and saves the point or rejects it."""
        samples_np = np.array(self.samples)
        means = np.mean(samples_np, axis=0)
        stds = np.std(samples_np, axis=0)

        # Quality Check: DX and DY standard deviation
        if stds[0] > self.sigma_threshold or stds[1] > self.sigma_threshold:
            self.is_collecting = False
            return False, f"Too much jitter (σx:{stds[0]:.3f}, σy:{stds[1]:.3f})"

        # Success: Save data
        self.calib_data.append((means[0], means[1]))
        if self.calib_idx == 0:
            self.calib_z = means[2]

        self.calib_idx += 1
        self.is_collecting = False

        if self.calib_idx >= 4:
            self.finalize_bounds()

        return True, "Point Accepted"

    def finalize_bounds(self):
        d = self.calib_data
        self.bounds["dx_min"] = (d[0][0] + d[2][0]) / 2
        self.bounds["dx_max"] = (d[1][0] + d[3][0]) / 2
        self.bounds["dy_min"] = (d[0][1] + d[1][1]) / 2
        self.bounds["dy_max"] = (d[2][1] + d[3][1]) / 2


# --- Initialization ---
calibrator = GazeCalibrator(sample_duration=1.0, sigma_threshold=0.04)
is_calibrated = False

# --- Settings ---
USER_TIMEOUT = 3.0
SCREEN_W, SCREEN_H = 1920, 1080
SIGMA_THRESHOLD = 0.05

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
    default="realsense",
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
    # Inside the loop...
    if not is_calibrated:
        target_pos = [(50, 50), (1870, 50), (50, 1030), (1870, 1030)][
            calibrator.calib_idx
        ]

        if not calibrator.is_collecting:
            cv2.putText(
                display,
                f"Look at {calibrator.points[calibrator.calib_idx]} - Press 'C'",
                (550, 540),
                1,
                2,
                (255, 255, 255),
                2,
            )
            cv2.circle(display, target_pos, 20, (0, 255, 0), -1)
        else:
            # Step 1: Collect
            progress, status = calibrator.collect_sample(
                current_dx, current_dy, current_z
            )

            cv2.putText(
                display, f"Recording: {progress}%", (650, 540), 1, 2, (0, 255, 255), 2
            )
            cv2.circle(display, target_pos, 25, (0, 255, 255), 3)

            # Step 2: Validate if finished
            if status == "FINISHED":
                success, message = calibrator.validate_and_save()
                if not success:
                    # Show error message
                    cv2.putText(display, message, (450, 600), 1, 2, (0, 0, 255), 2)
                    cv2.imshow("Monitor", display)
                    cv2.waitKey(2000)

                if calibrator.calib_idx >= 4:
                    is_calibrated = True

    cv2.imshow("Monitor", display)
    key = cv2.waitKey(1) & 0xFF

# --- Key Handler ---
if key == ord("c") and not is_calibrated and not calibrator.is_collecting:
    calibrator.start_collection()

cv2.destroyAllWindows()
