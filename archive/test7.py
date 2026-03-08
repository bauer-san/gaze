import freenect
import cv2
import numpy as np
import mediapipe as mp


class GazeFilter:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.state = None

    def apply(self, value):
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state


# Filters for the combined binocular signal
filter_x = GazeFilter(alpha=0.12)  # Slightly slower for extra stability
filter_y = GazeFilter(alpha=0.12)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype="double")


def get_kinect_data():
    depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth


def calculate_iris_displacement(
    iris_idx, l_corner_idx, r_corner_idx, face_landmarks, w, h
):
    """Calculates normalized dx, dy for a single eye."""
    iris = face_landmarks.landmark[iris_idx]
    l_corner = face_landmarks.landmark[l_corner_idx]
    r_corner = face_landmarks.landmark[r_corner_idx]

    ir = np.array([iris.x * w, iris.y * h])
    lc = np.array([l_corner.x * w, l_corner.y * h])
    rc = np.array([r_corner.x * w, r_corner.y * h])

    eye_center = (lc + rc) / 2
    eye_width = np.linalg.norm(lc - rc)

    # Normalized displacement
    dx = (ir[0] - eye_center[0]) / (eye_width / 2)
    dy = (ir[1] - eye_center[1]) / (eye_width / 4)
    return dx, dy, ir


while True:
    frame, depth_map = get_kinect_data()
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # LEFT EYE: Corner (33, 133), Iris (468)
            lx, ly, l_ir_px = calculate_iris_displacement(
                468, 33, 133, face_landmarks, w, h
            )

            # RIGHT EYE: Corner (362, 263), Iris (473)
            rx, ry, r_ir_px = calculate_iris_displacement(
                473, 362, 263, face_landmarks, w, h
            )

            # BINOCULAR FUSION: Average the displacement
            avg_dx = (lx + rx) / 2
            avg_dy = (ly + ry) / 2

            # APPLY FILTER
            final_dx = filter_x.apply(avg_dx)
            final_dy = filter_y.apply(avg_dy)

            # GET DEPTH (Average depth of both irises)
            z_raw_l = (
                depth_map[int(l_ir_px[1]), int(l_ir_px[0])]
                if 0 <= l_ir_px[1] < h
                else 2047
            )
            z_raw_r = (
                depth_map[int(r_ir_px[1]), int(r_ir_px[0])]
                if 0 <= r_ir_px[1] < h
                else 2047
            )
            z_raw = (z_raw_l + z_raw_r) / 2

            z_m = 0.1236 * np.tan(z_raw / 2842.5 + 1.1863) if z_raw < 2047 else 0

            if z_m > 0:
                # Mid-point between eyes in 3D
                mid_ir_px = (l_ir_px + r_ir_px) / 2
                world_x = (mid_ir_px[0] - K[0, 2]) * z_m / K[0, 0]
                world_y = (mid_ir_px[1] - K[1, 2]) * z_m / K[1, 1]
                eye_origin = np.array([world_x, world_y, z_m])

                # Construct Gaze Vector
                sensitivity = 1.4
                gaze_dir = np.array(
                    [final_dx * sensitivity, final_dy * sensitivity, 1.0]
                )
                gaze_dir /= np.linalg.norm(gaze_dir)

                # Project Ray
                gaze_end_3d = eye_origin + (gaze_dir * 0.4)
                end_2d, _ = cv2.projectPoints(
                    np.array([gaze_end_3d]), np.zeros(3), np.zeros(3), K, None
                )

                # Visuals
                p1 = (int(mid_ir_px[0]), int(mid_ir_px[1]))
                p2 = (int(end_2d[0][0][0]), int(end_2d[0][0][1]))
                cv2.line(frame, p1, p2, (255, 0, 255), 2)
                cv2.circle(frame, p2, 6, (0, 255, 255), -1)

    cv2.imshow("Binocular Gaze Fusion", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
