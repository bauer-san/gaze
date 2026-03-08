import freenect
import cv2
import numpy as np
import mediapipe as mp


class GazeFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.state = None

    def apply(self, value):
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state


# Initialize filters for dx and dy
filter_x = GazeFilter(alpha=0.15)
filter_y = GazeFilter(alpha=0.15)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype="double")


def get_kinect_data():
    depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth


while True:
    frame, depth_map = get_kinect_data()
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for Left Eye
            l_corner = face_landmarks.landmark[33]
            r_corner = face_landmarks.landmark[133]
            iris = face_landmarks.landmark[468]

            lc = np.array([l_corner.x * w, l_corner.y * h])
            rc = np.array([r_corner.x * w, r_corner.y * h])
            ir = np.array([iris.x * w, iris.y * h])

            # Calculate raw iris displacement
            eye_center = (lc + rc) / 2
            eye_width = np.linalg.norm(lc - rc)

            raw_dx = (ir[0] - eye_center[0]) / (eye_width / 2)
            raw_dy = (ir[1] - eye_center[1]) / (eye_width / 4)

            # --- APPLY FILTERING ---
            dx = filter_x.apply(raw_dx)
            dy = filter_y.apply(raw_dy)

            # Get depth and project
            ix, iy = int(ir[0]), int(ir[1])
            if 0 <= ix < w and 0 <= iy < h:
                z_raw = depth_map[iy, ix]
                z_m = 0.1236 * np.tan(z_raw / 2842.5 + 1.1863) if z_raw < 2047 else 0

                if z_m > 0:
                    world_x = (ix - K[0, 2]) * z_m / K[0, 0]
                    world_y = (iy - K[1, 2]) * z_m / K[1, 1]
                    eye_origin = np.array([world_x, world_y, z_m])

                    # Create the gaze direction vector
                    sensitivity = 1.2  # Adjust this if eye movement feels too small
                    gaze_dir = np.array([dx * sensitivity, dy * sensitivity, 1.0])
                    gaze_dir /= np.linalg.norm(gaze_dir)

                    # Project for visualization
                    gaze_end_3d = eye_origin + (gaze_dir * 0.4)
                    end_2d, _ = cv2.projectPoints(
                        np.array([gaze_end_3d]), np.zeros(3), np.zeros(3), K, None
                    )

                    p1 = (int(ir[0]), int(ir[1]))
                    p2 = (int(end_2d[0][0][0]), int(end_2d[0][0][1]))

                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    cv2.circle(frame, p2, 5, (255, 0, 0), -1)  # Visualize "hit point"

    cv2.imshow("Filtered Gaze", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
