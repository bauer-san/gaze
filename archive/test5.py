import freenect
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Kinect Internal Matrix (approximate)
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
            # --- EYE CORNERS & IRIS ---
            # Left eye: Left Corner (33), Right Corner (133), Iris (468)
            l_corner = face_landmarks.landmark[33]
            r_corner = face_landmarks.landmark[133]
            iris = face_landmarks.landmark[468]

            # Convert to pixel coordinates
            lc = np.array([l_corner.x * w, l_corner.y * h])
            rc = np.array([r_corner.x * w, r_corner.y * h])
            ir = np.array([iris.x * w, iris.y * h])

            # 1. Calculate Eye Center and horizontal/vertical eye dimensions
            eye_center = (lc + rc) / 2
            eye_width = np.linalg.norm(lc - rc)

            # 2. Calculate Iris Displacement (Normalized -1.0 to 1.0)
            # This detects the iris moving INSIDE the eye socket
            dx = (ir[0] - eye_center[0]) / (eye_width / 2)
            dy = (ir[1] - eye_center[1]) / (
                eye_width / 4
            )  # Vertical movement is more restricted

            # 3. Get Depth for 3D Origin
            ix, iy = int(ir[0]), int(ir[1])
            if 0 <= ix < w and 0 <= iy < h:
                z_raw = depth_map[iy, ix]
                z_m = 0.1236 * np.tan(z_raw / 2842.5 + 1.1863) if z_raw < 2047 else 0

                if z_m > 0:
                    # World position of eye
                    world_x = (ix - K[0, 2]) * z_m / K[0, 0]
                    world_y = (iy - K[1, 2]) * z_m / K[1, 1]
                    eye_origin = np.array([world_x, world_y, z_m])

                    # 4. CREATE GAZE DIRECTION
                    # Combine "Forward" direction with the Iris offset
                    # We multiply dx/dy by a sensitivity factor (e.g., 0.5)
                    sensitivity = 0.8
                    gaze_dir = np.array([dx * sensitivity, dy * sensitivity, 1.0])
                    gaze_dir /= np.linalg.norm(gaze_dir)  # Normalize vector

                    # 5. Project Ray (0.5m length)
                    gaze_end_3d = eye_origin + (gaze_dir * -1.5)

                    # Project back to 2D for drawing
                    end_2d, _ = cv2.projectPoints(
                        np.array([gaze_end_3d]), np.zeros(3), np.zeros(3), K, None
                    )

                    p1 = (int(ir[0]), int(ir[1]))
                    p2 = (int(end_2d[0][0][0]), int(end_2d[0][0][1]))

                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        "Gaze Active",
                        (p1[0], p1[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

    cv2.imshow("Responsive Gaze Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
