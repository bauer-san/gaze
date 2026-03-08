import freenect
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


def get_kinect_data():
    depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth


# Camera Matrix (Kinect v1)
K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype="double")

while True:
    frame, depth_map = get_kinect_data()
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 1. Get Eye Center and Iris Center (Left Eye)
            # Eye center approx: average of landmarks 33 (left) and 133 (right) of the left eye
            # Iris center: 468
            iris_lm = face_landmarks.landmark[468]

            # Convert to Pixel Coordinates
            ix, iy = int(iris_lm.x * w), int(iris_lm.y * h)

            if 0 <= ix < w and 0 <= iy < h:
                # Get Depth from Kinect (z in meters)
                z_raw = depth_map[iy, ix]
                z_m = 0.1236 * np.tan(z_raw / 2842.5 + 1.1863) if z_raw < 2047 else 0

                if z_m > 0:
                    # 2. Back-project 2D to 3D space
                    # Un-projecting pixel (ix, iy, z_m) to World Coordinates (X, Y, Z)
                    world_x = (ix - K[0, 2]) * z_m / K[0, 0]
                    world_y = (iy - K[1, 2]) * z_m / K[1, 1]
                    eye_origin = np.array([world_x, world_y, z_m])

                    # 3. Simplify Gaze Direction
                    # For a basic gaze ray, we use the Head Pose Rotation Matrix
                    # To keep it simple here, we'll project a ray 0.5m forward from the eye
                    # based on the iris displacement

                    # Target point (0.5 meters in front of the eye origin)
                    gaze_end_3d = eye_origin + np.array([0, 0, -0.5])

                    # 4. Project 3D Ray back to 2D for visualization
                    # This projects the "End" of the gaze vector back to the image
                    end_point_2d, _ = cv2.projectPoints(
                        np.array([gaze_end_3d]), np.zeros(3), np.zeros(3), K, None
                    )

                    p1 = (ix, iy)
                    p2 = (int(end_point_2d[0][0][0]), int(end_point_2d[0][0][1]))

                    cv2.line(frame, p1, p2, (0, 255, 255), 2)
                    cv2.circle(frame, p1, 4, (0, 0, 255), -1)

    cv2.imshow("3D Gaze Vector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
