import freenect
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_kinect_data():
    depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth


# 3D model points of a generic face (in mm or arbitrary scale)
# Points: Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
model_points = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float32,
)

while True:
    frame, depth_map = get_kinect_data()
    h, w, _ = frame.shape

    # Camera internals (Approximate for Kinect v1)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract specific 2D points for PnP
            image_points = []
            # Landmark IDs for: Nose(1), Chin(152), L-Eye(33), R-Eye(263), L-Mouth(61), R-Mouth(291)
            for idx in [1, 152, 33, 263, 61, 291]:
                lm = face_landmarks.landmark[idx]
                image_points.append([lm.x * w, lm.y * h])

            image_points = np.array(image_points, dtype="double")

            # Solve for PnP (Rotation and Translation)
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            # Project a 3D axis onto the image to visualize the pose
            nose_end_point2D, _ = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]), rvec, tvec, camera_matrix, dist_coeffs
            )

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(
                frame, p1, p2, (255, 0, 0), 2
            )  # Blue line pointing where face is looking

            # Calculate Euler Angles (Pitch, Yaw, Roll)
            rmat, _ = cv2.Rodrigues(rvec)

            # Create the projection matrix [R | t]
            proj_matrix = np.hstack((rmat, tvec))

            # Decompose returns 7 values
            decomposition = cv2.decomposeProjectionMatrix(proj_matrix)
            # The Euler angles are the 7th item (index 6)
            angles = decomposition[6]

            pitch, yaw, roll = angles.flatten()

            # Note: Depending on your camera orientation, you may need to scale
            # or offset these angles to match your specific Kinect placement.
            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f} Yaw: {yaw:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    cv2.imshow("3D Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
