import freenect
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Necessary for Iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_kinect_data():
    """Fetches aligned RGB and Depth frames."""
    # DEPTH_REGISTERED aligns the depth map to the RGB camera coordinates
    depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)

    # Convert RGB for MediaPipe (MediaPipe expects RGB, OpenCV uses BGR)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb_bgr, depth


def raw_to_meters(raw_depth):
    """Converts Kinect v1 raw 11-bit depth to meters."""
    # Standard approximation formula for Kinect v1
    if raw_depth < 2047:
        return 0.1236 * np.tan(raw_depth / 2842.5 + 1.1863)
    return 0


while True:
    frame, depth_map = get_kinect_data()
    h, w, _ = frame.shape

    # Process landmarks
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 1. Focus on the Left Iris center (Index 468)
            # 2. Focus on the Right Iris center (Index 473)
            for idx in [468, 473]:
                landmark = face_landmarks.landmark[idx]

                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Ensure coordinates are within frame bounds
                if 0 <= cx < w and 0 <= cy < h:
                    # Get depth at the exact landmark pixel
                    raw_z = depth_map[cy, cx]
                    z_meters = raw_to_meters(raw_z)

                    # Draw visual feedback
                    color = (0, 255, 0) if idx == 468 else (255, 0, 0)
                    cv2.circle(frame, (cx, cy), 3, color, -1)

                    # Display distance above the eye
                    cv2.putText(
                        frame,
                        f"{z_meters:.2f}m",
                        (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

    # Show the result
    cv2.imshow("Kinect Gaze Depth Fusion", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
freenect.sync_stop()
