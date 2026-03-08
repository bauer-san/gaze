import freenect
import cv2
import numpy as np


def get_aligned_data():
    # freenect.REGISTRATION_RGB aligns depth to the RGB camera's field of view
    depth, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)

    depth = depth.astype(np.uint8)

    # Optional: Convert depth to meters (Kinect v1 formula)
    # distance_m = 1.0 / (raw_depth * -0.0030711016 + 3.3309495161)

    return rgb, depth


while True:
    # Get the depth frame
    rgb_frame, depth_frame = get_aligned_data()

    # # Apply a colormap to make the depth data easier to see
    depth_colormap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

    # # Show the image
    cv2.imshow("Kinect Depth", depth_colormap)
    # #cv2.imshow('Kinect Depth', depth_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
