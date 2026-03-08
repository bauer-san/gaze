import freenect
import cv2
import numpy as np


def get_depth():

    # Sync get depth returns a tuple: (data, timestamp)
    depth, timestamp = freenect.sync_get_depth()

    # Kinect v1 returns 11-bit depth (0-2047)
    # We clip it and convert to 8-bit for visualization
    depth = depth.astype(np.uint8)

    # print("timestamp: %d, depth min: %d, depth max: %d" % (timestamp, np.min(depth), np.max(depth)))

    return depth


while True:
    # Get the depth frame
    depth_frame = get_depth()

    # # Apply a colormap to make the depth data easier to see
    depth_colormap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

    # # Show the image
    cv2.imshow("Kinect Depth", depth_colormap)
    # #cv2.imshow('Kinect Depth', depth_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
