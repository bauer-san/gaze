import cv2

# Opening the Kinect as a capture device via OpenNI
cap = cv2.VideoCapture(cv2.CAP_OPENNI)

while True:
    ret, frame = cap.read()
    # To get actual depth map instead of BGR:
    cap.grab()
    depth_map = cap.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)[1]

    cv2.imshow("Depth", depth_map)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
