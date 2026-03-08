import freenect
import open3d as o3d
import numpy as np


def get_depth():

    # Sync get depth returns a tuple: (data, timestamp)
    depth, timestamp = freenect.sync_get_depth()

    # Kinect v1 returns 11-bit depth (0-2047)
    # We clip it and convert to 8-bit for visualization
    depth = depth.astype(np.uint8)

    # print("timestamp: %d, depth min: %d, depth max: %d" % (timestamp, np.min(depth), np.max(depth)))

    return depth


def get_video():

    # Sync get depth returns a tuple: (data, timestamp)
    image, timestamp = freenect.sync_get_video()

    # Kinect v1 returns 11-bit depth (0-2047)
    # We clip it and convert to 8-bit for visualization
    image = image.astype(np.uint8)

    return image


while True:
    # Get the depth frame
    depth_frame = get_depth()
    image_frame = get_video()

    color_image = o3d.geometry.Image(image_frame)
    depth_image = o3d.geometry.Image(depth_frame)

    # --- 2. Create RGBD Image ---
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1000.0,
        depth_trunc=2.0,  # 3000.0,
        convert_rgb_to_intensity=False,
    )

    # --- 3. Define Camera Intrinsics ---
    # Example intrinsic parameters (replace with your actual camera's)
    fx, fy = 525.0, 525.0
    cx, cy = 320.0, 240.0
    width, height = 640, 480
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # --- 4. Generate Point Cloud ---
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # --- 5. Process & Visualize ---
    # Flip the point cloud to be oriented correctly (often needed)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])  # Visualize
