import open3d as o3d
import numpy as np

# --- 1. Load/Create Data (Example: Synthetic) ---
# In a real scenario, you'd load files:
# color_image = o3d.io.read_image("color.png")
# depth_image = o3d.io.read_image("depth.png")

# For demonstration, create dummy data:
color_np = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
depth_np = np.random.randint(1000, 2000, (480, 640), dtype=np.uint16)  # In mm

color_image = o3d.geometry.Image(color_np)
depth_image = o3d.geometry.Image(depth_np)

# --- 2. Create RGBD Image ---
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_image,
    depth_image,
    depth_scale=1000.0,
    depth_trunc=3000.0,
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
