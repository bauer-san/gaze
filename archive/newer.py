import freenect


def get_depth():
    # sync_get_depth returns a tuple: (depth_array, timestamp)
    depth, _ = freenect.sync_get_depth()

    # Optional: Convert to a usable format for visualization
    # Raw values are 11-bit; registered depth is in mm
    return depth


if __name__ == "__main__":
    depth_data = get_depth()
    print(f"Array Shape: {depth_data.shape}")  # Should be (480, 640)
    print(f"Distance at center pixel: {depth_data[240, 320]} mm")
