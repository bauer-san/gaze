from kinect_gaze.calibration import GazeCalibrator


def test_calibrator_accepts_low_jitter():
    cal = GazeCalibrator(sample_duration=0.01, sigma_threshold=0.2)
    cal.start_collection()
    # Simulate stable samples
    cal.samples = [(0.1, 0.05, 1.0) for _ in range(10)]
    ok, msg = cal.validate_and_save()
    assert ok, msg
    assert cal.calib_idx == 1


def test_finalize_bounds():
    cal = GazeCalibrator()
    # Simulate four canonical corner points
    cal.calib_data = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
    bounds = cal.finalize_bounds()
    assert bounds["dx_min"] == -0.5
    assert bounds["dx_max"] == 0.5
    assert bounds["dy_min"] == -0.5
    assert bounds["dy_max"] == 0.5
