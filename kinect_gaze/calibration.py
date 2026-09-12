"""Simple calibration routine for four-screen corners and bounds calculation."""

import time

import numpy as np


class GazeCalibrator:
    def __init__(self, sample_duration: float = 0.8, sigma_threshold: float = 0.06):
        self.sample_duration = sample_duration
        self.sigma_threshold = sigma_threshold

        self.is_collecting = False
        self.start_time = 0.0
        self.samples: list[tuple[float, float, float]] = []

        self.calib_idx = 0
        self.calib_data: list[tuple[float, float]] = []
        self.calib_z = 1.0

    def start_collection(self):
        self.is_collecting = True
        self.start_time = time.time()
        self.samples = []

    def collect(self, dx: float, dy: float, z: float) -> tuple[int, str]:
        elapsed = time.time() - self.start_time
        progress = int((elapsed / self.sample_duration) * 100)
        if elapsed <= self.sample_duration:
            self.samples.append((dx, dy, z))
            return max(0, min(100, progress)), "COLLECTING"
        return 100, "FINISHED"

    def validate_and_save(self) -> tuple[bool, str]:
        a = np.array(self.samples)
        if a.size == 0:
            self.is_collecting = False
            return False, "No samples"
        means = np.mean(a, axis=0)
        stds = np.std(a, axis=0)

        if stds[0] > self.sigma_threshold or stds[1] > self.sigma_threshold:
            self.is_collecting = False
            return False, f"Too much jitter (σx:{stds[0]:.3f}, σy:{stds[1]:.3f})"

        self.calib_data.append((float(means[0]), float(means[1])))
        if self.calib_idx == 0:
            self.calib_z = float(means[2])
        self.calib_idx += 1
        self.is_collecting = False
        return True, "Point Accepted"

    def is_finished(self) -> bool:
        return self.calib_idx >= 4

    def finalize_bounds(self):
        d = self.calib_data
        if len(d) >= 4:
            dx_min = (d[0][0] + d[2][0]) / 2.0
            dx_max = (d[1][0] + d[3][0]) / 2.0
            dy_min = (d[0][1] + d[1][1]) / 2.0
            dy_max = (d[2][1] + d[3][1]) / 2.0
            return dict(
                dx_min=dx_min,
                dx_max=dx_max,
                dy_min=dy_min,
                dy_max=dy_max,
                calib_z=self.calib_z,
            )
        raise RuntimeError("Insufficient calibration data")
