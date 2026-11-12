"""Small numeric helpers: rotations, smooth profiles and coloured noise.

Conventions used throughout the simulator:

* World frame, metres. Origin on the floor directly below the blade. ``+x``
  is to the operator's right as they face the saw, ``+y`` is up, and ``+z``
  points from the blade towards the operator. The operator faces ``-z``.
* Head frame, metres. MediaPipe's canonical face model scaled from
  centimetres: ``+x`` is the subject's *left*, ``+y`` up, ``+z`` out of the
  face (forward).
* Camera frame: OpenCV's convention, ``+x`` right, ``+y`` down, ``+z``
  forward along the optical axis.
"""

from __future__ import annotations

import math

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The minimal rotation taking unit vector ``a`` onto unit vector ``b``.

    For an eye whose primary position is ``a`` this is exactly Listing's law:
    every eye orientation is reached by a single rotation about an axis lying
    in the plane perpendicular to the primary direction, so there is no
    torsion beyond what that geometry implies.
    """
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        # Opposite vectors: any perpendicular axis will do.
        axis = normalize(np.cross(a, [1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = normalize(np.cross(a, [0.0, 1.0, 0.0]))
        return axis_angle(axis, math.pi)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """World-to-camera rotation for an OpenCV camera at ``eye`` facing ``target``.

    Rows are the camera axes expressed in world coordinates.
    """
    f = normalize(np.asarray(target, float) - np.asarray(eye, float))
    r = normalize(np.cross(f, np.asarray(up, float)))  # camera +x
    d = np.cross(f, r)  # camera +y (down)
    return np.stack([r, d, f])


def yaw_pitch(direction: np.ndarray) -> tuple[float, float]:
    """Yaw and pitch, in radians, of a direction in a +z-forward, +y-up frame.

    Yaw is positive towards +x; pitch is positive *downwards*, matching how
    the head and eye angles are reported everywhere else in the simulator.
    """
    x, y, z = normalize(direction)
    return math.atan2(x, z), math.atan2(-y, math.hypot(x, z))


def direction_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """Inverse of :func:`yaw_pitch`."""
    cp = math.cos(pitch)
    return np.array([cp * math.sin(yaw), -math.sin(pitch), cp * math.cos(yaw)])


def min_jerk(s: float) -> float:
    """Minimum-jerk position profile on [0, 1].

    Flash and Hogan's profile is the standard description of voluntary
    point-to-point movements, the head included.
    """
    s = min(1.0, max(0.0, s))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def saccade_profile(s: float) -> float:
    """Position profile of a saccade on [0, 1].

    Saccades accelerate faster than they decelerate: peak velocity comes at
    roughly the first 40% of the movement for anything but the smallest. A
    minimum-jerk profile warped in time reproduces that skew.
    """
    s = min(1.0, max(0.0, s))
    return min_jerk(s**0.8)


def saccade_duration(amplitude_deg: float) -> float:
    """Main-sequence duration, seconds. About 2.2 ms per degree plus 21 ms."""
    return 0.021 + 0.0022 * abs(amplitude_deg)


def head_movement_duration(amplitude_deg: float) -> float:
    """Duration of a voluntary head movement of this amplitude, seconds.

    Head movements are an order of magnitude slower than saccades: about
    0.3 s for a small re-orientation and 0.6 s for 70 degrees, with peak
    velocities of a few hundred degrees a second for the large ones.
    """
    return 0.2 + 0.0058 * abs(amplitude_deg)


class OUNoise:
    """Ornstein-Uhlenbeck process: bounded, smooth, mean-reverting noise.

    ``sigma`` is the stationary standard deviation and ``tau`` the correlation
    time in seconds, which are the two numbers anyone can reason about. Used
    for postural sway, head jitter and slow drifts in head roll.
    """

    def __init__(
        self, rng: np.random.Generator, sigma: float, tau: float, dim: int = 1
    ):
        self.rng = rng
        self.sigma = sigma
        self.tau = max(tau, 1e-6)
        self.state = rng.normal(0.0, sigma, size=dim)

    def step(self, dt: float) -> np.ndarray:
        a = math.exp(-dt / self.tau)
        self.state = a * self.state + self.sigma * math.sqrt(
            1.0 - a * a
        ) * self.rng.normal(size=self.state.shape)
        return self.state


def smoothstep(edge0: float, edge1: float, x):
    t = np.clip((np.asarray(x, dtype=float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
