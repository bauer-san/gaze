"""Material ids and the deferred shading pass.

Lighting is the plant's: overhead fluorescent fixtures above the saw, a
little fill bouncing off the stainless head housing that the camera is
mounted on, and a hemispherical ambient. No cast shadows; the darkening
that matters for an eye tracker -- the upper lid and lashes shading the top
of the eye -- is put in by the eye shader directly.
"""

from __future__ import annotations

import math

import numpy as np

from .mathutil import normalize

FACE = 1
SOCKET = 2
SHELL = 3
SKIN_PLAIN = 4
EYE = 5
GLASSES = 6
COAT = 7
APRON = 8
METAL = 10
BLADE = 11
BOARD = 12
WALL = 13
FLOOR = 14
DARK = 15
MEAT = 16
LAMP = 17
PANEL = 18

# Rounded human-eye constants, duplicated from face.py to avoid a cycle.
_EYEBALL_R = 0.012
_IRIS_ANGLE = math.asin(0.00585 / 0.012)
_CORNEA_CENTRE = 0.00534 / 0.012  # in eyeball radii, along the visual axis


def srgb(linear: np.ndarray) -> np.ndarray:
    a = np.clip(linear, 0.0, 1.0)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)


def linear(srgb_value) -> np.ndarray:
    s = np.asarray(srgb_value, dtype=float)
    return np.where(s <= 0.04045, s / 12.92, np.power((s + 0.055) / 1.055, 2.4))


class Lighting:
    """Lights in world space; :meth:`to_camera` puts them in a camera's frame."""

    def __init__(self, intensity: float = 1.0):
        self.up = np.array([0.0, 1.0, 0.0])
        # Directions *towards* the light.
        self.keys = [
            (normalize(np.array([0.25, 1.0, 0.35])), 0.95 * intensity),
            (normalize(np.array([-0.6, 0.8, -0.2])), 0.35 * intensity),
        ]
        self.sky = np.array([0.42, 0.43, 0.45]) * intensity
        self.ground = np.array([0.20, 0.19, 0.18]) * intensity
        # Fill from the saw side: the stainless housing behind the camera.
        self.fill_dir = normalize(np.array([0.0, 0.2, -1.0]))
        self.fill = 0.28 * intensity

    def to_camera(self, R_wc: np.ndarray):
        return {
            "up": R_wc @ self.up,
            "keys": [(R_wc @ d, k) for d, k in self.keys],
            "sky": self.sky,
            "ground": self.ground,
            "fill_dir": R_wc @ self.fill_dir,
            "fill": self.fill,
        }


def _sample(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear texture lookup. ``uv`` in [0, 1], v up."""
    n = tex.shape[0]
    x = np.clip(uv[:, 0] * n - 0.5, 0, n - 1.001)
    y = np.clip((1.0 - uv[:, 1]) * n - 0.5, 0, n - 1.001)
    x0 = x.astype(int)
    y0 = y.astype(int)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    return (
        tex[y0, x0] * (1 - fx) * (1 - fy)
        + tex[y0, x0 + 1] * fx * (1 - fy)
        + tex[y0 + 1, x0] * (1 - fx) * fy
        + tex[y0 + 1, x0 + 1] * fx * fy
    )


def _hash_noise(p: np.ndarray, scale: float) -> np.ndarray:
    """Cheap deterministic per-position noise in [0, 1)."""
    q = np.floor(p * scale)
    h = np.sin(q[:, 0] * 12.9898 + q[:, 1] * 78.233 + q[:, 2] * 37.719) * 43758.5453
    return h - np.floor(h)


def headwear_mask(head_pos: np.ndarray, kind: str) -> np.ndarray:
    """Which head-frame positions are covered by hair or headwear."""
    y, z = head_pos[:, 1], head_pos[:, 2]
    if kind == "cap":
        line = 0.066 + np.minimum(z - 0.04, 0.0) * 0.35
    else:
        line = 0.074 + np.minimum(z - 0.045, 0.0) * 0.72
    return y > line


def shade(gbuf, lights: dict, ctx: dict) -> np.ndarray:
    """Linear RGB for every pixel in the G-buffer."""
    n = gbuf.normal
    mat = gbuf.material
    attrs = gbuf.attrs
    p = len(mat)
    albedo = np.zeros((p, 3))
    spec_k = np.zeros(p)
    spec_exp = np.full(p, 16.0)
    wrap = np.zeros(p)
    emissive = np.zeros((p, 3))
    spec_n = n.copy()
    occl = np.ones(p)
    app = ctx["appearance"]
    skin = linear_colour(app.skin)

    m = mat == FACE
    if m.any():
        albedo[m] = _sample(ctx["face_texture"], attrs[m, 0:2])
        covered = headwear_mask(attrs[m, 2:5], app.headwear)
        if app.headwear != "hair":
            albedo[np.nonzero(m)[0][covered]] = _headwear_colour(
                attrs[m][covered, 2:5], app
            )
        spec_k[m] = 0.07
        spec_exp[m] = 18.0
        wrap[m] = 0.35

    m = mat == SHELL
    if m.any():
        pos = attrs[m, 0:3]
        covered = headwear_mask(pos, app.headwear)
        col = np.tile(skin, (int(m.sum()), 1))
        col[covered] = _headwear_colour(pos[covered], app)
        albedo[m] = col
        spec_k[m] = np.where(covered, 0.05, 0.06)
        wrap[m] = 0.3

    m = mat == SKIN_PLAIN
    albedo[m] = skin
    spec_k[m] = 0.05
    wrap[m] = 0.35

    m = mat == SOCKET
    albedo[m] = np.array([0.35, 0.13, 0.12])
    occl[m] = 0.35

    m = mat == EYE
    if m.any():
        a, k, ex, sn, oc = _eye(attrs[m], n[m], app)
        albedo[m], spec_k[m], spec_exp[m], spec_n[m], occl[m] = a, k, ex, sn, oc

    m = mat == GLASSES
    albedo[m] = 0.05
    spec_k[m] = 0.5
    spec_exp[m] = 60

    m = mat == COAT
    if m.any():
        tex = 0.94 + 0.06 * _hash_noise(attrs[m, :3], 900.0)
        albedo[m] = linear_colour(app.coat) * tex[:, None]
        wrap[m] = 0.2
    m = mat == APRON
    albedo[m] = linear_colour(app.apron)
    spec_k[m] = 0.25
    spec_exp[m] = 30

    m = mat == METAL
    if m.any():
        brushed = 0.9 + 0.1 * _hash_noise(
            attrs[m, :3] * np.array([1.0, 40.0, 1.0]), 60.0
        )
        albedo[m] = np.array([0.42, 0.44, 0.46]) * brushed[:, None]
        spec_k[m] = 0.6
        spec_exp[m] = 40
    m = mat == BLADE
    albedo[m] = np.array([0.55, 0.56, 0.58])
    spec_k[m] = 0.9
    spec_exp[m] = 80
    m = mat == BOARD
    albedo[m] = np.array([0.80, 0.80, 0.78])
    spec_k[m] = 0.1
    m = mat == WALL
    if m.any():
        albedo[m] = (
            np.array([0.60, 0.64, 0.62])
            * (0.97 + 0.03 * _hash_noise(attrs[m, :3], 8.0))[:, None]
        )
    m = mat == FLOOR
    if m.any():
        pos = attrs[m, :3]
        grout = (np.abs(((pos[:, 0] / 0.3) % 1) - 0.5) > 0.48) | (
            np.abs(((pos[:, 2] / 0.3) % 1) - 0.5) > 0.48
        )
        albedo[m] = np.where(grout[:, None], 0.18, 0.40) * np.array([1.0, 0.92, 0.85])
        spec_k[m] = 0.15
    m = mat == DARK
    albedo[m] = 0.06
    spec_k[m] = 0.2
    m = mat == PANEL
    albedo[m] = np.array([0.15, 0.30, 0.55])
    spec_k[m] = 0.3
    m = mat == MEAT
    if m.any():
        fat = _hash_noise(attrs[m, :3], 70.0) > 0.82
        albedo[m] = np.where(fat[:, None], [0.85, 0.75, 0.65], [0.55, 0.10, 0.09])
        spec_k[m] = 0.35
        spec_exp[m] = 40
    m = mat == LAMP
    emissive[m] = 3.0

    # Lighting.
    pos = ctx["positions"]
    view = -normalize(pos)
    hemi = 0.5 + 0.5 * (n @ lights["up"])
    amb = (
        lights["ground"][None] * (1 - hemi)[:, None]
        + lights["sky"][None] * hemi[:, None]
    )
    diffuse = amb * occl[:, None]
    spec = np.zeros(p)
    for d, k in lights["keys"] + [(lights["fill_dir"], lights["fill"])]:
        ndl = n @ d
        diffuse += k * (np.clip((ndl + wrap) / (1 + wrap), 0, 1) * occl)[:, None]
        h = normalize(d[None] + view)
        nh = np.clip(np.einsum("ij,ij->i", spec_n, h), 0, 1)
        spec += k * spec_k * nh**spec_exp * (ndl > -0.1) * occl
    col = albedo * diffuse + spec[:, None] + emissive
    return col


def linear_colour(c) -> np.ndarray:
    return np.asarray(c, dtype=float)


def _headwear_colour(pos: np.ndarray, app) -> np.ndarray:
    if app.headwear == "cap":
        weave = 0.93 + 0.07 * _hash_noise(pos, 1500.0)
        return linear_colour(app.cap_colour) * weave[:, None]
    strands = 0.8 + 0.4 * _hash_noise(pos * np.array([1.0, 1.0, 6.0]), 700.0)
    base = linear_colour(app.hair) * strands[:, None]
    if app.headwear == "hairnet":
        u = np.abs(np.sin(pos[:, 0] * 700)) < 0.12
        v = np.abs(np.sin((pos[:, 1] + pos[:, 2]) * 700)) < 0.12
        base = np.where((u | v)[:, None], base * 0.8 + 0.12, base)
    return base


def _eye(attrs: np.ndarray, normal: np.ndarray, app):
    """Sclera, iris and pupil laid out on the eyeball by angle from the axis."""
    d = normalize(attrs[:, 0:3])
    pupil = attrs[:, 3]
    head_off = attrs[:, 4:7] / _EYEBALL_R
    axis_cam = normalize(attrs[:, 7:10])
    theta = np.arccos(np.clip(d[:, 2], -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0])

    r = np.clip((theta - pupil) / np.maximum(_IRIS_ANGLE - pupil, 1e-3), 0, 1)
    stri = 0.5 + 0.5 * np.sin(phi * 41 + 2.5 * np.sin(phi * 7) + r * 3.0)
    iris = np.asarray(app.iris)[None] * (0.72 + 0.4 * stri[:, None])
    iris *= (1 - 0.55 * _smooth(0.72, 1.0, r))[:, None]  # limbal ring
    iris *= (1 + 0.35 * np.exp(-(((r - 0.22) / 0.09) ** 2)))[:, None]  # collarette

    sclera = np.asarray(app.sclera)[None] * np.ones((len(d), 1))
    sclera = sclera * (1 - 0.12 * _smooth(0.8, 1.6, theta))[:, None]
    sclera[:, 0] += 0.05 * _smooth(0.9, 1.6, theta)

    pupil_col = np.array([0.015, 0.012, 0.012])
    w_iris = 1 - _smooth(_IRIS_ANGLE - 0.012, _IRIS_ANGLE + 0.012, theta)
    w_pupil = 1 - _smooth(pupil - 0.01, pupil + 0.01, theta)
    albedo = sclera * (1 - w_iris)[:, None] + iris * w_iris[:, None]
    albedo = albedo * (1 - w_pupil)[:, None] + pupil_col * w_pupil[:, None]

    # The upper lid and lashes shade the top of the exposed eye.
    occl = 1 - 0.45 * _smooth(0.05, 0.75, head_off[:, 1])
    occl *= 1 - 0.25 * _smooth(0.55, 0.95, np.abs(head_off[:, 0]))

    # Specular off the cornea, not the eyeball: a tighter sphere in front.
    cornea_n = normalize(normal - axis_cam * _CORNEA_CENTRE)
    on_cornea = theta < _IRIS_ANGLE + 0.05
    spec_n = np.where(on_cornea[:, None], cornea_n, normal)
    spec_k = np.where(on_cornea, 1.4, 0.35)
    spec_exp = np.where(on_cornea, 400.0, 60.0)
    return albedo, spec_k, spec_exp, spec_n, occl


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
