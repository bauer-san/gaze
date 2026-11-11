"""The operator's head: face surface, eyeballs, eyelids, skull and hair.

The face surface is MediaPipe's own canonical face model (Apache-2.0, see
``assets/NOTICE``). That choice does two jobs. It is an anatomically
plausible adult face with the proportions FaceMesh was trained on, and its
468 vertices *are* the FaceMesh landmarks, so every rendered frame comes with
exact ground-truth landmark positions to score the detector against.

What the canonical model does not have is anything behind the face, or any
eyes: the eye openings are flat triangles. Those triangles are removed and
replaced by real eyeballs that rotate independently of the head, behind
eyelids that blink and that follow vertical gaze the way real lids do. A
skull, hair or a cap, ears and a jaw underside complete the head.

Everything here is in the head frame, metres (see :mod:`.mathutil`).
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import numpy as np

from . import materials as M
from .mathutil import normalize, rotation_between
from .meshes import uv_sphere
from .raster import Mesh, vertex_normals

ASSET = pathlib.Path(__file__).with_name("assets") / "canonical_face_model.obj"

# -- landmark topology (MediaPipe face mesh) ---------------------------------
#
# MediaPipe names eyes from the *image's* point of view: its "left eye",
# landmarks 33/133 and iris 468, is the subject's right eye. These lists are
# for that eye, ordered from the outer corner to the inner one; the other eye
# is found by mirroring, since the canonical model is exactly symmetric.
EYE_OUTER, EYE_INNER = 33, 133
UPPER_LID = (246, 161, 160, 159, 158, 157, 173)
LOWER_LID = (7, 163, 144, 145, 153, 154, 155)
UPPER_LID_1 = (247, 30, 29, 27, 28, 56, 190)
LOWER_LID_1 = (25, 110, 24, 23, 22, 26, 112)
UPPER_LID_2 = (113, 225, 224, 223, 222, 221, 189)
BROW_UPPER = (156, 70, 63, 105, 66, 107, 55, 193)
BROW_LOWER = (35, 124, 46, 53, 52, 65)
LIPS_OUTER = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
)  # fmt: skip
LIPS_SEAM = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)  # fmt: skip
NOSE_ALAR = (98, 327)
NOSE_BOTTOM = 2
CHEEK = (50, 280)

# Human eye dimensions, metres. The eyeball's radius of rotation and the
# iris's diameter are two of the most consistent dimensions on the body.
EYEBALL_RADIUS = 0.012
IRIS_RADIUS = 0.00585
LID_CLEARANCE = 0.0007

# How far each ring of eyelid vertices follows the lid margin. The margin
# moves fully; the skin above it folds, so it moves progressively less.
UPPER_WEIGHTS = (1.0, 0.55, 0.2)
LOWER_WEIGHTS = (1.0, 0.45)

# Lid-gaze coupling, defaults for Appearance. The upper lid tracks vertical
# gaze (roughly 0.7 to 1.0 of the eye's elevation) so that the lid margin
# keeps its relation to the iris; the lower lid moves less. Both vary between
# people and both change how much iris FaceMesh gets to see. For an operator
# looking down at a blade this is not a detail: the upper lid drops over the
# iris, which is a large part of why downward gaze is hard to measure.
UPPER_LID_GAIN = 0.7
LOWER_LID_GAIN = 0.45


def load_canonical(path: pathlib.Path = ASSET):
    """Vertices (metres), faces and per-vertex UVs of the canonical face."""
    verts, uvs, faces = [], [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "vt":
            uvs.append([float(x) for x in parts[1:3]])
        elif parts[0] == "f":
            faces.append(
                [tuple(int(q) - 1 for q in t.split("/")[:2]) for t in parts[1:4]]
            )
    verts = np.asarray(verts) * 0.01
    uvs = np.asarray(uvs)
    vert_uv = np.zeros((len(verts), 2))
    tri = np.zeros((len(faces), 3), dtype=np.int64)
    for i, f in enumerate(faces):
        for j, (v, t) in enumerate(f):
            tri[i, j] = v
            vert_uv[v] = uvs[t]
    return verts, tri, vert_uv


def mirror_indices(verts: np.ndarray) -> np.ndarray:
    """For each vertex, the index of its mirror image across x = 0."""
    m = verts * np.array([-1.0, 1.0, 1.0])
    d = np.linalg.norm(verts[None, :, :] - m[:, None, :], axis=2)
    return np.argmin(d, axis=1)


@dataclass(frozen=True)
class Appearance:
    """What the operator looks like. Every field has a plausible default."""

    skin: tuple[float, float, float] = (0.78, 0.60, 0.50)  # linear-ish RGB
    lips: tuple[float, float, float] = (0.62, 0.34, 0.32)
    brows: tuple[float, float, float] = (0.16, 0.11, 0.08)
    hair: tuple[float, float, float] = (0.18, 0.12, 0.08)
    iris: tuple[float, float, float] = (0.36, 0.22, 0.10)
    sclera: tuple[float, float, float] = (0.86, 0.83, 0.80)
    # "hair", "cap" (a white food-plant cap) or "hairnet".
    headwear: str = "cap"
    cap_colour: tuple[float, float, float] = (0.88, 0.89, 0.90)
    coat: tuple[float, float, float] = (0.86, 0.87, 0.88)
    apron: tuple[float, float, float] = (0.70, 0.74, 0.80)
    head_scale: float = 1.0
    # Face width relative to height; 1.0 is the canonical model.
    face_width: float = 1.0
    stature: float = 1.75
    pupil_radius: float = 0.0018
    # Opens (positive) or narrows (negative) the resting palpebral fissure,
    # radians of lid elevation. The canonical model's eyes are a little narrow.
    lid_open: float = math.radians(4.0)
    upper_lid_gain: float = UPPER_LID_GAIN
    lower_lid_gain: float = LOWER_LID_GAIN
    safety_glasses: bool = False
    seed: int = 0


PRESETS = {
    "default": Appearance(),
    "dark": Appearance(
        skin=(0.36, 0.24, 0.17),
        lips=(0.30, 0.17, 0.15),
        brows=(0.05, 0.04, 0.03),
        hair=(0.05, 0.04, 0.035),
        iris=(0.16, 0.09, 0.05),
        headwear="hairnet",
        face_width=1.03,
        stature=1.82,
    ),
    "light": Appearance(
        skin=(0.86, 0.70, 0.62),
        lips=(0.70, 0.42, 0.40),
        brows=(0.45, 0.33, 0.20),
        hair=(0.55, 0.42, 0.25),
        iris=(0.30, 0.42, 0.55),
        headwear="hair",
        face_width=0.97,
        stature=1.66,
    ),
    "medium": Appearance(
        skin=(0.62, 0.44, 0.32),
        lips=(0.50, 0.28, 0.25),
        brows=(0.10, 0.07, 0.05),
        hair=(0.09, 0.07, 0.05),
        iris=(0.24, 0.15, 0.07),
        headwear="cap",
        stature=1.70,
    ),
}


@dataclass
class EyeRig:
    centre: np.ndarray  # head frame, metres
    upper: list  # rings of vertex indices, margin first
    lower: list
    corners: tuple[int, int]  # outer, inner
    contour: tuple  # all margin vertices, for removing the opening


@dataclass
class PosedHead:
    meshes: list
    landmarks: np.ndarray  # (478, 3) head frame
    eye_centres: np.ndarray  # (2, 3) head frame
    eye_axes: np.ndarray  # (2, 3) head frame, unit


class FaceModel:
    """The rigged head. Build once per operator; :meth:`pose` per frame."""

    def __init__(self, appearance: Appearance | None = None):
        self.appearance = appearance or Appearance()
        a = self.appearance
        verts, tri, uv = load_canonical()
        scale = np.array([a.face_width, 1.0, 1.0]) * a.head_scale
        verts = verts * scale
        self.base_vertices = verts
        self.uv = uv
        mirror = mirror_indices(verts)

        def m(seq):
            return tuple(int(mirror[i]) for i in seq)

        self.eyes = []
        for side in (0, 1):
            f = (lambda s: tuple(s)) if side == 0 else m
            upper = [f(UPPER_LID), f(UPPER_LID_1), f(UPPER_LID_2)]
            lower = [f(LOWER_LID), f(LOWER_LID_1)]
            corners = f((EYE_OUTER, EYE_INNER))
            contour = tuple(upper[0]) + tuple(lower[0]) + tuple(corners)
            self.eyes.append(
                EyeRig(
                    centre=self._eye_centre(verts, upper[0], lower[0], corners),
                    upper=upper,
                    lower=lower,
                    corners=corners,
                    contour=contour,
                )
            )

        # Remove the flat eye-opening triangles. They are kept, pushed back
        # into the orbit and darkened, as the socket that shows in the
        # corners of the eye where the eyeball does not reach.
        opening = np.zeros(len(tri), dtype=bool)
        for eye in self.eyes:
            s = set(eye.contour)
            opening |= np.array([all(int(v) in s for v in f) for f in tri])
        self.faces = tri[~opening]
        self.socket_faces = tri[opening]

        # Resting lid-margin elevations, for the rig.
        self._rest = []
        for eye in self.eyes:
            self._rest.append(
                {
                    idx: _spherical(verts[idx] - eye.centre)
                    for ring in eye.upper + eye.lower
                    for idx in ring
                }
            )

        self.texture = paint_face_texture(self.appearance, verts, uv)
        # The shell is tucked behind the *closed* surface, eye openings
        # included; otherwise it would show through the holes.
        self.shell = self._build_shell(verts, tri)
        self.ears = [
            uv_sphere(
                centre=(sx * 0.076 * a.face_width * a.head_scale, 0.008, -0.030),
                radii=np.array([0.010, 0.030, 0.019]) * a.head_scale,
                rings=8,
                segments=14,
                material=M.SKIN_PLAIN,
            )
            for sx in (-1.0, 1.0)
        ]
        self.eyeball = uv_sphere(
            radii=(EYEBALL_RADIUS,) * 3,
            rings=18,
            segments=32,
            theta_max=math.radians(110),
            material=M.EYE,
        )
        if a.safety_glasses:
            self.glasses = _glasses_frame(self.eyes, a)
        else:
            self.glasses = None

    # -- construction --------------------------------------------------------

    @staticmethod
    def _eye_centre(verts, upper, lower, corners) -> np.ndarray:
        """Place the eyeball so its surface sits just behind every lid vertex."""
        ring = np.array(list(upper) + list(lower) + list(corners))
        pts = verts[ring]
        cx = (verts[corners[0], 0] + verts[corners[1], 0]) / 2.0
        cy = (verts[list(upper)][:, 1].mean() + verts[list(lower)][:, 1].mean()) / 2.0
        d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        inside = d2 < (EYEBALL_RADIUS - 1e-4) ** 2
        sag = np.sqrt(np.maximum(EYEBALL_RADIUS**2 - d2[inside], 0.0))
        cz = float(np.min(pts[inside, 2] - sag)) - LID_CLEARANCE
        return np.array([cx, cy, cz])

    def _build_shell(self, face_verts: np.ndarray, face_tris: np.ndarray) -> Mesh:
        """Skull, hair and jaw underside: an ellipsoid tucked behind the face.

        Every shell vertex whose ray from the skull centre would pass in front
        of the face surface is pulled back behind it, so the face always wins
        the depth test and the two meet without a visible seam at the edges.
        """
        a = self.appearance
        s = a.head_scale
        centre = np.array([0.0, 0.016, -0.026]) * s
        radii = np.array([0.079 * a.face_width, 0.106, 0.094]) * s
        sphere = uv_sphere(centre=(0, 0, 0), radii=(1, 1, 1), rings=28, segments=40)
        unit = sphere.vertices
        verts = centre + unit * radii
        dirs = normalize(verts - centre)
        dist = np.linalg.norm(verts - centre, axis=1)
        hit = _ray_mesh_distance(centre, dirs, face_verts, face_tris)
        pull = np.isfinite(hit) & (hit < dist + 0.004)
        dist = np.where(pull, np.minimum(dist, hit - 0.004), dist)
        verts = centre + dirs * dist[:, None]
        # attrs: head-frame position, which the shader uses to place hair.
        return Mesh(
            verts,
            sphere.faces,
            material=M.SHELL,
            attrs=verts.copy(),
            normals=vertex_normals(verts, sphere.faces),
        )

    # -- posing --------------------------------------------------------------

    def pose(
        self,
        eye_dirs: np.ndarray,
        blink: float = 0.0,
        pupil_radius: float | None = None,
        squint: float = 0.0,
    ) -> PosedHead:
        """Head-frame meshes for one instant.

        ``eye_dirs`` are the two visual axes in the head frame (index 0 is
        MediaPipe's "left" eye, the subject's right). ``blink`` is lid closure
        in [0, 1]. Eyelids follow each eye's elevation.
        """
        a = self.appearance
        verts = self.base_vertices.copy()
        axes = normalize(np.asarray(eye_dirs, float))
        for eye, rest, axis in zip(self.eyes, self._rest, axes, strict=True):
            elev = math.asin(float(np.clip(axis[1], -1.0, 1.0)))
            self._pose_lids(verts, eye, rest, elev, blink, squint, a)

        face = Mesh(
            verts,
            self.faces,
            material=M.FACE,
            attrs=np.concatenate([self.uv, verts], axis=1),
        )
        socket_verts = self.base_vertices.copy()
        for eye in self.eyes:
            idx = list(eye.contour)
            socket_verts[idx] = (
                socket_verts[idx] + (eye.centre - socket_verts[idx]) * 0.25
            )
        socket = Mesh(socket_verts, self.socket_faces, material=M.SOCKET)

        meshes = [face, socket, self.shell, *self.ears]
        rp = a.pupil_radius if pupil_radius is None else pupil_radius
        for eye, axis in zip(self.eyes, axes, strict=True):
            R = rotation_between(np.array([0.0, 0.0, 1.0]), axis)
            ball = self.eyeball.transformed(R, eye.centre)
            # attrs: [unit local (3), pupil angle, head-frame offset (3)]
            n = len(ball.vertices)
            pupil_angle = math.asin(min(rp / EYEBALL_RADIUS, 0.9))
            ball.attrs = np.concatenate(
                [
                    self.eyeball.attrs,
                    np.full((n, 1), pupil_angle),
                    ball.vertices - eye.centre,
                ],
                axis=1,
            )
            meshes.append(ball)
        if self.glasses is not None:
            meshes.extend(self.glasses)

        landmarks = np.zeros((478, 3))
        landmarks[:468] = verts
        for k, (eye, axis) in enumerate(zip(self.eyes, axes, strict=True)):
            landmarks[468 + 5 * k : 473 + 5 * k] = iris_points(eye.centre, axis)
        return PosedHead(
            meshes=meshes,
            landmarks=landmarks,
            eye_centres=np.stack([e.centre for e in self.eyes]),
            eye_axes=axes,
        )

    def _pose_lids(self, verts, eye, rest, elev, blink, squint, a: Appearance) -> None:
        up_shift = a.upper_lid_gain * elev + a.lid_open - squint * 0.12
        low_shift = a.lower_lid_gain * elev + squint * 0.10
        upper_margin = eye.upper[0]
        lower_margin = eye.lower[0]
        # Where each lower-lid margin vertex ends up, for the blink to close onto.
        lower_elev = {}
        for w, ring in zip(LOWER_WEIGHTS, eye.lower, strict=False):
            for i, idx in enumerate(ring):
                r, e, az = rest[idx]
                e_new = e + w * low_shift
                if w == 1.0:
                    lower_elev[i] = e_new
                verts[idx] = eye.centre + _cartesian(r, e_new, az)
        for w, ring in zip(UPPER_WEIGHTS, eye.upper, strict=False):
            for i, idx in enumerate(ring):
                r, e, az = rest[idx]
                e_open = e + w * up_shift
                floor = lower_elev.get(i, rest[lower_margin[i]][1]) + 0.01
                if w == 1.0:
                    e_open = max(e_open, floor)
                    e_new = e_open + (floor - e_open) * blink
                else:
                    # Upper rings follow the margin part of the way down.
                    margin_drop = (rest[upper_margin[i]][1] + up_shift) - floor
                    e_new = e_open - w * blink * max(margin_drop, 0.0)
                r = max(r, EYEBALL_RADIUS + LID_CLEARANCE * 1.5)
                verts[idx] = eye.centre + _cartesian(r, e_new, az)


def iris_points(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """MediaPipe's five iris landmarks: centre, then four limbus points."""
    axis = normalize(axis)
    R = rotation_between(np.array([0.0, 0.0, 1.0]), axis)
    depth = math.sqrt(EYEBALL_RADIUS**2 - IRIS_RADIUS**2)
    local = np.array(
        [
            [0.0, 0.0, depth],
            [IRIS_RADIUS, 0.0, depth],
            [0.0, IRIS_RADIUS, depth],
            [-IRIS_RADIUS, 0.0, depth],
            [0.0, -IRIS_RADIUS, depth],
        ]
    )
    return local @ R.T + centre


def _spherical(v: np.ndarray):
    r = float(np.linalg.norm(v))
    return r, math.asin(float(v[1]) / r), math.atan2(float(v[0]), float(v[2]))


def _cartesian(r: float, elev: float, azim: float) -> np.ndarray:
    ce = math.cos(elev)
    return np.array(
        [r * ce * math.sin(azim), r * math.sin(elev), r * ce * math.cos(azim)]
    )


def _ray_mesh_distance(origin, dirs, verts, faces) -> np.ndarray:
    """Distance along each ray to the furthest hit on the mesh, or inf.

    The furthest rather than the nearest hit, because the rays start inside
    the head and the face surface is the outermost thing they cross.
    """
    v0, v1, v2 = (verts[faces[:, i]] for i in range(3))
    e1, e2 = v1 - v0, v2 - v0
    out = np.full(len(dirs), np.inf)
    best = np.full(len(dirs), -np.inf)
    for start in range(0, len(dirs), 256):
        d = dirs[start : start + 256][:, None, :]
        p = np.cross(d, e2[None])
        det = np.einsum("rtk,tk->rt", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = origin - v0
        u = np.einsum("rtk,tk->rt", p, s) * inv
        q = np.cross(s, e1)
        v = np.einsum("rk,tk->rt", d[:, 0, :], q) * inv
        t = np.einsum("tk,tk->t", e2, q)[None, :] * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        tt = np.where(hit, t, -np.inf).max(axis=1)
        best[start : start + 256] = tt
    out[np.isfinite(best)] = best[np.isfinite(best)]
    return out


def _glasses_frame(eyes, a: Appearance) -> list:
    """Thin dark rims in front of each eye. Lenses are left out on purpose:
    a clear lens's only visible effect is reflections, handled by the shader
    as a specular term on the rim rather than as geometry."""
    from .meshes import tube

    meshes = []
    for eye in eyes:
        c = eye.centre + np.array([0.0, 0.0, 0.026])
        pts = []
        for k in range(17):
            t = 2 * math.pi * k / 16
            pts.append(c + np.array([0.024 * math.cos(t), 0.018 * math.sin(t), 0.0]))
        for p, q in zip(pts[:-1], pts[1:], strict=True):
            meshes.append(
                tube(p, q, 0.0015, segments=6, rings=1, caps=False, material=M.GLASSES)
            )
    return meshes


# -- texture ----------------------------------------------------------------

TEXTURE_SIZE = 512


def paint_face_texture(a: Appearance, verts: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """A procedural albedo map in the canonical model's UV space.

    Skin with low-frequency mottling, cheeks, lips with a darker seam,
    eyebrows with hair-direction streaks, a lash line and a lid crease. It
    does not need to be beautiful. It needs to give the landmark model the
    contrast cues a real face does: brows, lash line and lips are what
    FaceMesh locks onto.
    """
    rng = np.random.default_rng(a.seed + 17)
    n = TEXTURE_SIZE
    tex = np.ones((n, n, 3)) * np.asarray(a.skin)
    tex *= 1.0 + 0.05 * (_value_noise(rng, n, 8)[..., None] - 0.5)
    tex *= 1.0 + 0.03 * (_value_noise(rng, n, 32)[..., None] - 0.5)

    def px(i):
        return np.array([uv[i, 0] * n, (1.0 - uv[i, 1]) * n])

    mirror = mirror_indices(verts)

    def both(seq):
        return [tuple(seq), tuple(int(mirror[i]) for i in seq)]

    # Cheeks: warmer.
    for c in (CHEEK[0], CHEEK[1]):
        _blend_disc(
            tex, px(c), 30, np.asarray(a.skin) * np.array([1.06, 0.92, 0.9]), 0.5
        )
    # Under-eye: slightly darker.
    for ring in both(LOWER_LID_1):
        _blend_poly(tex, [px(i) for i in ring], 9, np.asarray(a.skin) * 0.88, 0.5)
    # Lid crease.
    for ring in both(UPPER_LID_1):
        _blend_line(
            tex, [px(i) for i in ring], 2.2, np.asarray(a.skin) * 0.7, 0.6, blur=2.0
        )
    # Lash line, upper heavy, lower light.
    for ring in both((EYE_OUTER,) + UPPER_LID + (EYE_INNER,)):
        _blend_line(
            tex,
            [px(i) for i in ring],
            2.2,
            np.array([0.03, 0.025, 0.02]),
            0.95,
            blur=0.7,
        )
    for ring in both((EYE_OUTER,) + LOWER_LID + (EYE_INNER,)):
        _blend_line(
            tex, [px(i) for i in ring], 1.0, np.array([0.12, 0.08, 0.07]), 0.6, blur=0.7
        )
    # Eyebrows, with hair-direction streaks.
    streak = 0.75 + 0.5 * _streaks(rng, n)
    for upper, lower in zip(both(BROW_UPPER), both(BROW_LOWER), strict=True):
        poly = [px(i) for i in upper] + [px(i) for i in reversed(lower)]
        mask = _poly_mask(n, poly, blur=2.0) * streak.clip(0, 1)
        tex = tex * (1 - 0.9 * mask[..., None]) + 0.9 * mask[..., None] * np.asarray(
            a.brows
        )
    # Lips and the seam between them.
    _blend_poly(tex, [px(i) for i in LIPS_OUTER], 1.5, np.asarray(a.lips), 0.9)
    _blend_line(
        tex, [px(i) for i in LIPS_SEAM], 1.6, np.asarray(a.lips) * 0.35, 0.9, blur=1.0
    )
    # Nostrils.
    for alar in NOSE_ALAR:
        p = 0.45 * px(alar) + 0.55 * px(NOSE_BOTTOM)
        _blend_disc(tex, p, 4.0, np.asarray(a.skin) * 0.25, 0.8)
    return np.clip(tex, 0.0, 1.0).astype(np.float32)


def _value_noise(rng, n: int, cells: int) -> np.ndarray:
    g = rng.random((cells + 1, cells + 1))
    x = np.linspace(0, cells, n, endpoint=False)
    i = x.astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    a = g[i][:, i]
    b = g[i][:, i + 1]
    c = g[i + 1][:, i]
    d = g[i + 1][:, i + 1]
    fx = f[None, :]
    fy = f[:, None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _streaks(rng, n: int) -> np.ndarray:
    """Short, roughly horizontal strokes: eyebrow hair."""
    base = rng.random((n // 2, n // 16))
    s = np.repeat(np.repeat(base, 16, axis=1), 2, axis=0)
    return _blur(s, 1.0)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, img)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)


def _poly_mask(n: int, poly, blur: float = 0.0) -> np.ndarray:
    poly = np.asarray(poly, float)
    ys, xs = np.mgrid[0:n, 0:n] + 0.5
    inside = np.zeros((n, n), dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1, strict=True):
        if b == d:
            continue
        cond = (b > ys) != (d > ys)
        xcross = a + (ys - b) * (c - a) / (d - b)
        inside ^= cond & (xs < xcross)
    return _blur(inside.astype(float), blur)


def _line_mask(n: int, pts, width: float, blur: float) -> np.ndarray:
    ys, xs = np.mgrid[0:n, 0:n] + 0.5
    d = np.full((n, n), np.inf)
    pts = np.asarray(pts, float)
    for p, q in zip(pts[:-1], pts[1:], strict=True):
        v = q - p
        ln = max(float(v @ v), 1e-9)
        t = np.clip(((xs - p[0]) * v[0] + (ys - p[1]) * v[1]) / ln, 0, 1)
        dist = np.hypot(xs - (p[0] + t * v[0]), ys - (p[1] + t * v[1]))
        d = np.minimum(d, dist)
    return _blur((d <= width / 2).astype(float), blur)


def _blend(tex, mask, colour, alpha):
    m = (mask * alpha)[..., None]
    tex[:] = tex * (1 - m) + m * np.asarray(colour)


def _blend_poly(tex, poly, blur, colour, alpha):
    _blend(tex, _poly_mask(tex.shape[0], poly, blur), colour, alpha)


def _blend_line(tex, pts, width, colour, alpha, blur=1.0):
    _blend(tex, _line_mask(tex.shape[0], pts, width, blur), colour, alpha)


def _blend_disc(tex, centre, radius, colour, alpha):
    n = tex.shape[0]
    ys, xs = np.mgrid[0:n, 0:n] + 0.5
    d2 = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2
    mask = np.exp(-0.5 * d2 / radius**2)
    _blend(tex, mask, colour, alpha)
