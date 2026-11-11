"""Mesh primitives. Every face is counter-clockwise seen from outside."""

from __future__ import annotations

import numpy as np

from .mathutil import normalize, rotation_between
from .raster import Mesh


def _grid_faces(rows: int, cols: int, wrap: bool) -> np.ndarray:
    """Quads between consecutive rings of ``cols`` vertices, as triangles."""
    faces = []
    ncol = cols if wrap else cols - 1
    for r in range(rows - 1):
        for c in range(ncol):
            a = r * cols + c
            b = r * cols + (c + 1) % cols
            d = (r + 1) * cols + c
            e = (r + 1) * cols + (c + 1) % cols
            faces.append((a, d, b))
            faces.append((b, d, e))
    return np.array(faces, dtype=np.int64)


def uv_sphere(
    centre=(0.0, 0.0, 0.0),
    radii=(1.0, 1.0, 1.0),
    rings: int = 16,
    segments: int = 24,
    material: int = 0,
    theta_max: float = np.pi,
) -> Mesh:
    """An ellipsoid with its pole along +z, so the "front" is easy to find.

    ``theta_max`` below pi gives a cap rather than a closed surface, which is
    all an eyeball needs: nothing behind the equator is ever visible.

    The attribute columns are the unit-sphere position, which shaders use as a
    local coordinate (the eyeball's iris is laid out in it).
    """
    theta = np.linspace(0.0, theta_max, rings + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    unit = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
    unit = unit.reshape(-1, 3)
    radii = np.asarray(radii, float)
    verts = unit * radii + np.asarray(centre, float)
    # Rows run from the +z pole outwards; with phi increasing counter-clockwise
    # about +z this ordering faces outwards.
    faces = _grid_faces(rings + 1, segments, wrap=True)
    normals = normalize(unit / radii**2)
    return Mesh(verts, faces, material=material, attrs=unit.copy(), normals=normals)


def box(lo, hi, material: int = 0, subdivisions: int = 1) -> Mesh:
    """An axis-aligned box. ``subdivisions`` splits each face into a grid.

    Large faces are subdivided so that a triangle partly behind the camera is
    dropped as a small piece rather than taking half a wall with it.
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    n = max(1, subdivisions)
    verts, faces, norms = [], [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = [a for a in range(3) if a != axis]
            if sign < 0:
                u, v = v, u
            g = np.linspace(0.0, 1.0, n + 1)
            gu, gv = np.meshgrid(g, g, indexing="ij")
            p = np.zeros((n + 1, n + 1, 3))
            p[..., axis] = hi[axis] if sign > 0 else lo[axis]
            p[..., u] = lo[u] + gu * (hi[u] - lo[u])
            p[..., v] = lo[v] + gv * (hi[v] - lo[v])
            base = sum(len(x) for x in verts)
            verts.append(p.reshape(-1, 3))
            nrm = np.zeros(3)
            nrm[axis] = sign
            norms.append(np.tile(nrm, ((n + 1) ** 2, 1)))
            f = _grid_faces(n + 1, n + 1, wrap=False) + base
            faces.append(f)
    verts = np.concatenate(verts)
    faces = np.concatenate(faces)
    norms = np.concatenate(norms)
    # Make every face wind outwards regardless of the axis bookkeeping above.
    v0, v1, v2 = (verts[faces[:, i]] for i in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    flip = np.einsum("ij,ij->i", fn, norms[faces[:, 0]]) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return Mesh(verts, faces, material=material, attrs=verts.copy(), normals=norms)


def tube(
    a,
    b,
    radius_a: float,
    radius_b: float | None = None,
    segments: int = 16,
    rings: int = 4,
    caps: bool = True,
    material: int = 0,
    scale_xy=(1.0, 1.0),
) -> Mesh:
    """A tapered tube from ``a`` to ``b`` with hemispherical-ish end caps.

    ``scale_xy`` flattens the cross-section, for a torso or a forearm.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    radius_b = radius_a if radius_b is None else radius_b
    axis = b - a
    length = float(np.linalg.norm(axis))
    R = rotation_between(np.array([0.0, 0.0, 1.0]), axis / max(length, 1e-9))
    sx, sy = scale_xy
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rows = []
    nrows = []
    cap_rings = 3 if caps else 0
    # Start cap, from the pole outwards.
    for k in range(cap_rings):
        ang = (k / cap_rings) * (np.pi / 2)
        r = radius_a * np.sin(ang)
        z = -radius_a * np.cos(ang) * 0.6
        rows.append(
            np.stack(
                [r * sx * np.cos(phi), r * sy * np.sin(phi), np.full_like(phi, z)], -1
            )
        )
        nrows.append(
            np.stack(
                [
                    np.sin(ang) * np.cos(phi) / sx,
                    np.sin(ang) * np.sin(phi) / sy,
                    np.full_like(phi, -np.cos(ang)),
                ],
                -1,
            )
        )
    for k in range(rings + 1):
        s = k / rings
        r = radius_a + (radius_b - radius_a) * s
        rows.append(
            np.stack(
                [
                    r * sx * np.cos(phi),
                    r * sy * np.sin(phi),
                    np.full_like(phi, s * length),
                ],
                -1,
            )
        )
        nrows.append(
            np.stack([np.cos(phi) / sx, np.sin(phi) / sy, np.zeros_like(phi)], -1)
        )
    for k in range(cap_rings - 1, -1, -1):
        ang = (k / cap_rings) * (np.pi / 2)
        r = radius_b * np.sin(ang)
        z = length + radius_b * np.cos(ang) * 0.6
        rows.append(
            np.stack(
                [r * sx * np.cos(phi), r * sy * np.sin(phi), np.full_like(phi, z)], -1
            )
        )
        nrows.append(
            np.stack(
                [
                    np.sin(ang) * np.cos(phi) / sx,
                    np.sin(ang) * np.sin(phi) / sy,
                    np.full_like(phi, np.cos(ang)),
                ],
                -1,
            )
        )
    local = np.concatenate(rows)
    nl = normalize(np.concatenate(nrows))
    faces = _grid_faces(len(rows), segments, wrap=True)[:, [0, 2, 1]]
    verts = local @ R.T + a
    normals = nl @ R.T
    return Mesh(verts, faces, material=material, attrs=local.copy(), normals=normals)


def quad(corners, material: int = 0, subdivisions: int = 8) -> Mesh:
    """A planar quad from four corners in counter-clockwise order from the front."""
    c = np.asarray(corners, float)
    n = subdivisions
    g = np.linspace(0.0, 1.0, n + 1)
    gu, gv = np.meshgrid(g, g, indexing="ij")
    p = (
        (1 - gu)[..., None] * (1 - gv)[..., None] * c[0]
        + gu[..., None] * (1 - gv)[..., None] * c[1]
        + gu[..., None] * gv[..., None] * c[2]
        + (1 - gu)[..., None] * gv[..., None] * c[3]
    ).reshape(-1, 3)
    faces = _grid_faces(n + 1, n + 1, wrap=False)
    normal = normalize(np.cross(c[1] - c[0], c[3] - c[0]))
    v0, v1, v2 = (p[faces[:, i]] for i in range(3))
    flip = np.einsum("ij,j->i", np.cross(v1 - v0, v2 - v0), normal) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return Mesh(
        p,
        faces,
        material=material,
        attrs=p.copy(),
        normals=np.tile(normal, (len(p), 1)),
    )


def loft(
    loop_a: np.ndarray, loop_b: np.ndarray, rows: int = 4, material: int = 0
) -> Mesh:
    """A surface joining two closed loops with the same number of points.

    Winding is fixed up afterwards against ``outward``-ness from the loops'
    centroid, which is right for the convex-ish skirts it is used for.
    """
    loop_a = np.asarray(loop_a, float)
    loop_b = np.asarray(loop_b, float)
    s = np.linspace(0.0, 1.0, rows + 1)[:, None, None]
    pts = (1 - s) * loop_a[None] + s * loop_b[None]
    n = loop_a.shape[0]
    verts = pts.reshape(-1, 3)
    faces = _grid_faces(rows + 1, n, wrap=True)
    centre = verts.mean(axis=0)
    v0, v1, v2 = (verts[faces[:, i]] for i in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    flip = np.einsum("ij,ij->i", fn, (v0 + v1 + v2) / 3 - centre) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return Mesh(verts, faces, material=material, attrs=verts.copy())
