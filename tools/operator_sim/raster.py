"""A vectorised triangle rasteriser in numpy.

No OpenGL, no GPU and no compiled extension: the simulator has to run in CI
and on a headless Jetson, where neither a display nor a GL context can be
assumed. It is a deferred renderer. Rasterisation produces, for every pixel,
the nearest triangle and its perspective-correct barycentric coordinates,
from which any per-vertex attribute is then interpolated. Shading happens
afterwards on whole images at once.

Everything is vectorised over (triangle, pixel) pairs. Each triangle's
bounding box is expanded into its candidate pixels, the edge functions are
evaluated on all of them together, and the depth test is a scatter-min. For
the few thousand triangles of an operator at camera distance that is tens of
milliseconds, which is fast enough to generate minutes of footage offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Candidate (triangle, pixel) pairs evaluated per batch. Bounds peak memory
# when a few triangles cover most of the frame, as walls and floors do.
CHUNK = 3_000_000
NEAR = 0.02


@dataclass
class Mesh:
    """Triangles with per-vertex attributes and one material per triangle.

    ``attrs`` columns are interpolated across each triangle; what they mean is
    up to the material that shades them (texture coordinates, a position in
    some local frame, and so on). ``normals`` default to smooth vertex normals
    computed from the faces.
    """

    vertices: np.ndarray
    faces: np.ndarray
    material: int | np.ndarray = 0
    attrs: np.ndarray | None = None
    normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.normals is None:
            self.normals = vertex_normals(self.vertices, self.faces)

    def transformed(self, rotation: np.ndarray, translation: np.ndarray) -> Mesh:
        """A copy with vertices mapped by ``R @ v + t`` and normals by ``R``."""
        return Mesh(
            vertices=self.vertices @ rotation.T + translation,
            faces=self.faces,
            material=self.material,
            attrs=self.attrs,
            normals=self.normals @ rotation.T,
        )


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth normals. Faces are counter-clockwise from outside."""
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    n = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


@dataclass
class Batch:
    """Several meshes concatenated, so one rasterisation pass covers them all."""

    vertices: list = field(default_factory=list)
    normals: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    attrs: list = field(default_factory=list)
    count: int = 0

    def add(self, mesh: Mesh, n_attrs: int) -> None:
        nv = len(mesh.vertices)
        self.vertices.append(mesh.vertices)
        self.normals.append(mesh.normals)
        self.faces.append(mesh.faces + self.count)
        mat = np.broadcast_to(
            np.asarray(mesh.material, dtype=np.int16), (len(mesh.faces),)
        )
        self.materials.append(np.array(mat))
        a = np.zeros((nv, n_attrs))
        if mesh.attrs is not None:
            k = min(n_attrs, mesh.attrs.shape[1])
            a[:, :k] = mesh.attrs[:, :k]
        self.attrs.append(a)
        self.count += nv

    def arrays(self):
        return (
            np.concatenate(self.vertices),
            np.concatenate(self.normals),
            np.concatenate(self.faces),
            np.concatenate(self.materials),
            np.concatenate(self.attrs),
        )


@dataclass
class GBuffer:
    """Per-pixel geometry, flat arrays indexed by ``pixel``."""

    width: int
    height: int
    pixel: np.ndarray  # linear pixel index, y * width + x
    depth: np.ndarray  # camera-space z, metres
    normal: np.ndarray  # camera-space unit normal
    material: np.ndarray
    attrs: np.ndarray

    def depth_image(self, background: float = 0.0) -> np.ndarray:
        img = np.full(self.width * self.height, background, dtype=np.float32)
        img[self.pixel] = self.depth
        return img.reshape(self.height, self.width)


def rasterize(
    cam_vertices: np.ndarray,
    cam_normals: np.ndarray,
    faces: np.ndarray,
    materials: np.ndarray,
    attrs: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    cull_backfaces: bool = True,
) -> GBuffer:
    """Rasterise camera-space triangles into a :class:`GBuffer`.

    The camera looks down +z with OpenCV's axes. Triangles crossing the near
    plane are dropped rather than clipped; nothing the simulator renders
    comes within ``NEAR`` of the lens.
    """
    p0, p1, p2 = (cam_vertices[faces[:, i]] for i in range(3))
    keep = (p0[:, 2] > NEAR) & (p1[:, 2] > NEAR) & (p2[:, 2] > NEAR)
    if cull_backfaces:
        n = np.cross(p1 - p0, p2 - p0)
        keep &= np.einsum("ij,ij->i", n, p0) < 0.0
    tri_ids = np.nonzero(keep)[0]

    z = cam_vertices[:, 2]
    sx = fx * cam_vertices[:, 0] / np.where(z > NEAR, z, 1.0) + cx
    sy = fy * cam_vertices[:, 1] / np.where(z > NEAR, z, 1.0) + cy

    f = faces[tri_ids]
    X = sx[f]
    Y = sy[f]
    Z = z[f]
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (
        Y[:, 1] - Y[:, 0]
    )
    ok = np.abs(area) > 1e-9
    # Pixel i has its centre at i + 0.5.
    y0 = np.maximum(np.ceil(Y.min(axis=1) - 0.5), 0).astype(np.int64)
    y1 = np.minimum(np.floor(Y.max(axis=1) - 0.5), height - 1).astype(np.int64)
    xmin = X.min(axis=1)
    xmax = X.max(axis=1)
    ok &= (y1 >= y0) & (xmax >= 0.5) & (xmin <= width - 0.5)
    sel = np.nonzero(ok)[0]
    tri_ids, X, Y, Z, area, y0, y1 = (a[sel] for a in (tri_ids, X, Y, Z, area, y0, y1))

    # Screen-space barycentrics are affine in (x, y): w_i = A_i x + B_i y + C_i.
    # So is 1/z, which is what makes perspective-correct depth cheap: the
    # depth test needs one dot product per candidate pixel, and barycentrics
    # are only worked out for the fragments that win it.
    inv = 1.0 / area
    A = (
        np.stack([Y[:, 1] - Y[:, 2], Y[:, 2] - Y[:, 0], Y[:, 0] - Y[:, 1]], 1)
        * inv[:, None]
    )
    B = (
        np.stack([X[:, 2] - X[:, 1], X[:, 0] - X[:, 2], X[:, 1] - X[:, 0]], 1)
        * inv[:, None]
    )
    C = (
        np.stack(
            [
                X[:, 1] * Y[:, 2] - X[:, 2] * Y[:, 1],
                X[:, 2] * Y[:, 0] - X[:, 0] * Y[:, 2],
                X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0],
            ],
            1,
        )
        * inv[:, None]
    )
    iz = 1.0 / Z
    za = (A * iz).sum(1)
    zb = (B * iz).sum(1)
    zc = (C * iz).sum(1)

    # Scanline spans: one (triangle, row) pair per covered row, its x extent
    # from where the row's centre line crosses the triangle's edges.
    rows = y1 - y0 + 1
    rt = np.repeat(np.arange(len(tri_ids)), rows)
    ry = np.repeat(y0, rows) + (
        np.arange(int(rows.sum())) - np.repeat(np.cumsum(rows) - rows, rows)
    )
    yc = ry + 0.5
    xl = np.full(len(rt), np.inf)
    xr = np.full(len(rt), -np.inf)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        ya, yb = Y[rt, i], Y[rt, j]
        xa, xb = X[rt, i], X[rt, j]
        lo = np.minimum(ya, yb)
        hi = np.maximum(ya, yb)
        cross = (yc >= lo) & (yc <= hi) & (hi > lo)
        xi = xa + (yc - ya) * (xb - xa) / np.where(hi > lo, yb - ya, 1.0)
        xl = np.where(cross, np.minimum(xl, xi), xl)
        xr = np.where(cross, np.maximum(xr, xi), xr)
    px0 = np.maximum(np.ceil(xl - 0.5), 0)
    px1 = np.minimum(np.floor(xr - 0.5), width - 1)
    good = np.isfinite(xl) & (px1 >= px0)
    rt, ry, px0, px1 = (
        rt[good],
        ry[good],
        px0[good].astype(np.int64),
        px1[good].astype(np.int64),
    )
    counts = px1 - px0 + 1

    out_pix, out_tri, out_iz = [], [], []
    cum = np.cumsum(counts)
    start = 0
    n = len(rt)
    while start < n:
        base = cum[start - 1] if start > 0 else 0
        stop = max(int(np.searchsorted(cum, base + CHUNK, side="right")), start + 1)
        c = counts[start:stop]
        span = np.repeat(np.arange(start, stop), c)
        first = np.cumsum(c) - c
        px = px0[span] + (np.arange(int(c.sum())) - np.repeat(first, c))
        py = ry[span]
        t = rt[span]
        izp = za[t] * (px + 0.5) + zb[t] * (py + 0.5) + zc[t]
        lin = py * width + px
        lin, t, izp = _nearest(lin, t, izp, width * height)
        out_pix.append(lin)
        out_tri.append(t)
        out_iz.append(izp)
        start = stop

    if out_pix:
        lin = np.concatenate(out_pix)
        t = np.concatenate(out_tri)
        izp = np.concatenate(out_iz)
        if len(out_pix) > 1:
            lin, t, izp = _nearest(lin, t, izp, width * height)
    else:
        lin = np.zeros(0, np.int64)
        t = np.zeros(0, np.int64)
        izp = np.zeros(0)

    yy, xx = np.divmod(lin, width)
    fxp = xx + 0.5
    fyp = yy + 0.5
    w = np.clip(A[t] * fxp[:, None] + B[t] * fyp[:, None] + C[t], 0.0, 1.0)
    pw = w * iz[t]
    bary = pw / np.maximum(pw.sum(1), 1e-12)[:, None]
    depth = 1.0 / izp

    fv = faces[tri_ids[t]]
    normal = np.einsum("pk,pkj->pj", bary, cam_normals[fv])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    interp = np.einsum("pk,pkj->pj", bary, attrs[fv])
    return GBuffer(
        width=width,
        height=height,
        pixel=lin,
        depth=depth,
        normal=normal,
        material=materials[tri_ids[t]],
        attrs=interp,
    )


def _nearest(lin, tri, inv_z, size):
    """Keep the nearest fragment (largest 1/z) per pixel."""
    zbuf = np.zeros(size)
    np.maximum.at(zbuf, lin, inv_z)
    win = inv_z >= zbuf[lin]
    lin, tri, inv_z = lin[win], tri[win], inv_z[win]
    _, first = np.unique(lin, return_index=True)
    return lin[first], tri[first], inv_z[first]
