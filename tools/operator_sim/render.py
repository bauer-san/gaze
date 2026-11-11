"""Render the operator at the saw as the RealSense would see it.

Produces exactly what :class:`gaze_monitor.capture.RealSenseSource` does: a
BGR uint8 colour image and a float32 depth image in metres, aligned to the
colour pixels, with 0.0 for "no reading". Plus the ground truth that no real
camera can give.

The sensor model is deliberately unflattering, because a simulator that is
cleaner than the hardware tunes the pipeline for conditions it will never
see:

* Colour: optical blur, shot and read noise, 8-bit quantisation.
* Depth: the D435's error grows with the square of distance (about a
  millimetre at the working distance), quantised to the 1 mm depth unit,
  spatially correlated, with the stereo occlusion shadow down the left side
  of anything nearer than its background, and random dropouts.

The head is supersampled. Eyes are where the measurement lives, and an
aliased iris edge stepping one pixel at a time is a noise source of the
renderer's, not the scene's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import materials as M
from .operator import PosedOperator
from .raster import Batch, GBuffer, rasterize
from .scene import Scene, camera_rotation

N_ATTRS = 10
# D435 stereo baseline.
BASELINE_M = 0.050
# RMS subpixel disparity error of the D435 depth engine, from Intel's
# published depth-error model.
SUBPIXEL = 0.08
MIN_DEPTH_M = 0.28
MAX_DEPTH_M = 6.0


@dataclass
class SensorModel:
    blur_px: float = 0.6
    read_noise: float = 1.2 / 255.0
    shot_noise: float = 0.012
    depth_noise: bool = True
    depth_dropout: float = 0.004
    exposure: float = 1.0
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed + 99)


@dataclass
class RenderResult:
    color: np.ndarray  # BGR uint8
    depth_m: np.ndarray  # float32 metres, 0 = no reading
    landmarks_px: np.ndarray  # (478, 2) ground-truth image positions
    landmarks_z: np.ndarray  # (478,) camera-frame depth
    iris_visible: np.ndarray  # (2,) bool: pupil centre not covered by a lid


class Renderer:
    def __init__(
        self,
        scene: Scene,
        sensor: SensorModel | None = None,
        ssaa: int = 2,
        static: bool = True,
        static_ssaa: int = 1,
    ):
        self.scene = scene
        # Supersampling of the room and the saw. Costs nothing per frame,
        # since they are rendered once; worth it for presentation renders,
        # where stair-stepped stainless edges are the first thing you see.
        self.static_ssaa = max(1, int(static_ssaa))
        self._floor = None
        self.sensor = sensor or SensorModel()
        self.ssaa = max(1, int(ssaa))
        mount = scene.mount
        self.width, self.height = mount.width, mount.height
        self.fx = (self.width / 2) / math.tan(math.radians(mount.hfov_deg) / 2)
        self.fy = (self.height / 2) / math.tan(math.radians(mount.vfov_deg) / 2)
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        self.R = camera_rotation(mount)
        self.C = np.asarray(mount.position, float)
        self.lights = M.Lighting().to_camera(self.R)
        self.appearance = None
        self.face_texture = None
        # The room and the saw never move, so they are rendered once. Skipped
        # for truth-only use, where nothing is rasterised at all.
        self._static = None
        if static:
            self._static = self._render_static()

    @property
    def _static_colour(self):
        if self._static is None:
            self._static = self._render_static()
        return self._static[0]

    @property
    def _static_depth(self):
        if self._static is None:
            self._static = self._render_static()
        return self._static[1]

    # -- geometry ------------------------------------------------------------

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.C) @ self.R.T

    def project(self, points: np.ndarray) -> np.ndarray:
        p = self.to_camera(points)
        return np.stack(
            [
                self.fx * p[..., 0] / p[..., 2] + self.cx,
                self.fy * p[..., 1] / p[..., 2] + self.cy,
            ],
            axis=-1,
        )

    def _gbuffer(self, meshes, fx, fy, cx, cy, w, h, cull=True) -> GBuffer:
        batch = Batch()
        for m in meshes:
            batch.add(m, N_ATTRS)
        V, N, F, Mt, A = batch.arrays()
        Vc = (V - self.C) @ self.R.T
        Nc = N @ self.R.T
        # Eye attrs 7:10 carry the visual axis in the world; shade wants it in
        # the camera frame.
        eye = np.zeros(len(V), dtype=bool)
        eye[F[Mt == M.EYE].ravel()] = True
        A[eye, 7:10] = A[eye, 7:10] @ self.R.T
        return rasterize(Vc, Nc, F, Mt, A, fx, fy, cx, cy, w, h, cull_backfaces=cull)

    def _shade(self, g: GBuffer, fx, fy, cx, cy) -> np.ndarray:
        yy, xx = np.divmod(g.pixel, g.width)
        pos = (
            np.stack(
                [(xx + 0.5 - cx) / fx, (yy + 0.5 - cy) / fy, np.ones(len(xx))], axis=1
            )
            * g.depth[:, None]
        )
        # Two-sided lighting for the static set, which is not culled.
        flip = np.einsum("ij,ij->i", g.normal, pos) > 0
        g.normal[flip] *= -1
        ctx = {
            "appearance": self.appearance,
            "face_texture": self.face_texture,
            "positions": pos,
        }
        return M.shade(g, self.lights, ctx)

    def _render_static(self):
        from .face import Appearance

        self.appearance = Appearance()
        s = self.static_ssaa
        W, H = self.width * s, self.height * s
        fx, fy, cx, cy = self.fx * s, self.fy * s, self.cx * s, self.cy * s
        g = self._gbuffer(self.scene.static_meshes(), fx, fy, cx, cy, W, H, cull=False)
        col = np.zeros((W * H, 3))
        col[g.pixel] = self._shade(g, fx, fy, cx, cy)
        depth = np.full(W * H, np.inf)
        depth[g.pixel] = g.depth
        # Where the floor is, in the world, for contact shadows.
        fxz = np.full((W * H, 2), np.nan)
        floor = g.material == M.FLOOR
        fxz[g.pixel[floor]] = g.attrs[floor][:, [0, 2]]
        col = col.reshape(H, W, 3)
        depth = depth.reshape(H, W)
        fxz = fxz.reshape(H, W, 2)
        if s > 1:
            h, w = self.height, self.width
            col = col.reshape(h, s, w, s, 3).mean(axis=(1, 3))
            depth = depth.reshape(h, s, w, s).min(axis=(1, 3))
            fxz = fxz[s // 2 :: s, s // 2 :: s]
        ok = np.isfinite(fxz[..., 0])
        self._floor = (np.nonzero(ok.ravel())[0], fxz[ok])
        return col, depth

    def contact_shadows(self, colour: np.ndarray, blobs) -> None:
        """Darken the floor under ``blobs``: (x, z, radius, strength) each.

        There are no cast shadows in this renderer, and without at least a
        soft contact shadow anything standing on the floor looks pasted on.
        """
        if not blobs or self._floor is None:
            return
        idx, xz = self._floor
        f = np.ones(len(idx))
        for x, z, r, k in blobs:
            d2 = (xz[:, 0] - x) ** 2 + (xz[:, 1] - z) ** 2
            f *= 1.0 - k * np.exp(-d2 / (r * r))
        flat = colour.reshape(-1, 3)
        flat[idx] *= f[:, None]

    # -- frames ----------------------------------------------------------------

    def render(
        self, posed: PosedOperator, appearance, face_texture, shadows=None
    ) -> RenderResult:
        self.appearance = appearance
        self.face_texture = face_texture
        W, H = self.width, self.height
        colour = self._static_colour.copy()
        depth = self._static_depth.copy()
        self.contact_shadows(colour, shadows)

        # Body at native resolution.
        if posed.body_meshes:
            g = self._gbuffer(
                posed.body_meshes, self.fx, self.fy, self.cx, self.cy, W, H
            )
            flat_c = colour.reshape(-1, 3)
            flat_d = depth.reshape(-1)
            nearer = g.depth < flat_d[g.pixel]
            if nearer.any():
                pix = g.pixel[nearer]
                sub = _subset(g, nearer)
                flat_c[pix] = self._shade(sub, self.fx, self.fy, self.cx, self.cy)
                flat_d[pix] = sub.depth

        # Head, supersampled inside its bounding box.
        verts = np.concatenate([m.vertices for m in posed.head_meshes])
        uv = self.project(verts)
        x0 = int(max(0, math.floor(uv[:, 0].min()) - 2))
        x1 = int(min(W, math.ceil(uv[:, 0].max()) + 2))
        y0 = int(max(0, math.floor(uv[:, 1].min()) - 2))
        y1 = int(min(H, math.ceil(uv[:, 1].max()) + 2))
        iris_visible = np.zeros(2, dtype=bool)
        lm_px = self.project(posed.landmarks_world)
        if x1 > x0 and y1 > y0 and np.all(self.to_camera(verts)[:, 2] > 0.05):
            s = self.ssaa
            bw, bh = (x1 - x0) * s, (y1 - y0) * s
            fx, fy = self.fx * s, self.fy * s
            cx, cy = (self.cx - x0) * s, (self.cy - y0) * s
            gh = self._gbuffer(posed.head_meshes, fx, fy, cx, cy, bw, bh)
            hc = np.zeros((bh * bw, 3))
            hd = np.full(bh * bw, np.inf)
            hm = np.zeros(bh * bw, dtype=np.int16)
            hc[gh.pixel] = self._shade(gh, fx, fy, cx, cy)
            hd[gh.pixel] = gh.depth
            hm[gh.pixel] = gh.material
            hc = hc.reshape(bh, bw, 3)
            hd = hd.reshape(bh, bw)
            hm = hm.reshape(bh, bw)
            base_d = np.repeat(np.repeat(depth[y0:y1, x0:x1], s, 0), s, 1)
            vis = hd < base_d
            cov = vis.reshape(y1 - y0, s, x1 - x0, s).mean(axis=(1, 3))
            head_sum = (
                (hc * vis[..., None])
                .reshape(y1 - y0, s, x1 - x0, s, 3)
                .sum(axis=(1, 3))
            )
            region = colour[y0:y1, x0:x1]
            colour[y0:y1, x0:x1] = region * (1 - cov[..., None]) + head_sum / (s * s)
            dsum = np.where(vis, hd, 0).reshape(y1 - y0, s, x1 - x0, s).sum(axis=(1, 3))
            cnt = vis.reshape(y1 - y0, s, x1 - x0, s).sum(axis=(1, 3))
            hdepth = np.where(cnt > 0, dsum / np.maximum(cnt, 1), np.inf)
            take = cov >= 0.5
            depth[y0:y1, x0:x1] = np.where(take, hdepth, depth[y0:y1, x0:x1])
            # Is each pupil centre actually showing between the lids?
            for k, idx in enumerate((468, 473)):
                px = (lm_px[idx, 0] - x0) * s
                py = (lm_px[idx, 1] - y0) * s
                if 0 <= px < bw and 0 <= py < bh:
                    iris_visible[k] = hm[int(py), int(px)] == M.EYE

        color, depth_m = self._sensor(colour, depth)
        lm_cam = self.to_camera(posed.landmarks_world)
        return RenderResult(
            color=color,
            depth_m=depth_m,
            landmarks_px=lm_px,
            landmarks_z=lm_cam[:, 2],
            iris_visible=iris_visible,
        )

    # -- sensor ------------------------------------------------------------------

    def _sensor(self, colour: np.ndarray, depth: np.ndarray):
        sn = self.sensor
        rng = sn.rng
        img = (colour * sn.exposure).astype(np.float32)
        img = _gauss_blur(img, sn.blur_px)
        img = M.srgb(img).astype(np.float32)
        if sn.read_noise > 0 or sn.shot_noise > 0:
            sigma = np.sqrt(sn.read_noise**2 + (sn.shot_noise**2) * img)
            img = img + rng.standard_normal(size=img.shape, dtype=np.float32) * sigma
        bgr = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)[..., ::-1]

        d = depth.copy()
        d[~np.isfinite(d)] = 0.0
        if sn.depth_noise:
            z = np.where(d > 0, d, 1.0)
            sigma = z * z * SUBPIXEL / (self.fx * BASELINE_M)
            field_ = _box3(rng.standard_normal(size=d.shape, dtype=np.float32))
            d = d + field_ * sigma
            d = np.round(d * 1000.0) / 1000.0  # 1 mm depth unit
            d[
                _stereo_shadow(np.where(depth > 0, depth, np.inf), self.fx * BASELINE_M)
            ] = 0.0
            d[rng.random(d.shape, dtype=np.float32) < sn.depth_dropout] = 0.0
        d[(d < MIN_DEPTH_M) | (d > MAX_DEPTH_M)] = 0.0
        return np.ascontiguousarray(bgr), d.astype(np.float32)


def _subset(g: GBuffer, mask: np.ndarray) -> GBuffer:
    return GBuffer(
        width=g.width,
        height=g.height,
        pixel=g.pixel[mask],
        depth=g.depth[mask],
        normal=g.normal[mask],
        material=g.material[mask],
        attrs=g.attrs[mask],
    )


def _gauss_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    r = max(1, int(math.ceil(2.5 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.zeros_like(img)
    pad = np.pad(img, ((0, 0), (r, r), (0, 0)), mode="edge")
    for i, w in enumerate(k):
        out += w * pad[:, i : i + img.shape[1]]
    pad = np.pad(out, ((r, r), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * pad[i : i + img.shape[0]]
    return out


def _box3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    s = sum(
        p[i : i + a.shape[0], j : j + a.shape[1]] for i in range(3) for j in range(3)
    )
    return s / 3.0  # unit variance for 9 independent samples


def _stereo_shadow(depth: np.ndarray, fB: float) -> np.ndarray:
    """Pixels the right imager cannot see: the D435's occlusion shadow.

    With the left imager as the reference, a point is hidden from the right
    one when something nearer, further right in the image, has a disparity
    large enough to cover it. That leaves an invalid band down the left side
    of every foreground edge, as wide as the disparity step.
    """
    disp = fB / depth  # inf depth -> 0 disparity
    h, w = disp.shape
    shadow = np.zeros_like(disp, dtype=bool)
    reach = np.full(h, -np.inf)
    for x in range(w - 1, -1, -1):
        shadow[:, x] = reach > disp[:, x] + 0.5
        reach = np.maximum(reach - 1.0, disp[:, x])
    return shadow
