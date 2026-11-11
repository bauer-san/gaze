"""A short promotional video of the operator at the saw.

A 3/4 view of the operator and the saw, with the monitor's view of things
laid over it:

* the blade zone, tinted by the monitor's state;
* the operator's line of sight, from the simulation's ground truth;
* a status badge driven by **the real pipeline**: the face camera's frames
  are rendered, FaceMesh runs on them, and the package's own
  ``GazeMeasurer``, ``GazeCalibrator`` and ``AttentionMonitor`` decide the
  state, exactly as :mod:`.evaluate` replays them. Nothing in the badge is
  scripted;
* an inset of what the face camera sees, with FaceMesh's landmarks. The
  camera itself is not drawn: in production it is built into the saw head.

    python -m tools.operator_sim promo --out docs/media/operator-promo.mp4

writes the full 1080p video, a small copy sized for a text message beside
it (``*-sms.mp4``), and a poster frame (``*.jpg``). Needs the deployment
stack (MediaPipe, OpenCV), Pillow, and ffmpeg with libx264.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from .behaviour import Segment, calibration_segments
from .evaluate import Run, generate, replay
from .render import Renderer, SensorModel
from .scene import BLADE_ZONE, CameraMount, Scene

W, H = 1920, 1080
FPS = 30
# The 3/4 view: in front of the operator and to their left, a little above
# head height, looking across the carriage at the operator and the saw.
VIEW = dict(position=(-2.0, 1.95, -0.55), target=(-0.08, 1.18, 0.30), hfov_deg=52.0)

# The blade zone drawn as a volume around the blade: the calibrated zone is
# a tilted plane that this view sees nearly edge-on, which reads as nothing.
_C = BLADE_ZONE.corners
ZONE_BOX = np.array(
    [
        [x, y, z]
        for x in (_C[0][0], _C[1][0])
        for y in (0.93, _C[0][1])
        for z in (-0.04, 0.16)
    ]
)

GREEN = (46, 204, 113)
AMBER = (243, 156, 18)
RED = (231, 76, 60)
BLUE = (52, 152, 219)
STATE_STYLE = {
    "attentive": ("WATCHING THE BLADE", GREEN),
    "warning": ("EYES OFF THE BLADE", AMBER),
    "alert": ("ATTENTION LOST", RED),
    "fault": ("SENSOR FAULT", RED),
}


def promo_segments() -> list[Segment]:
    """Commissioning, then a short, readable stretch of work.

    Long enough at each step for the state machine to do its thing in real
    time: a glance shorter than the 1 s grace period, then a conversation
    long enough to pass the 3 s alert threshold, then back to work.
    """
    work = [
        Segment("cut", 5.0, scan=0.02, activity="watch_cut"),
        Segment("tray", 0.5, activity="glance"),
        Segment("cut", 3.7, scan=0.02, activity="watch_cut"),
        Segment("coworker_left", 4.6, head="face", hands="rest", activity="talk"),
        Segment("cut", 5.2, scan=0.02, activity="watch_cut"),
    ]
    return calibration_segments(head="still") + work


def promo_mount() -> CameraMount:
    vfov = math.degrees(
        2 * math.atan(math.tan(math.radians(VIEW["hfov_deg"] / 2)) * H / W)
    )
    return CameraMount(
        position=VIEW["position"],
        target=VIEW["target"],
        width=W,
        height=H,
        hfov_deg=VIEW["hfov_deg"],
        vfov_deg=vfov,
    )


# -- the 3/4 view, rendered on a pool --------------------------------------------

_W = None


def _init_worker(seed: int) -> None:
    global _W
    from .operator import Operator

    op = Operator()
    sensor = SensorModel(
        blur_px=0.45,
        read_noise=0.5 / 255,
        shot_noise=0.003,
        depth_noise=False,
        seed=seed,
    )
    rend = Renderer(Scene(mount=promo_mount()), sensor, ssaa=2, static_ssaa=2)
    _W = (op, rend)


def _render_view(job):
    index, state = job
    op, rend = _W
    rend.sensor.rng = np.random.default_rng(index)
    posed = op.pose(state)
    # The whole operator goes through the supersampled pass: at this size
    # the body's edges show aliasing as much as the face's do.
    posed = dataclasses.replace(
        posed, head_meshes=posed.head_meshes + posed.body_meshes, body_meshes=[]
    )
    x, z = state.stance
    shadows = [
        (x - 0.12, z + 0.06, 0.13, 0.55),
        (x + 0.12, z + 0.06, 0.13, 0.55),
        (x, z, 0.40, 0.30),
        (0.18, -0.03, 0.42, 0.45),
        (-0.30, 0.0, 0.45, 0.25),
        (0.80, 0.35, 0.45, 0.35),
    ]
    r = rend.render(posed, op.appearance, op.face.texture, shadows=shadows)
    # The beam ends where the operator is looking if that is on the saw,
    # and at a fixed length otherwise, so a look at someone off-screen does
    # not throw a line across the frame.
    d = state.fixation - state.cyclopean
    dist = float(np.linalg.norm(d))
    end = state.fixation if dist < 1.2 else state.cyclopean + d / dist * 0.9
    proj = {
        "zone": rend.project(BLADE_ZONE.corners),
        "zone_box": rend.project(ZONE_BOX),
        "eyes": rend.project(state.cyclopean[None])[0],
        "beam_end": rend.project(end[None])[0],
    }
    return index, r.color, proj


# -- overlays ----------------------------------------------------------------------


class _Fonts:
    def __init__(self):
        from PIL import ImageFont

        def font(names, size):
            for n in names:
                for d in (
                    "/usr/share/fonts/opentype/inter",
                    "/usr/share/fonts/truetype/dejavu",
                ):
                    p = os.path.join(d, n)
                    if os.path.exists(p):
                        return ImageFont.truetype(p, size)
            return ImageFont.load_default()

        bold = ["Inter-Bold.otf", "DejaVuSans-Bold.ttf"]
        semi = ["Inter-SemiBold.otf", "DejaVuSans-Bold.ttf"]
        reg = ["Inter-Regular.otf", "DejaVuSans.ttf"]
        self.title = font(bold, 84)
        self.subtitle = font(reg, 38)
        self.badge = font(bold, 34)
        self.small = font(semi, 24)
        self.caption = font(semi, 36)
        self.tiny = font(reg, 22)


def _lerp(a, b, t):
    return a + (b - a) * t


def _fade(t, t0, t1, ramp=0.4):
    """1 inside [t0, t1], ramping in and out over ``ramp`` seconds."""
    if t < t0 or t > t1:
        return 0.0
    return min(1.0, (t - t0) / ramp, (t1 - t) / ramp)


def _overlay(img, info, fonts, inset=None):
    """Draw the zone, gaze beam, badge, captions and inset. Returns RGB uint8."""
    import cv2
    from PIL import Image, ImageDraw

    frame = img[..., ::-1].copy()  # BGR -> RGB
    layer = np.zeros_like(frame)
    alpha = np.zeros(frame.shape[:2], np.float32)

    def blend(colour, mask, a):
        nonlocal layer, alpha
        m = mask.astype(np.float32) * a
        layer = np.where(m[..., None] > alpha[..., None], colour, layer)
        alpha = np.maximum(alpha, m)

    state = info.get("state")
    colour = STATE_STYLE.get(state, ("", BLUE))[1] if state else BLUE
    # The zone, as a quad on the saw.
    if info.get("show_zone", True):
        z = info["proj"]["zone"]
        poly = cv2.convexHull(np.asarray(info["proj"]["zone_box"], np.float32)).astype(
            np.int32
        )
        m = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(m, [poly], 1, cv2.LINE_AA)
        blend(np.array(colour, np.uint8), m, 0.30)
        frame = (frame * (1 - alpha[..., None]) + layer * alpha[..., None]).astype(
            np.uint8
        )
        cv2.polylines(frame, [poly], True, colour, 3, cv2.LINE_AA)
        for k, p in enumerate(z):
            done = k < info.get("corners_done", 4)
            active = info.get("calibrating") and k == info.get("corners_done", 4)
            if info.get("calibrating"):
                r = 14 if active else 10
                cv2.circle(
                    frame,
                    tuple(np.int32(p)),
                    r,
                    (255, 255, 255) if active else colour,
                    -1 if (done or active) else 2,
                    cv2.LINE_AA,
                )
    # The line of sight.
    if info.get("show_gaze", True):
        e, f = info["proj"]["eyes"], info["proj"]["beam_end"]
        beam = GREEN if info.get("true_in_zone") else (255, 255, 255)
        glow = frame.copy()
        cv2.line(glow, tuple(np.int32(e)), tuple(np.int32(f)), beam, 9, cv2.LINE_AA)
        frame = cv2.addWeighted(glow, 0.35, frame, 0.65, 0)
        cv2.line(frame, tuple(np.int32(e)), tuple(np.int32(f)), beam, 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(np.int32(f)), 7, beam, -1, cv2.LINE_AA)

    if inset is not None:
        ih, iw = inset.shape[:2]
        x0, y0 = W - iw - 40, H - ih - 40
        shadow = (
            frame[y0 + 8 : y0 + ih + 8, x0 + 8 : x0 + iw + 8].astype(np.float32) * 0.55
        )
        frame[y0 + 8 : y0 + ih + 8, x0 + 8 : x0 + iw + 8] = shadow.astype(np.uint8)
        frame[y0 : y0 + ih, x0 : x0 + iw] = inset
        cv2.rectangle(
            frame, (x0, y0), (x0 + iw - 1, y0 + ih - 1), (255, 255, 255), 2, cv2.LINE_AA
        )

    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil, "RGBA")
    if inset is not None:
        d.rectangle((x0 + 2, y0 + 2, x0 + iw - 3, y0 + 36), fill=(15, 20, 28, 170))
        d.text(
            (x0 + 12, y0 + 8),
            "FACE CAMERA  ·  hidden in the saw head",
            font=fonts.tiny,
            fill=(255, 255, 255, 230),
        )
    # Status badge.
    if info.get("badge"):
        text, col = info["badge"]
        tw = d.textlength(text, font=fonts.badge)
        x, y = 48, 44
        d.rounded_rectangle((x, y, x + tw + 84, y + 70), radius=35, fill=col + (235,))
        d.ellipse((x + 22, y + 24, x + 44, y + 46), fill=(255, 255, 255, 255))
        d.text((x + 58, y + 15), text, font=fonts.badge, fill=(255, 255, 255, 255))
        if info.get("badge_note"):
            d.text(
                (x + 6, y + 82),
                info["badge_note"],
                font=fonts.small,
                fill=(255, 255, 255, 235),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 160),
            )
    # Caption, lower left.
    cap = info.get("caption")
    if cap and info.get("caption_alpha", 1) > 0:
        a = int(255 * info.get("caption_alpha", 1))
        lines = cap.split("\n")
        widths = [d.textlength(line, font=fonts.caption) for line in lines]
        bx, by = 48, H - 48 - 54 * len(lines) - 24
        d.rounded_rectangle(
            (bx, by, bx + max(widths) + 48, H - 48),
            radius=16,
            fill=(15, 20, 28, int(0.72 * a)),
        )
        for k, line in enumerate(lines):
            d.text(
                (bx + 24, by + 14 + 54 * k),
                line,
                font=fonts.caption,
                fill=(255, 255, 255, a),
            )
    # Title / end cards.
    card = info.get("card")
    if card:
        a = info.get("card_alpha", 1.0)
        dim = Image.new("RGBA", pil.size, (10, 14, 20, int(170 * a)))
        pil = Image.alpha_composite(pil.convert("RGBA"), dim)
        d = ImageDraw.Draw(pil, "RGBA")
        title, sub = card
        tw = d.textlength(title, font=fonts.title)
        d.text(
            ((W - tw) / 2, H / 2 - 110),
            title,
            font=fonts.title,
            fill=(255, 255, 255, int(255 * a)),
        )
        for k, line in enumerate(sub.split("\n")):
            sw = d.textlength(line, font=fonts.subtitle)
            d.text(
                ((W - sw) / 2, H / 2 + 10 + 56 * k),
                line,
                font=fonts.subtitle,
                fill=(220, 228, 236, int(255 * a)),
            )
        pil = pil.convert("RGB")
    return np.asarray(pil)


def _inset(run_dir: pathlib.Path, run: Run, i: int, box, size=(560, 420)) -> np.ndarray:
    """The face camera's frame, cropped to the face, with FaceMesh's points."""
    import cv2

    img = cv2.imread(str(run_dir / "frames" / f"{i:06d}.jpg"))
    x0, y0, x1, y1 = box
    lm = run.mp_landmarks[i]
    if np.isfinite(lm[0, 0]):
        w, h = run.meta["width"], run.meta["height"]
        pts = lm[:, :2] * [w, h]
        for k in list(range(0, 468, 3)):
            cv2.circle(
                img, tuple(np.int32(pts[k])), 1, (190, 230, 255), -1, cv2.LINE_AA
            )
        for c in (468, 473):
            ring = pts[c + 1 : c + 5]
            r = float(np.mean(np.linalg.norm(ring - pts[c], axis=1)))
            cv2.circle(
                img,
                tuple(np.int32(pts[c])),
                max(2, int(round(r))),
                (113, 204, 46),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(img, tuple(np.int32(pts[c])), 2, (113, 204, 46), -1, cv2.LINE_AA)
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)[..., ::-1]


def _face_box(run: Run, aspect=4 / 3, pad=1.35):
    w, h = run.meta["width"], run.meta["height"]
    px = run.truth_landmarks_px[:, :468]
    mon = np.array([tr["activity"] != "calibrate" for tr in run.truth])
    lo = np.percentile(px[mon].min(axis=1), 10, axis=0)
    hi = np.percentile(px[mon].max(axis=1), 90, axis=0)
    c = (lo + hi) / 2
    bh = (hi - lo)[1] * pad
    bw = bh * aspect
    x0 = int(np.clip(c[0] - bw / 2, 0, w - bw))
    y0 = int(np.clip(c[1] - bh / 2, 0, h - bh))
    return x0, y0, int(x0 + bw), int(y0 + bh)


# -- the cut -----------------------------------------------------------------------


def _timeline(run: Run, res: dict) -> list[dict]:
    """Output frames: which simulated frame each shows, and what is laid over it."""
    truth = run.truth
    n = len(truth)
    calib_end = next(i for i, tr in enumerate(truth) if tr["activity"] != "calibrate")
    shots = []
    # Title: the first second and a half of commissioning, at half speed.
    for k in range(int(3.5 * FPS)):
        shots.append(
            {"sim": min(k // 2, calib_end - 1), "phase": "title", "t": k / FPS}
        )
    # Commissioning, three times as fast.
    for k, i in enumerate(range(45, calib_end, 3)):
        shots.append({"sim": i, "phase": "calibrate", "t": k / FPS})
    # Work, in real time.
    for k, i in enumerate(range(calib_end, n)):
        shots.append({"sim": i, "phase": "work", "t": k / FPS})
    # End card over the last frame.
    for k in range(int(4.0 * FPS)):
        shots.append({"sim": n - 1, "phase": "end", "t": k / FPS})

    work_t = [tr["t"] - truth[calib_end]["t"] for tr in truth]
    captions = _captions(truth, calib_end)
    out = []
    for s in shots:
        i = s["sim"]
        info = {
            "sim": i,
            "true_in_zone": bool(res["true_in_zone"][i]) or s["phase"] != "work",
        }
        tr = truth[i]
        u, v = tr["zone_u"], tr["zone_v"]
        if s["phase"] != "work":
            info["true_in_zone"] = (
                math.isfinite(u) and -0.1 <= u <= 1.1 and -0.1 <= v <= 1.1
            )
        if s["phase"] in ("title", "calibrate"):
            info.update(
                calibrating=True,
                corners_done=int(res["corners_done"][i]),
                badge=("COMMISSIONING", BLUE),
                show_gaze=s["phase"] == "calibrate",
            )
            info["badge_note"] = (
                f"corner {min(int(res['corners_done'][i]) + 1, 4)} of 4"
            )
            if s["phase"] == "title":
                info["card"] = (
                    "Operator Attention Monitor",
                    "Gaze monitoring at the band saw\nfrom a single depth camera",
                )
                info["card_alpha"] = _fade(s["t"], 0.0, 3.5, 0.5)
                info["show_zone"] = False
                info.pop("badge")
            else:
                info["caption"] = (
                    "Commissioning, once: look at each corner\nof the blade zone"
                )
                info["caption_alpha"] = _fade(s["t"], 0.0, 99, 0.4)
        elif s["phase"] == "work":
            st = res["state"][i]
            st = st if isinstance(st, str) and st else "attentive"
            info["state"] = st
            text, col = STATE_STYLE[st]
            info["badge"] = (text, col)
            if st != "attentive":
                info["badge_note"] = f"eyes off the blade {res['away'][i]:.1f} s"
            cap, a = captions(work_t[i])
            info["caption"], info["caption_alpha"] = cap, a
            info["inset"] = True
        else:
            st = res["state"][i]
            info["state"] = st if isinstance(st, str) and st else "attentive"
            info["card"] = (
                "An attention aid, not a machine guard",
                "Eyes tracked independently of the head  ·  one depth camera\n"
                "Alerts when the operator's attention leaves the blade",
            )
            info["card_alpha"] = _fade(s["t"], 0.0, 99, 0.6)
            info["show_gaze"] = False
        out.append(info)
    return out


def _captions(truth, calib_end):
    """Caption for each moment of the work section, keyed off the script."""
    spans = []
    t0 = truth[calib_end]["t"]
    cur = None
    for tr in truth[calib_end:]:
        a = tr["activity"]
        if cur is None or a != cur[0]:
            cur = [a, tr["t"] - t0, tr["t"] - t0]
            spans.append(cur)
        cur[2] = tr["t"] - t0
    text = {
        "watch_cut": "Watching the blade",
        "glance": "A quick glance away is normal:\nno alarm inside the grace period",
        "talk": "Eyes off the blade: a warning after 1 s,\nan alert after 3 s",
    }
    seen_cut = 0
    keyed = []
    for a, s0, s1 in spans:
        if a == "watch_cut":
            seen_cut += 1
            msg = (
                text[a]
                if seen_cut == 1
                else (
                    "Back on the blade: the alarm clears" if seen_cut >= 3 else text[a]
                )
            )
        else:
            msg = text.get(a, "")
        # The glance caption stays up a little longer than the glance itself.
        hold = 1.6 if a == "glance" else 0.0
        keyed.append((s0, s1 + hold, msg))

    def at(t):
        # A glance or a conversation caption is held a little past its own
        # span, and wins over the "watching" caption it overlaps.
        best, rank = ("", 0.0), -1
        for s0, s1, msg in keyed:
            if s0 <= t <= s1 and msg:
                r = 0 if msg.startswith(("Watching", "Back on")) else 1
                if r >= rank:
                    best, rank = (msg, _fade(t, s0, s1, 0.35)), r
        return best

    return at


def render_promo(
    out: pathlib.Path, seed: int = 3, workers: int = 0, keep: pathlib.Path | None = None
) -> dict:
    """Render the promo. Returns the paths written."""
    import multiprocessing as mp

    import cv2

    from .sim import Simulation

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to encode the video")
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    segments = promo_segments()
    work = (
        pathlib.Path(keep) if keep else pathlib.Path(tempfile.mkdtemp(prefix="promo-"))
    )

    # 1. The face camera: render, FaceMesh, and the real pipeline on it.
    run_dir = work / "face"
    if not (run_dir / "meta.json").exists():
        print("Rendering the face camera and running FaceMesh ...", file=sys.stderr)
        generate(
            run_dir, scenario=segments, seed=seed, workers=workers, save_frames=True
        )
    run = Run.load(run_dir)
    res = replay(run)
    if res["zone"] is None:
        raise RuntimeError("the simulated commissioning failed; try another --seed")
    timeline = _timeline(run, res)

    # 2. The 3/4 view, for every simulated frame the cut uses.
    sim = Simulation(segments, seed=seed, render=False)
    wanted = sorted({s["sim"] for s in timeline})
    states = {}
    for _ in range(sim.frame_count):
        i, st = sim.next_state()
        if i in wanted:
            states[i] = st
    workers = workers or os.cpu_count() or 1
    cache = work / "view"
    cache.mkdir(parents=True, exist_ok=True)
    views = {}
    for i in wanted:
        f = cache / f"{i:06d}.json"
        if f.exists():
            proj = {k: np.asarray(v) for k, v in json.loads(f.read_text()).items()}
            views[i] = (cv2.imread(str(f.with_suffix(".jpg"))), proj)
    todo = [i for i in wanted if i not in views]
    print(f"Rendering {len(todo)} frames of the 3/4 view ...", file=sys.stderr)
    started = time.monotonic()
    if todo:
        with mp.get_context("spawn").Pool(workers, _init_worker, (seed,)) as pool:
            jobs = [(i, states[i]) for i in todo]
            for k, (i, img, proj) in enumerate(
                pool.imap_unordered(_render_view, jobs, chunksize=2)
            ):
                views[i] = (img, proj)
                if keep:
                    cv2.imwrite(
                        str(cache / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 96]
                    )
                    (cache / f"{i:06d}.json").write_text(
                        json.dumps({k: np.asarray(v).tolist() for k, v in proj.items()})
                    )
                if k % 30 == 0:
                    rate = (k + 1) / (time.monotonic() - started)
                    print(
                        f"\r  {k + 1}/{len(todo)}  {rate:.1f} fps",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
    print(file=sys.stderr)

    # 3. Composite and encode.
    fonts = _Fonts()
    box = _face_box(run)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    poster = None
    n_out = len(timeline)
    for k, info in enumerate(timeline):
        img, proj = views[info["sim"]]
        # A slow push-in over the whole piece.
        zoom = _lerp(1.0, 1.08, k / max(n_out - 1, 1))
        if zoom > 1.0:
            cw, ch = int(W / zoom), int(H / zoom)
            cx0, cy0 = (W - cw) // 2, int((H - ch) * 0.45)
            img = cv2.resize(
                img[cy0 : cy0 + ch, cx0 : cx0 + cw],
                (W, H),
                interpolation=cv2.INTER_CUBIC,
            )
            proj = {key: _zoomed(v, zoom, cx0, cy0) for key, v in proj.items()}
        info = dict(info, proj=proj)
        inset = _inset(run_dir, run, info["sim"], box) if info.get("inset") else None
        rgb = _overlay(img, info, fonts, inset)
        enc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        if poster is None and info.get("state") == "attentive" and info.get("caption"):
            poster = rgb
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")

    poster_path = out.with_suffix(".jpg")
    if poster is not None:
        cv2.imwrite(
            str(poster_path),
            np.ascontiguousarray(poster[..., ::-1]),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
    small = out.with_name(out.stem + "-sms.mp4")
    encode_small(out, small)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return {"video": out, "small": small, "poster": poster_path}


def _zoomed(p, zoom, cx0, cy0):
    return (np.asarray(p, float) - [cx0, cy0]) * zoom


def encode_small(
    src: pathlib.Path, dst: pathlib.Path, max_bytes: int = 950_000
) -> None:
    """A copy small enough to send as a text message.

    Carriers cap MMS attachments at around 1 MB (some lower), so this aims
    under that: 640x360, 24 fps, two-pass H.264 at a bitrate worked out from
    the duration. If the first try lands over, it tries again lower.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(src)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    duration = float(probe.stdout.strip())
    kbps = int(max_bytes * 8 / duration / 1000 * 0.92)
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(4):
            log = os.path.join(tmp, "pass")
            common = [
                "-vf", "scale=640:360:flags=lanczos,fps=24", "-c:v", "libx264",
                "-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p",
                "-profile:v", "main", "-an", "-passlogfile", log,
            ]  # fmt: skip
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    *common,
                    "-pass",
                    "1",
                    "-f",
                    "mp4",
                    os.devnull,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    *common,
                    "-pass",
                    "2",
                    "-movflags",
                    "+faststart",
                    str(dst),
                ],
                check=True,
            )
            if dst.stat().st_size <= max_bytes:
                return
            kbps = int(kbps * max_bytes / dst.stat().st_size * 0.95)
