"""Generate simulated runs, replay the real pipeline on them, and score it.

The work is split in two because the two halves cost very different amounts.

``generate`` renders the scenario and runs MediaPipe FaceMesh on every frame,
exactly as :func:`gaze_monitor.ui.run_monitor` does. That is the expensive
part, and nothing downstream of the landmarks changes it, so its output is
saved: the raw landmarks, the depth around the point the pipeline probes,
and the ground truth.

``evaluate`` replays everything downstream of FaceMesh -- ``GazeMeasurer``,
the disparity gate, the EMA, the four-corner ``GazeCalibrator`` and the
``AttentionMonitor`` -- using the package's own classes, and scores the
result against the truth. It takes milliseconds per minute of footage, which
is what makes sweeping the tuning parameters practical: one rendering, as
many configurations as you like, all judged on identical frames.

The score is against an oracle: the same ``AttentionMonitor`` with the same
thresholds, fed the operator's *true* line of sight. So a disagreement is the
measurement's fault, never the state machine's policy.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import pathlib
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gaze_monitor.attention import (  # noqa: E402
    AttentionMonitor,
    AttentionState,
    AttentionZone,
)
from gaze_monitor.calibration import GazeCalibrator  # noqa: E402
from gaze_monitor.gaze import (  # noqa: E402
    GazeFilter,
    GazeMeasurer,
    GazeSample,
    depth_at,
)
from gaze_monitor.headpose import sample_landmark_depth  # noqa: E402

FORMAT_VERSION = 2
# Format 1 runs lack per-landmark depth; they still load, and replay with
# head pose falls back to FaceMesh's relative depth for them.
READABLE_FORMATS = (1, 2)
DEPTH_WINDOW = 4  # half-size of the stored depth patch, pixels
LEFT_IRIS, RIGHT_IRIS = 468, 473
CORNERS = (33, 133, 362, 263)


# -- landmarks without MediaPipe ---------------------------------------------


class _Point:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _Landmarks:
    """Duck-types MediaPipe's NormalizedLandmarkList over a (478, 3) array."""

    def __init__(self, array: np.ndarray):
        self._a = array
        self.array = array  # the fast path gaze.landmark_array takes

    @property
    def landmark(self):
        return self

    def __len__(self) -> int:
        return len(self._a)

    def __getitem__(self, i):
        x, y, z = self._a[i]
        return _Point(x, y, z)


# -- generation ----------------------------------------------------------------

_WORKER = None


def _worker_init(config: dict) -> None:
    global _WORKER
    from .sim import Simulation

    _WORKER = Simulation(**config)


def _worker_render(job):
    index, state = job
    frame = _WORKER.render_state(index, state)
    return frame


def generate(
    out_dir: str | pathlib.Path,
    scenario: str = "shift",
    duration: float = 60.0,
    fps: float = 30.0,
    seed: int = 0,
    appearance: str = "default",
    calibration_head: str = "natural",
    camera: tuple | None = None,
    aim: tuple | None = None,
    traits=None,
    scene=None,
    workers: int = 0,
    ssaa: int = 2,
    video: bool = False,
    save_frames: bool = False,
    progress: bool = True,
) -> pathlib.Path:
    """Render a scenario, run FaceMesh on it, and save everything for replay."""
    import cv2
    import mediapipe as mp

    from .sim import Simulation

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(
        scenario=scenario,
        duration=duration,
        fps=fps,
        seed=seed,
        appearance=appearance,
        ssaa=ssaa,
        calibration_head=calibration_head,
        camera=camera,
        aim=aim,
    )
    if traits is not None:
        config["traits"] = traits
    if scene is not None:
        config["scene"] = scene
    sim = Simulation(**config)
    n = sim.frame_count
    w, h = sim.renderer.width, sim.renderer.height

    if workers <= 0:
        import os

        workers = max(1, (os.cpu_count() or 1))
    pool = None
    if workers > 1:
        import multiprocessing as mproc

        pool = mproc.get_context("spawn").Pool(workers, _worker_init, (config,))

    def jobs():
        for _ in range(n):
            yield sim.next_state()

    frames = (
        pool.imap(_worker_render, jobs(), chunksize=2)
        if pool is not None
        else (sim.render_state(i, s) for i, s in jobs())
    )

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_lm = np.full((n, 478, 3), np.nan, dtype=np.float32)
    truth_lm = np.zeros((n, 478, 2), dtype=np.float32)
    truth_z = np.zeros((n, 478), dtype=np.float32)
    # The depth image's reading at each of FaceMesh's landmarks, for head
    # pose in replay: the full depth image is far too big to keep.
    mp_depth = np.zeros((n, 478), dtype=np.float16)
    windows = np.zeros(
        (n, 2 * DEPTH_WINDOW + 1, 2 * DEPTH_WINDOW + 1), dtype=np.float32
    )
    origins = np.full((n, 2), -1, dtype=np.int32)
    writer = None
    if video:
        writer = cv2.VideoWriter(
            str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
    if save_frames:
        (out / "frames").mkdir(exist_ok=True)

    started = time.monotonic()
    with open(out / "truth.jsonl", "w") as truth_file:
        for frame in frames:
            i = frame.index
            r = frame.render
            rgb = cv2.cvtColor(r.color, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = face_mesh.process(rgb)
            truth = dict(frame.truth)
            truth_lm[i] = np.asarray(truth.pop("landmarks_px"), dtype=np.float32)
            truth_z[i] = r.landmarks_z
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0]
                arr = np.array(
                    [(p.x, p.y, p.z) for p in lms.landmark], dtype=np.float32
                )
                mp_lm[i] = arr
                mp_depth[i] = sample_landmark_depth(r.depth_m, arr, w, h)
                sample = GazeMeasurer().measure(lms, w, h, r.depth_m)
                if sample is not None:
                    origins[i] = (sample.px, sample.py)
                    windows[i] = _window(r.depth_m, sample.px, sample.py)
            truth_file.write(json.dumps(truth, default=_json_default) + "\n")
            if writer is not None:
                writer.write(annotate(r.color, truth, truth_lm[i], mp_lm[i], w, h))
            if save_frames:
                cv2.imwrite(
                    str(out / "frames" / f"{i:06d}.jpg"),
                    r.color,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                mm = np.clip(np.round(r.depth_m * 1000.0), 0, 65535).astype(np.uint16)
                cv2.imwrite(str(out / "frames" / f"{i:06d}_depth.png"), mm)
            if progress and (i % 30 == 0 or i == n - 1):
                rate = (i + 1) / max(time.monotonic() - started, 1e-6)
                print(
                    f"\r  frame {i + 1}/{n}  {rate:.1f} fps"
                    f"  eta {(n - i - 1) / rate:.0f}s   ",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    if progress:
        print(file=sys.stderr)
    face_mesh.close()
    if writer is not None:
        writer.release()
    if pool is not None:
        pool.close()
        pool.join()

    np.savez_compressed(
        out / "arrays.npz",
        mp_landmarks=mp_lm,
        truth_landmarks_px=truth_lm,
        truth_landmarks_z=truth_z,
        mp_landmark_depth=mp_depth,
        depth_windows=windows,
        depth_origins=origins,
    )
    meta = {
        "format": FORMAT_VERSION,
        "config": config,
        "frames": n,
        "width": w,
        "height": h,
        "intrinsics": sim.intrinsics,
        "zone_corners": sim.scene.zone.corners.tolist(),
        "camera": {
            "position": list(sim.scene.mount.position),
            "target": list(sim.scene.mount.target),
        },
        "segments": [_segment_dict(s) for s in sim.segments],
        "generated_seconds": round(time.monotonic() - started, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    return out


def simulate_truth(
    scenario="shift",
    duration: float = 60.0,
    fps: float = 30.0,
    seed: int = 0,
    appearance: str = "default",
    calibration_head: str = "natural",
    **sim_kwargs,
) -> Run:
    """A run with ground truth only: no rendering and no FaceMesh.

    Replay it with ``landmarks="truth"`` to see what the pipeline does with a
    perfect landmark detector. That isolates the gaze model -- calibration,
    eye-in-head as a stand-in for gaze, the filter, the state machine -- from
    the detector, in seconds rather than minutes, and needs only numpy.
    """
    from .sim import Simulation

    sim = Simulation(
        scenario,
        duration=duration,
        fps=fps,
        seed=seed,
        appearance=appearance,
        calibration_head=calibration_head,
        render=False,
        **sim_kwargs,
    )
    n = sim.frame_count
    truth, lm = [], np.zeros((n, 478, 2), dtype=np.float32)
    lz = np.zeros((n, 478), dtype=np.float32)
    for _ in range(n):
        frame = sim.truth_state(*sim.next_state())
        tr = dict(frame.truth)
        lm[frame.index] = np.asarray(tr.pop("landmarks_px"), dtype=np.float32)
        lz[frame.index] = frame.render.landmarks_z
        truth.append(json.loads(json.dumps(tr, default=_json_default)))
    r = 2 * DEPTH_WINDOW + 1
    meta = {
        "format": FORMAT_VERSION,
        "config": {
            "scenario": scenario if isinstance(scenario, str) else "custom",
            "fps": fps, "duration": duration, "seed": seed, "appearance": appearance,
        },
        "frames": n,
        "width": sim.renderer.width,
        "height": sim.renderer.height,
        "intrinsics": sim.intrinsics,
        "zone_corners": sim.scene.zone.corners.tolist(),
        "truth_only": True,
    }  # fmt: skip
    return Run(
        meta=meta,
        truth=truth,
        mp_landmarks=np.full((n, 478, 3), np.nan, dtype=np.float32),
        truth_landmarks_px=lm,
        truth_landmarks_z=lz,
        depth_windows=np.zeros((n, r, r), dtype=np.float32),
        depth_origins=np.full((n, 2), -1, dtype=np.int32),
    )


def _segment_dict(seg) -> dict:
    d = dataclasses.asdict(seg)
    return {k: v for k, v in d.items() if v is not None}


def _json_default(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def _window(depth: np.ndarray, px: int, py: int) -> np.ndarray:
    r = DEPTH_WINDOW
    out = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
    h, w = depth.shape
    ys, xs = slice(max(0, py - r), min(h, py + r + 1)), slice(
        max(0, px - r), min(w, px + r + 1)
    )
    out[
        ys.start - (py - r) : ys.stop - (py - r),
        xs.start - (px - r) : xs.stop - (px - r),
    ] = depth[ys, xs]
    return out


def annotate(color, truth, truth_px, mp_norm, w, h) -> np.ndarray:
    """A frame with the truth (green) and FaceMesh (red) irises drawn on."""
    import cv2

    img = color.copy()
    for idx in (LEFT_IRIS, RIGHT_IRIS):
        x, y = truth_px[idx]
        cv2.circle(img, (int(round(x)), int(round(y))), 3, (0, 220, 0), 1, cv2.LINE_AA)
        if np.isfinite(mp_norm[idx, 0]):
            mx, my = mp_norm[idx, 0] * w, mp_norm[idx, 1] * h
            cv2.circle(
                img, (int(round(mx)), int(round(my))), 2, (0, 0, 255), -1, cv2.LINE_AA
            )
    u, v = truth["zone_u"], truth["zone_v"]
    inside = math.isfinite(u) and -0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
    lines = [
        f"t {truth['t']:6.2f}s  {truth['activity']}  -> {truth['target']}",
        f"zone u,v {u:+.2f},{v:+.2f} {'IN' if inside else 'OUT'}"
        f"  dist {truth['distance_m']:.2f}m",
        "eye-in-head yaw/pitch "
        + "  ".join(
            f"{a:+5.1f}/{b:+5.1f}"
            for a, b in zip(truth["eye_yaw_deg"], truth["eye_pitch_deg"], strict=True)
        ),
        f"head yaw {truth['head_yaw_deg']:+5.1f} pitch {truth['head_pitch_deg']:+5.1f}"
        f" roll {truth['head_roll_deg']:+5.1f}  blink {truth['blink']:.2f}",
    ]
    for k, line in enumerate(lines):
        y = 24 + 22 * k
        cv2.putText(
            img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            img,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


# -- replay --------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """The pipeline's tuning knobs, with the shipped defaults."""

    filter_alpha: float = 0.12
    max_eye_disparity: float = 0.0
    zone_margin: float = 0.1
    warn_after: float = 1.0
    alert_after: float = 3.0
    clear_after: float = 0.4
    sample_duration: float = 1.0
    sigma_threshold: float = 0.05
    # The same, for the head model, whose units are degrees.
    sigma_threshold_deg: float = 3.0
    # "head": head pose plus eye-in-head, the line of sight (the default).
    # "eye": the iris's offset within the eye alone, as before head pose.
    gaze_model: str = "head"
    # When, into each calibration fixation, the simulated operator presses c.
    calib_press_after: float = 1.0


@dataclass
class Run:
    meta: dict
    truth: list
    mp_landmarks: np.ndarray
    truth_landmarks_px: np.ndarray
    depth_windows: np.ndarray
    depth_origins: np.ndarray
    truth_landmarks_z: np.ndarray | None = None
    mp_landmark_depth: np.ndarray | None = None

    @property
    def intrinsics(self):
        from gaze_monitor.headpose import Intrinsics

        k = self.meta.get("intrinsics")
        return Intrinsics(k["fx"], k["fy"], k["cx"], k["cy"]) if k else None

    def landmark_depth(self, i: int, source: str) -> np.ndarray | None:
        """Depth at each landmark for frame i: measured, or the truth."""
        if source == "truth":
            z = self.truth_landmarks_z
        else:
            z = self.mp_landmark_depth
        return None if z is None else np.asarray(z[i], dtype=float)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Run:
        p = pathlib.Path(path)
        meta = json.loads((p / "meta.json").read_text())
        if meta.get("format") not in READABLE_FORMATS:
            raise ValueError(
                f"{p} is format {meta.get('format')}, expected one of"
                f" {READABLE_FORMATS}"
            )
        truth = [
            json.loads(line) for line in (p / "truth.jsonl").read_text().splitlines()
        ]
        a = np.load(p / "arrays.npz")
        return cls(
            meta=meta,
            truth=truth,
            mp_landmarks=a["mp_landmarks"],
            truth_landmarks_px=a["truth_landmarks_px"],
            depth_windows=a["depth_windows"],
            depth_origins=a["depth_origins"],
            truth_landmarks_z=(
                a["truth_landmarks_z"] if "truth_landmarks_z" in a else None
            ),
            mp_landmark_depth=(
                a["mp_landmark_depth"] if "mp_landmark_depth" in a else None
            ),
        )

    def depth_image(self, i: int) -> np.ndarray | None:
        px, py = self.depth_origins[i]
        if px < 0:
            return None
        w, h = self.meta["width"], self.meta["height"]
        img = np.zeros((h, w), dtype=np.float32)
        r = DEPTH_WINDOW
        y0, x0 = py - r, px - r
        ys = slice(max(0, y0), min(h, py + r + 1))
        xs = slice(max(0, x0), min(w, px + r + 1))
        img[ys, xs] = self.depth_windows[i][
            ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0
        ]
        return img


@dataclass
class _Oracle:
    dx: float
    dy: float
    z_m: float = 0.0


def landmarks_for(run: Run, source: str = "mediapipe") -> np.ndarray:
    """Per-frame normalised landmarks: FaceMesh's, or the rendered truth.

    ``truth`` is a perfect landmark detector. Replaying it shows what the
    rest of the pipeline would do with flawless landmarks, which separates
    landmark error from the error in the gaze model itself.
    """
    if source == "mediapipe":
        return run.mp_landmarks
    if source != "truth":
        raise ValueError(f"Unknown landmark source {source!r}")
    w, h = run.meta["width"], run.meta["height"]
    px = run.truth_landmarks_px / np.array([w, h], dtype=np.float32)
    # FaceMesh's z is depth relative to the face, in units of the frame
    # width. From true depth, that is (z - z_face) * fx / z_face / width.
    z = np.zeros(px.shape[:2] + (1,), np.float32)
    if run.truth_landmarks_z is not None and run.meta.get("intrinsics"):
        tz = run.truth_landmarks_z.astype(float)
        ref = np.median(tz[:, :468], axis=1, keepdims=True)
        fx = run.meta["intrinsics"]["fx"]
        z[..., 0] = ((tz - ref) * fx / np.maximum(ref, 1e-6) / w).astype(np.float32)
    return np.concatenate([px, z], axis=2)


def replay(
    run: Run, params: Params | None = None, landmarks: str = "mediapipe"
) -> dict:
    """Run the pipeline on a generated run and return per-frame results."""
    params = params or Params()
    w, h = run.meta["width"], run.meta["height"]
    n = len(run.truth)
    all_lm = landmarks_for(run, landmarks)
    head = params.gaze_model == "head"
    measurer = GazeMeasurer(model=params.gaze_model)
    intrinsics = run.intrinsics
    fx = GazeFilter(alpha=params.filter_alpha)
    fy = GazeFilter(alpha=params.filter_alpha)
    calibrator = GazeCalibrator(
        sample_duration=params.sample_duration,
        sigma_threshold=params.sigma_threshold_deg if head else params.sigma_threshold,
        units="deg" if head else "eye",
    )
    monitor = None
    oracle = AttentionMonitor(
        zone=AttentionZone(0.0, 1.0, 0.0, 1.0, margin=params.zone_margin),
        warn_after=params.warn_after,
        alert_after=params.alert_after,
        clear_after=params.clear_after,
    )
    calib_log = []
    seg_start = None
    last_corner = None
    out = {
        "t": np.zeros(n),
        "monitoring": np.zeros(n, bool),
        "raw_uv": np.full((n, 2), np.nan),
        "uv": np.full((n, 2), np.nan),
        "true_uv": np.full((n, 2), np.nan),
        "z_m": np.zeros(n),
        "disparity": np.full(n, np.nan),
        "face": np.zeros(n, bool),
        "sample": np.zeros(n, bool),
        "state": np.zeros(n, dtype=object),
        "oracle": np.zeros(n, dtype=object),
        "in_zone": np.zeros(n, bool),
        "true_in_zone": np.zeros(n, bool),
        "away": np.zeros(n),
        "corners_done": np.zeros(n, int),
    }
    zone = None
    for i, tr in enumerate(run.truth):
        t = tr["t"]
        out["t"][i] = t
        lm = all_lm[i]
        raw = sample = None
        if np.isfinite(lm[0, 0]):
            out["face"][i] = True
            ld = run.landmark_depth(i, landmarks)
            raw = measurer.measure(
                _Landmarks(lm),
                w,
                h,
                None if ld is not None else run.depth_image(i),
                intrinsics=intrinsics,
                landmark_depth=ld,
            )
            if raw is not None:
                out["disparity"][i] = raw.disparity
                if not (
                    params.max_eye_disparity > 0
                    and raw.disparity > params.max_eye_disparity
                ):
                    sample = _smooth(raw, fx, fy)
        out["sample"][i] = sample is not None
        if sample is not None:
            out["z_m"][i] = sample.z_m

        out["corners_done"][i] = calibrator.corner_index
        corner = tr.get("calib_corner")
        if corner is not None and corner != last_corner:
            seg_start = t
        last_corner = corner

        if zone is None:
            # Commissioning, driven the way a person drives it: settle on the
            # corner, press c, and press again if the fixation was rejected.
            if (
                corner is not None
                and corner == calibrator.corner_index
                and not calibrator.is_collecting
                and t - seg_start >= params.calib_press_after
            ):
                calibrator.start_collection(t)
            if calibrator.is_collecting:
                if sample is not None:
                    calibrator.collect(t, sample.dx, sample.dy, sample.z_m)
                if calibrator.should_finish(t):
                    ok, msg = calibrator.validate_and_save()
                    calib_log.append(
                        {"t": t, "corner": corner, "ok": ok, "message": msg}
                    )
                    if calibrator.is_finished:
                        zone = calibrator.build_zone(margin=params.zone_margin)
                        monitor = AttentionMonitor(
                            zone=zone,
                            warn_after=params.warn_after,
                            alert_after=params.alert_after,
                            clear_after=params.clear_after,
                        )
            continue
        if tr["activity"] == "calibrate":
            continue  # the closing look back at the cut

        out["monitoring"][i] = True
        status = monitor.update(t, sample, camera_ok=True)
        u, v = tr["zone_u"], tr["zone_v"]
        true = _Oracle(u, v) if (math.isfinite(u) and math.isfinite(v)) else None
        ostatus = oracle.update(t, true, camera_ok=True)
        out["state"][i] = status.state.value
        out["away"][i] = status.away_seconds
        out["oracle"][i] = ostatus.state.value
        out["in_zone"][i] = status.in_zone
        out["true_in_zone"][i] = ostatus.in_zone
        out["true_uv"][i] = (u, v)
        if sample is not None:
            out["uv"][i] = zone.project(sample.dx, sample.dy, sample.z_m)
        if raw is not None:
            out["raw_uv"][i] = zone.project(raw.dx, raw.dy, raw.z_m)

    out["calibration"] = calib_log
    out["zone"] = zone.to_dict() if zone is not None else None
    return out


def _smooth(sample: GazeSample, fx: GazeFilter, fy: GazeFilter):
    dx = fx.apply(sample.dx)
    dy = fy.apply(sample.dy)
    if dx is None or dy is None:
        return None
    return dataclasses.replace(sample, dx=dx, dy=dy)


# -- scoring -------------------------------------------------------------------

_SEVERITY = {
    AttentionState.ATTENTIVE.value: 0,
    AttentionState.WARNING.value: 1,
    AttentionState.ALERT.value: 2,
    AttentionState.FAULT.value: 2,
}


def score(
    run: Run,
    params: Params | None = None,
    result: dict | None = None,
    landmarks: str = "mediapipe",
    ideal: bool = True,
) -> dict:
    """Headline metrics for one configuration on one run.

    With ``ideal`` the run is also replayed on perfect landmarks, and the
    report carries that under ``ideal_landmarks``: the error left over there
    is the gaze model's, and the difference is FaceMesh's.
    """
    params = params or Params()
    r = result if result is not None else replay(run, params, landmarks)
    fps = run.meta["config"]["fps"]
    truth = run.truth
    m = r["monitoring"]
    report: dict = {
        "params": dataclasses.asdict(params),
        "frames": len(truth),
        "monitored_seconds": round(float(m.sum()) / fps, 1),
        "calibration_ok": r["zone"] is not None,
        "calibration_attempts": len(r["calibration"]),
        "calibration": r["calibration"],
    }
    report["landmarks"] = landmark_errors(run)
    report["eye_in_head"] = eye_in_head_fit(run, landmarks)
    if ideal and landmarks != "truth":
        sub = score(run, params, landmarks="truth", ideal=False)
        report["ideal_landmarks"] = {
            k: sub.get(k)
            for k in (
                "calibration_ok",
                "eye_in_head",
                "gaze_error",
                "in_zone",
                "state_agreement",
                "events",
            )
        }
    if not report["calibration_ok"] or not m.any():
        return report

    face = np.array(
        [tr["face_to_camera_deg"] < 50 and tr["blink"] < 0.5 for tr in truth]
    )
    report["face_detected"] = _frac(r["face"][m])
    report["sample_rate"] = _frac(r["sample"][m])
    report["sample_rate_facing_eyes_open"] = _frac(r["sample"][m & face])

    zw = np.array([tr["zone_size_deg"][0] for tr in truth])
    zh = np.array([tr["zone_size_deg"][1] for tr in truth])
    for key, label in (("uv", "gaze_error"), ("raw_uv", "gaze_error_unsmoothed")):
        ok = m & np.isfinite(r[key][:, 0]) & np.isfinite(r["true_uv"][:, 0])
        if ok.any():
            du = (r[key][ok, 0] - r["true_uv"][ok, 0]) * zw[ok]
            dv = (r[key][ok, 1] - r["true_uv"][ok, 1]) * zh[ok]
            e = np.hypot(du, dv)
            report[label] = {
                "rms_deg": _r(np.sqrt(np.mean(e**2))),
                "median_deg": _r(np.median(e)),
                "p95_deg": _r(np.percentile(e, 95)),
                "bias_deg": [_r(np.mean(du)), _r(np.mean(dv))],
                "rms_axes_deg": [
                    _r(np.sqrt(np.mean(du**2))),
                    _r(np.sqrt(np.mean(dv**2))),
                ],
            }
    report["zone_size_deg"] = [_r(np.median(zw[m])), _r(np.median(zh[m]))]

    tz, pz = r["true_in_zone"][m], r["in_zone"][m]
    report["in_zone"] = {
        "accuracy": _frac(tz == pz),
        "false_out": _frac(~pz[tz]) if tz.any() else None,
        "false_in": _frac(pz[~tz]) if (~tz).any() else None,
    }
    states = r["state"][m]
    oracle = r["oracle"][m]
    report["state_agreement"] = _frac(states == oracle)
    labels = [s.value for s in AttentionState if s is not AttentionState.FAULT]
    report["state_confusion"] = {
        o: {p: int(np.sum((oracle == o) & (states == p))) for p in labels}
        for o in labels
    }
    report["events"] = events(r["t"][m], oracle, states, fps)
    z = r["z_m"][m]
    dist = np.array([tr["distance_m"] for tr in truth])[m]
    zok = z > 0
    report["depth"] = {
        "valid": _frac(zok),
        "median_error_m": _r(np.median(z[zok] - dist[zok]), 4) if zok.any() else None,
    }
    report["by_activity"] = by_activity(truth, r, m, zw, zh, fps)
    return report


def events(t: np.ndarray, oracle: np.ndarray, pred: np.ndarray, fps: float) -> dict:
    """Alarm episodes: missed, late and false, at WARNING and ALERT."""
    out = {}
    minutes = len(t) / fps / 60.0
    for level, name in ((1, "warning"), (2, "alert")):
        o = np.array([_SEVERITY[s] >= level for s in oracle])
        p = np.array([_SEVERITY[s] >= level for s in pred])
        o_eps = _episodes(o)
        p_eps = _episodes(p)
        tol = int(round(1.0 * fps))
        detected, latencies, clears = 0, [], []
        for a, b in o_eps:
            hits = [(pa, pb) for pa, pb in p_eps if pb >= a - tol and pa <= b + tol]
            if hits:
                detected += 1
                latencies.append((hits[0][0] - a) / fps)
                # How long the alarm outlasted the real excursion: the time
                # to clear once the operator is looking back.
                clears.append((max(pb for _, pb in hits) - b) / fps)
        false = [
            (a, b)
            for a, b in p_eps
            if not any(b >= oa - tol and a <= ob + tol for oa, ob in o_eps)
        ]
        out[name] = {
            "true_episodes": len(o_eps),
            "detected": detected,
            "missed": len(o_eps) - detected,
            "latency_median_s": _r(np.median(latencies)) if latencies else None,
            "latency_max_s": _r(np.max(latencies)) if latencies else None,
            "false_episodes": len(false),
            "false_per_min": _r(len(false) / minutes) if minutes > 0 else None,
            "false_seconds": _r(sum(b - a + 1 for a, b in false) / fps),
            "clear_lag_median_s": _r(np.median(clears)) if clears else None,
            # Raw per-episode values, so a benchmark can pool across runs.
            "latencies_s": [_r(x) for x in latencies],
            "clear_lags_s": [_r(x) for x in clears],
            "minutes": _r(minutes, 4),
        }
    return out


def _episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    eps = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            eps.append((start, i - 1))
            start = None
    if start is not None:
        eps.append((start, len(mask) - 1))
    return eps


def by_activity(truth, r, m, zw, zh, fps: float) -> dict:
    acts = {}
    for i in np.nonzero(m)[0]:
        acts.setdefault(truth[i]["activity"], []).append(i)
    out = {}
    for name, idx in sorted(acts.items()):
        idx = np.array(idx)
        ok = idx[np.isfinite(r["uv"][idx, 0]) & np.isfinite(r["true_uv"][idx, 0])]
        e = (
            np.hypot(
                (r["uv"][ok, 0] - r["true_uv"][ok, 0]) * zw[ok],
                (r["uv"][ok, 1] - r["true_uv"][ok, 1]) * zh[ok],
            )
            if len(ok)
            else np.array([])
        )
        out[name] = {
            "seconds": _r(len(idx) / fps, 1),
            "sample_rate": _frac(r["sample"][idx]),
            "true_in_zone": _frac(r["true_in_zone"][idx]),
            "in_zone_accuracy": _frac(r["true_in_zone"][idx] == r["in_zone"][idx]),
            "median_error_deg": _r(np.median(e)) if len(e) else None,
        }
    return out


def eye_in_head_fit(run: Run, landmarks: str = "mediapipe") -> dict | None:
    """How well the measured (dx, dy) track the eyes' true rotation in the head.

    An affine fit of the true eye-in-head yaw and pitch on the raw (dx, dy),
    over every frame with both eyes open and showing. The residual is the
    measurement's noise floor in degrees, independent of calibration, of the
    filter, and of whether eye-in-head is even the right quantity to measure.
    This is the number to watch when changing the landmark-to-gaze geometry.
    """
    w, h = run.meta["width"], run.meta["height"]
    lms = landmarks_for(run, landmarks)
    measurer = GazeMeasurer()
    X, Y = [], []
    for i, tr in enumerate(run.truth):
        lm = lms[i]
        if not np.isfinite(lm[0, 0]):
            continue
        s = measurer.measure(_Landmarks(lm), w, h)
        if s is None or tr["blink"] > 0.2 or not all(tr["iris_visible"]):
            continue
        X.append((s.dx, s.dy))
        Y.append(
            (float(np.mean(tr["eye_yaw_deg"])), float(np.mean(tr["eye_pitch_deg"])))
        )
    if len(X) < 10:
        return None
    X = np.asarray(X)
    Y = np.asarray(Y)
    M = np.c_[X, np.ones(len(X))]
    out = {"frames": len(X)}
    for k, name in ((0, "yaw"), (1, "pitch")):
        coef, *_ = np.linalg.lstsq(M, Y[:, k], rcond=None)
        res = Y[:, k] - M @ coef
        out[f"{name}_resid_rms_deg"] = _r(np.sqrt(np.mean(res**2)))
        out[f"{name}_range_deg"] = [_r(Y[:, k].min()), _r(Y[:, k].max())]
        out[f"{name}_corr"] = _r(np.corrcoef(X[:, k], Y[:, k])[0, 1])
    return out


def landmark_errors(run: Run) -> dict:
    """FaceMesh's landmark error against the rendered truth, in pixels."""
    w, h = run.meta["width"], run.meta["height"]
    mp = run.mp_landmarks
    ok = np.isfinite(mp[:, 0, 0])
    vis = np.array([all(tr["iris_visible"]) for tr in run.truth])
    out = {"frames_with_face": _frac(ok)}
    if not ok.any():
        return out
    px = mp[..., :2] * np.array([w, h], dtype=np.float32)
    err = np.linalg.norm(px - run.truth_landmarks_px, axis=2)
    sel = ok & vis
    if sel.any():
        iris = err[sel][:, [LEFT_IRIS, RIGHT_IRIS]]
        out["iris_centre_median_px"] = _r(np.median(iris))
        out["iris_centre_p95_px"] = _r(np.percentile(iris, 95))
    corners = err[ok][:, list(CORNERS)]
    out["eye_corner_median_px"] = _r(np.median(corners))
    out["all_468_median_px"] = _r(np.median(err[ok][:, :468]))
    return out


def _frac(a) -> float | None:
    a = np.asarray(a)
    return _r(float(a.mean()), 4) if a.size else None


def _r(x, nd: int = 3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# -- sweeps --------------------------------------------------------------------


def parse_grid(items: Iterable[str]) -> dict[str, list]:
    """``name=v1,v2`` items into a grid. Values take their field's type, so
    ``gaze_model=eye,head`` works as well as ``filter_alpha=0.1,0.2``."""
    grid = {}
    types = {f.name: type(f.default) for f in dataclasses.fields(Params)}
    fields = set(types)
    for item in items:
        name, _, values = item.partition("=")
        name = name.strip().replace("-", "_")
        if name not in fields:
            raise ValueError(
                f"Unknown parameter {name!r}. Known: {', '.join(sorted(fields))}"
            )
        cast = str if types[name] is str else float
        grid[name] = [cast(v.strip()) for v in values.split(",") if v.strip()]
    return grid


def sweep(
    runs: list[Run], grid: dict[str, list[float]], base: Params | None = None
) -> list[dict]:
    """Score every combination in ``grid`` on every run; aggregate per combination."""
    base = base or Params()
    names = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in names)):
        params = dataclasses.replace(base, **dict(zip(names, combo, strict=True)))
        try:
            reports = [score(run, params) for run in runs]
        except ValueError as exc:
            rows.append({**dict(zip(names, combo, strict=True)), "error": str(exc)})
            continue
        rows.append({**dict(zip(names, combo, strict=True)), **summarise(reports)})
    return rows


def summarise(reports: list[dict]) -> dict:
    """The handful of numbers that decide a tuning question, summed over runs."""

    def total(path):
        vals = []
        for rep in reports:
            v = rep
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if v is not None:
                vals.append(v)
        return vals

    def mean(path):
        v = total(path)
        return _r(float(np.mean(v))) if v else None

    return {
        "calibrated": sum(1 for r in reports if r.get("calibration_ok")),
        "runs": len(reports),
        "eih_yaw_deg": mean(("eye_in_head", "yaw_resid_rms_deg")),
        "eih_pitch_deg": mean(("eye_in_head", "pitch_resid_rms_deg")),
        "gaze_rms_deg": mean(("gaze_error", "rms_deg")),
        "gaze_median_deg": mean(("gaze_error", "median_deg")),
        "in_zone_acc": mean(("in_zone", "accuracy")),
        "state_agree": mean(("state_agreement",)),
        "warn_missed": sum(total(("events", "warning", "missed"))),
        "warn_true": sum(total(("events", "warning", "true_episodes"))),
        "warn_latency_s": mean(("events", "warning", "latency_median_s")),
        "false_warn": sum(total(("events", "warning", "false_episodes"))),
        "false_warn_per_min": mean(("events", "warning", "false_per_min")),
        "alert_missed": sum(total(("events", "alert", "missed"))),
        "false_alert": sum(total(("events", "alert", "false_episodes"))),
    }


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    for r in rows[1:]:
        cols += [k for k in r if k not in cols]
    cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    lines = ["  ".join(c.rjust(w) for c, w in zip(cols, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    for row in cells:
        lines.append("  ".join(v.rjust(w) for v, w in zip(row, widths, strict=True)))
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


__all__ = [
    "Params",
    "Run",
    "generate",
    "replay",
    "score",
    "sweep",
    "parse_grid",
    "format_table",
    "depth_at",
]
