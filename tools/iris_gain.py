#!/usr/bin/env python3
"""How much of an eye rotation does MediaPipe's iris register, on a real face?

The operator simulator found that on rendered faces MediaPipe's iris offset
moves with about half the geometrically expected gain horizontally and about
8% vertically. That is what collapses the head model's calibrated zone, and
whether it holds for real eyes decides whether the head model is worth
pursuing. This measures it.

Two steps, run from the repository root:

    # 1. With the monitor serving landmarks at full rate:
    python3 demo.py --source realsense --preview-port 8080 --preview-max-fps 0

    #    capture, guided, one directory per person:
    python3 tools/iris_gain.py capture --out runs/iris-gain/alice --motion

    # 2. analyse one or several people:
    python3 tools/iris_gain.py analyse runs/iris-gain/*

``capture`` is standard library only. ``analyse`` uses the monitor's own
gaze and head-pose code, so it measures exactly what the monitor sees.

Protocol
--------
A target sheet 60 cm from the subject's eyes (``--distance``), the centre
mark at eye level, the camera just below it and aimed at the face. Nine
marks in a cross: the centre, and 10 and 20 degrees up, down, left and right
of it -- 10.6 cm and 21.8 cm from the centre at 60 cm. Left and right are
the *subject's*. ``capture`` prints the offsets for the distance given.

Keep the head still: chin on fists with elbows on the table, or the back of
the head against a wall. Head movement is measured and subtracted, but the
less there is, the cleaner the number.

For each mark: look at it, press Enter, and keep looking until the second
beep. The first second of each hold is discarded while the eye settles.
``--repeats`` sets of nine (default 2). With ``--motion``, two more runs:
stare at the centre and nod slowly, then shake slowly. Those show whether
the eye term cancels the head term when the eyes counter-rotate.

Reading the result
------------------
The vertical gain, measured iris movement over what the geometry predicts,
decides it:

  0.5 or more, noise <= 3 deg   the rendered eyes were the problem; the head
                                model is viable
  0.2 to 0.5                    usable only with a per-person gain set at
                                commissioning
  under 0.2                     the simulation is confirmed; keep the eye
                                model, and look at another iris tracker
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import urllib.request

REFINED = 478

# (name, yaw towards the subject's left, pitch downwards), degrees.
MARKS = (
    ("centre", 0, 0),
    ("up10", 0, -10),
    ("up20", 0, -20),
    ("down10", 0, 10),
    ("down20", 0, 20),
    ("left10", 10, 0),
    ("left20", 20, 0),
    ("right10", -10, 0),
    ("right20", -20, 0),
    ("centre-end", 0, 0),  # drift check: the same mark, last
)
MOTIONS = (
    ("nod", "keep staring at the CENTRE mark and nod slowly, about 15 deg up and down"),
    (
        "shake",
        "keep staring at the CENTRE mark and turn your head slowly left and right",
    ),
)

# The verdict thresholds, from the protocol above.
VIABLE_GAIN, MARGINAL_GAIN, NOISE_LIMIT_DEG = 0.5, 0.2, 3.0


def describe(name: str, yaw: float, pitch: float, distance: float) -> str:
    if yaw == 0 and pitch == 0:
        return "the CENTRE mark"
    angle = abs(yaw or pitch)
    cm = 100.0 * distance * math.tan(math.radians(angle))
    where = (
        ("UP" if pitch < 0 else "DOWN") if pitch else ("LEFT" if yaw > 0 else "RIGHT")
    )
    return f"{where} {angle:g} ({cm:.1f} cm from the centre)"


# -- capture ---------------------------------------------------------------


def fetch(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=3.0) as response:
            return json.load(response)
    except Exception as exc:  # noqa: BLE001 - report and carry on
        print(f"  fetch failed: {exc}", file=sys.stderr)
        time.sleep(0.5)
        return None


def record(url: str, seconds: float, path: pathlib.Path) -> int:
    """Every new frame with a full landmark set, for this long."""
    seen = set()
    kept = 0
    deadline = time.monotonic() + seconds
    with open(path, "w", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            payload = fetch(url)
            if payload is None:
                continue
            seq = payload.get("frame")
            if seq in seen:
                time.sleep(0.005)
                continue
            seen.add(seq)
            if payload.get("count", 0) < REFINED:
                continue
            payload["t"] = time.monotonic() - float(payload.get("age_seconds", 0.0))
            handle.write(json.dumps(payload) + "\n")
            kept += 1
    return kept


def capture(args) -> int:
    url = args.url.rstrip("/") + "/landmarks.json"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe = fetch(url)
    if probe is None:
        print(
            f"No monitor at {url}. Start demo.py with --preview-port.", file=sys.stderr
        )
        return 1
    if probe.get("count", 0) < REFINED:
        print("Connected, but no face yet. Sit in front of the camera.")
    meta = {
        "subject": out.name,
        "distance_m": args.distance,
        "hold_s": args.hold,
        "discard_s": args.discard,
        "marks": MARKS,
        "size": probe.get("size"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nMarks for a sheet {100 * args.distance:.0f} cm from the eyes:")
    for name, yaw, pitch in MARKS[:-1]:
        print(f"  {name:8s} {describe(name, yaw, pitch, args.distance)}")
    print("\nHead still for these. Look at each mark, press Enter, and keep")
    print("looking until the second beep.\n")
    thin = 0
    for rep in range(1, args.repeats + 1):
        for name, yaw, pitch in MARKS:
            where = describe(name, yaw, pitch, args.distance)
            input(f"[set {rep}/{args.repeats}] Look at {where}, then press Enter ")
            print("\a  recording...", end="", flush=True)
            n = record(url, args.hold, out / f"r{rep}-{name}.jsonl")
            print(f"\a {n} frames")
            if n < 10 * args.hold:
                thin += 1
                print("  thin: is the face in view, and --preview-max-fps 0 set?")
    if args.motion:
        print("\nNow the head moves. Keep your eyes on the centre mark throughout.")
        for name, instruction in MOTIONS:
            input(f"{instruction}. Press Enter to start ({args.motion_seconds:g} s) ")
            print("\a  recording...", end="", flush=True)
            n = record(url, args.motion_seconds, out / f"{name}.jsonl")
            print(f"\a {n} frames")
    print(f"\nSaved to {out}. Analyse with: python3 tools/iris_gain.py analyse {out}")
    if thin:
        print(f"{thin} holds were thin; consider recapturing.", file=sys.stderr)
    return 0


# -- analysis --------------------------------------------------------------


def _numpy_and_monitor():
    repo = pathlib.Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import numpy as np

    from gaze_monitor import headpose
    from gaze_monitor.gaze import GazeMeasurer

    return np, headpose, GazeMeasurer


class _Landmarks:
    """Duck-types MediaPipe's landmark list over a (478, 3) array."""

    class _P:
        __slots__ = ("x", "y", "z")

        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    def __init__(self, array):
        self.array = array

    @property
    def landmark(self):
        return self

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        return self._P(*map(float, self.array[i]))


def measure_frames(frames) -> list[dict]:
    """Per frame: time, eye offset (dx, dy) and head angles, as the monitor
    sees them. No depth on this path, so head pose is FaceMesh's own."""
    np, _, GazeMeasurer = _numpy_and_monitor()
    measurer = GazeMeasurer(model="head")
    out = []
    for f in frames:
        lm = np.asarray(f["points"], dtype=float)
        w, h = f["size"]["width"], f["size"]["height"]
        s = measurer.measure(_Landmarks(lm), w, h)
        if s is None:
            continue
        yaw, pitch, roll = s.head_deg
        out.append(
            {
                "t": f.get("t", 0.0),
                "dx": s.eye_dx,
                "dy": s.eye_dy,
                "head_yaw": yaw,
                "head_pitch": pitch,
                "gaze_yaw": s.dx,
                "gaze_pitch": s.dy,
            }
        )
    return out


def summarise_hold(rows: list[dict], discard_s: float) -> dict | None:
    np, _, _ = _numpy_and_monitor()
    if not rows:
        return None
    t0 = rows[0]["t"]
    kept = [r for r in rows if r["t"] - t0 >= discard_s] or rows
    if len(kept) < 5:
        return None
    a = {k: np.array([r[k] for r in kept]) for k in kept[0]}
    return {
        "n": len(kept),
        **{k: float(np.median(a[k])) for k in ("dx", "dy", "head_yaw", "head_pitch")},
        "sd_dx": float(np.std(a["dx"])),
        "sd_dy": float(np.std(a["dy"])),
    }


def geometric_per_unit() -> tuple[float, float]:
    """Offset per unit sine of eye rotation, for the canonical eye: dx in half
    eye widths, dy in quarter eye widths (gaze_monitor.headpose)."""
    _, H, _ = _numpy_and_monitor()
    r, w = H.IRIS_PLANE_RADIUS, H.CANONICAL_EYE_WIDTH
    return r / (w / 2.0), r / (w / 4.0)


def fit_gains(holds: dict[tuple[int, str], dict], correct_head: bool = True) -> dict:
    """Measured offset against the geometric prediction, through the origin,
    per axis. Offsets and head angles are taken relative to each set's
    centre hold, so a fixed offset between eye and camera cancels."""
    np, _, _ = _numpy_and_monitor()
    gx, gy = geometric_per_unit()
    angles = {name: (yaw, pitch) for name, yaw, pitch in MARKS}
    pred, meas, sds = {"x": [], "y": []}, {"x": [], "y": []}, {"x": [], "y": []}
    head_range = []
    for rep in sorted({r for r, _ in holds}):
        ref = holds.get((rep, "centre"))
        if ref is None:
            continue
        for (r, name), h in holds.items():
            if r != rep or name not in angles:
                continue
            yaw, pitch = angles[name]
            hy = h["head_yaw"] - ref["head_yaw"]
            hp = h["head_pitch"] - ref["head_pitch"]
            head_range.append(math.hypot(hy, hp))
            if correct_head:
                yaw, pitch = yaw - hy, pitch - hp
            pred["x"].append(gx * math.sin(math.radians(yaw)))
            pred["y"].append(gy * math.sin(math.radians(pitch)))
            meas["x"].append(h["dx"] - ref["dx"])
            meas["y"].append(h["dy"] - ref["dy"])
            sds["x"].append(h["sd_dx"])
            sds["y"].append(h["sd_dy"])
    out = {
        "holds": len(pred["x"]),
        "head_movement_deg_max": max(head_range, default=0.0),
    }
    for axis, geo in (("x", gx), ("y", gy)):
        p, m = np.array(pred[axis]), np.array(meas[axis])
        if len(p) < 3 or not np.any(np.abs(p) > 1e-9):
            continue
        gain = float(p @ m / (p @ p))
        resid = m - gain * p
        per_deg = geo * math.radians(1.0) * gain
        noise = float(np.median(sds[axis]))
        out[axis] = {
            "gain": gain,
            "fit_residual": float(np.sqrt(np.mean(resid**2))),
            "noise": noise,
            # Within-hold scatter as eye rotation: in degrees of what the
            # tracker registers, which is what a zone threshold sees.
            "noise_deg": noise / abs(per_deg) if abs(gain) > 0.02 else math.inf,
        }
    return out


def verdict(gains: dict) -> str:
    y = gains.get("y")
    if y is None:
        return "no vertical fit (were the up and down marks captured?)"
    g, n = y["gain"], y["noise_deg"]
    if g < -MARGINAL_GAIN:
        return (
            "INVERTED: the iris moved the wrong way; were the marks looked at in order?"
        )
    if g >= VIABLE_GAIN and n <= NOISE_LIMIT_DEG:
        return (
            "VIABLE: real eyes do better than rendered ones; "
            "the head model is worth a commissioning test"
        )
    if g >= MARGINAL_GAIN:
        return "MARGINAL: usable only with a per-person gain set at commissioning"
    return "CONFIRMED: the iris barely registers vertical rotation; keep the eye model"


def analyse_motion(rows: list[dict]) -> dict | None:
    """With the gaze held on one point, eye-in-head should counter-rotate
    the head one for one, leaving the line of sight still."""
    np, _, _ = _numpy_and_monitor()
    if len(rows) < 30:
        return None
    gx, gy = geometric_per_unit()
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    eye_yaw = np.degrees(np.arcsin(np.clip(a["dx"] / gx, -1, 1)))
    eye_pitch = np.degrees(np.arcsin(np.clip(a["dy"] / gy, -1, 1)))
    out = {}
    for axis, head, eye, gaze in (
        ("yaw", a["head_yaw"], eye_yaw, a["gaze_yaw"]),
        ("pitch", a["head_pitch"], eye_pitch, a["gaze_pitch"]),
    ):
        hc = head - head.mean()
        span = float(np.percentile(head, 95) - np.percentile(head, 5))
        # Perfect VOR, perfectly measured: eye = -head, a gain of 1.
        vor = float(-(hc @ (eye - eye.mean())) / (hc @ hc)) if hc @ hc > 0 else math.nan
        out[axis] = {
            "head_span_deg": span,
            "vor_gain": vor,
            "gaze_sd_deg": float(np.std(gaze)),
            "head_sd_deg": float(np.std(head)),
        }
    return out


def load_session(path: pathlib.Path):
    if not (path / "meta.json").exists():
        raise SystemExit(f"{path} is not a capture: no meta.json in it")
    meta = json.loads((path / "meta.json").read_text())
    holds = {}
    for f in sorted(path.glob("r*-*.jsonl")):
        rep, name = f.stem[1:].split("-", 1)
        frames = [json.loads(line) for line in f.read_text().splitlines() if line]
        h = summarise_hold(measure_frames(frames), meta["discard_s"])
        if h is not None:
            holds[(int(rep), name)] = h
    motion = {}
    for name, _ in MOTIONS:
        f = path / f"{name}.jsonl"
        if f.exists():
            frames = [json.loads(line) for line in f.read_text().splitlines() if line]
            motion[name] = analyse_motion(measure_frames(frames))
    return meta, holds, motion


def report(path: pathlib.Path) -> dict:
    meta, holds, motion = load_session(path)
    gains = fit_gains(holds)
    raw = fit_gains(holds, correct_head=False)
    print(
        f"\n== {meta['subject']}  ({gains['holds']} holds, "
        f"sheet at {meta['distance_m']:g} m)"
    )
    print(
        f"head movement between holds, max {gains['head_movement_deg_max']:.1f} deg",
        end="",
    )
    print("  (subtracted below)" if gains["head_movement_deg_max"] > 0.5 else "")
    for axis, label in (("x", "horizontal"), ("y", "vertical")):
        g = gains.get(axis)
        if g is None:
            print(f"  {label:10s} no fit")
            continue
        r = raw.get(axis, {}).get("gain", math.nan)
        print(
            f"  {label:10s} gain {g['gain']:5.2f}  (uncorrected {r:5.2f})"
            f"  noise {_deg(g['noise_deg'])}  fit residual {g['fit_residual']:.3f}"
        )
    for name, m in motion.items():
        if m is None:
            print(f"  {name}: too few frames")
            continue
        axis = "pitch" if name == "nod" else "yaw"
        v = m[axis]
        print(
            f"  {name:5s} head span {v['head_span_deg']:4.1f} deg, VOR gain seen "
            f"{v['vor_gain']:4.2f} (1 is perfect), head-model gaze sd "
            f"{v['gaze_sd_deg']:.1f} deg against head sd {v['head_sd_deg']:.1f}"
        )
    print("  " + verdict(gains))
    return {
        "subject": meta["subject"],
        "gains": gains,
        "uncorrected": raw,
        "motion": motion,
    }


def _deg(x: float) -> str:
    return f"{x:5.1f} deg" if math.isfinite(x) else "  n/a (no gain)"


def analyse(args) -> int:
    results = [report(pathlib.Path(p)) for p in args.paths]
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="guided recording from a running monitor")
    cap.add_argument("--url", default="http://localhost:8080")
    cap.add_argument("--out", required=True, help="a directory, named for the subject")
    cap.add_argument(
        "--distance", type=float, default=0.60, help="eyes to sheet, metres"
    )
    cap.add_argument("--hold", type=float, default=4.0, help="seconds per mark")
    cap.add_argument("--discard", type=float, default=1.0, help="settling time ignored")
    cap.add_argument("--repeats", type=int, default=2)
    cap.add_argument("--motion", action="store_true", help="add the nod and shake runs")
    cap.add_argument("--motion-seconds", type=float, default=10.0)
    cap.set_defaults(func=capture)

    ana = sub.add_parser("analyse", help="gain and noise, per subject")
    ana.add_argument("paths", nargs="+")
    ana.add_argument("--json", help="also save the numbers here")
    ana.set_defaults(func=analyse)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
