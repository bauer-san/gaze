#!/usr/bin/env python3
"""Measure the gaze pipeline's noise floor, and compare the two algorithms.

Both the old and the new eye-displacement algorithms are pure functions of
the raw landmarks, and the preview's /landmarks endpoint serves all 478 of
them. So the honest comparison is not before-and-after across two sessions,
where any difference in how the operator happened to sit contaminates the
result: it is one capture, with both algorithms applied to identical frames.
Both are therefore implemented here rather than imported, so this stays a
stable reference whichever version of gaze_monitor is deployed.

Standard library only, so it runs with a bare python3 anywhere.

    # 1. capture, with somebody sitting in front of the camera
    python3 tools/gaze_noise.py capture --url http://localhost:8080 \
        --seconds 30 --out fixation.jsonl

    # 2. analyse
    python3 tools/gaze_noise.py analyse fixation.jsonl

Protocol matters more than anything in here. The noise estimate uses frame
to frame differences, which only separates noise from signal when the true
signal is near constant between samples. So:

  fixation  Sit still, head level, and stare at one fixed point. This is the
            run that measures the noise floor.
  roll      Keep staring at that same point and tilt your head slowly left
            and right, as far as is comfortable. This is the run that shows
            the roll artifact, which was the larger of the two bugs.

Run both, analyse both, and keep the files. They are the baseline that makes
the next change to this pipeline measurable rather than arguable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request

# Landmark indices, copied rather than imported for the reason in the
# module docstring.
LEFT_IRIS, LEFT_RING = 468, (469, 470, 471, 472)
RIGHT_IRIS, RIGHT_RING = 473, (474, 475, 476, 477)
LEFT_OUTER, LEFT_INNER = 33, 133
RIGHT_INNER, RIGHT_OUTER = 362, 263
REFINED = 478

EYES = (
    ("left", LEFT_IRIS, LEFT_RING, LEFT_OUTER, LEFT_INNER),
    ("right", RIGHT_IRIS, RIGHT_RING, RIGHT_INNER, RIGHT_OUTER),
)

# Physical constants, for putting the numbers in units somebody can argue
# with. The iris is about 11.7 mm across in adults and is one of the most
# stable dimensions on the body.
IRIS_MM = 11.7


# -- the two algorithms ----------------------------------------------------


def old_eye(points, iris, _ring, a_idx, b_idx, w, h):
    """Centre landmark only, image axes, normalised coordinates.

    Two faults, both systematic. The offset is projected onto the image axes
    rather than the eye's, so head roll leaks into dy as tan(roll). And dy
    divides a normalised-y offset by a scale taken from normalised x, so it
    carries a factor of the frame aspect ratio.
    """
    ix, iy = points[iris][0], points[iris][1]
    ax, ay = points[a_idx][0], points[a_idx][1]
    bx, by = points[b_idx][0], points[b_idx][1]
    width = abs(bx - ax)
    if width < 1e-3:
        return None
    cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
    return (ix - cx) / (width / 2.0), (iy - cy) / (width / 4.0)


def new_eye(points, iris, ring, a_idx, b_idx, w, h, width_px=None):
    """Five-point iris centre, pixel space, the eye's own axes."""
    group = [points[iris]] + [points[i] for i in ring]
    ix = sum(p[0] for p in group) / len(group) * w
    iy = sum(p[1] for p in group) / len(group) * h
    ax, ay = points[a_idx][0] * w, points[a_idx][1] * h
    bx, by = points[b_idx][0] * w, points[b_idx][1] * h
    vx, vy = bx - ax, by - ay
    measured = math.hypot(vx, vy)
    if measured < 1.0:
        return None
    width = measured if width_px is None else width_px
    ux, uy = vx / measured, vy / measured
    ox, oy = ix - (ax + bx) / 2.0, iy - (ay + by) / 2.0
    along = ox * ux + oy * uy
    across = -ox * uy + oy * ux
    return along / (width / 2.0), across / (width / 4.0)


def iris_centre_px(points, iris, ring, w, h, five):
    group = [points[iris]] + ([points[i] for i in ring] if five else [])
    return (
        sum(p[0] for p in group) / len(group) * w,
        sum(p[1] for p in group) / len(group) * h,
    )


def eye_width_px(points, a_idx, b_idx, w, h):
    ax, ay = points[a_idx][0] * w, points[a_idx][1] * h
    bx, by = points[b_idx][0] * w, points[b_idx][1] * h
    return math.hypot(bx - ax, by - ay)


def iris_radius_px(points, iris, ring, w, h):
    cx, cy = points[iris][0] * w, points[iris][1] * h
    return statistics.fmean(
        math.hypot(points[i][0] * w - cx, points[i][1] * h - cy) for i in ring
    )


def head_roll_deg(points, w, h):
    """Roll from the line joining the two eyes' outer corners."""
    ax, ay = points[LEFT_OUTER][0] * w, points[LEFT_OUTER][1] * h
    bx, by = points[RIGHT_OUTER][0] * w, points[RIGHT_OUTER][1] * h
    return math.degrees(math.atan2(by - ay, bx - ax))


def rotate(points, degrees, w, h):
    """Rotate every landmark about the frame centre, in pixel space.

    A rigid rotation changes no eye-in-head geometry, so whatever it does to
    the reported gaze is pure artifact. Done in pixels because normalised
    coordinates are anisotropic and rotating in them is a shear.
    """
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    cx, cy = w / 2.0, h / 2.0
    out = []
    for p in points:
        x, y = p[0] * w - cx, p[1] * h - cy
        out.append((((x * c - y * s) + cx) / w, ((x * s + y * c) + cy) / h, p[2]))
    return out


# -- statistics ------------------------------------------------------------


def noise(series):
    """Per-sample noise, from successive differences.

    std(x[i+1] - x[i]) / sqrt(2) estimates the per-sample noise when the true
    signal barely moves between samples, which is what the fixation protocol
    is for. Independent of any constant offset, and unlike the plain standard
    deviation it does not count real eye movement as error.
    """
    if len(series) < 3:
        return float("nan")
    diffs = [b - a for a, b in zip(series, series[1:], strict=False)]
    return statistics.stdev(diffs) / math.sqrt(2.0)


def correlation(xs, ys):
    if len(xs) < 3:
        return float("nan")
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return float("nan")


def pct(series, q):
    ordered = sorted(series)
    if not ordered:
        return float("nan")
    i = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[i]


# -- capture ---------------------------------------------------------------


def capture(args):
    url = args.url.rstrip("/") + "/landmarks"
    seen = set()
    kept = 0
    deadline = time.monotonic() + args.seconds
    no_face = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=3.0) as response:
                    payload = json.load(response)
            except Exception as exc:
                print(f"  fetch failed: {exc}", file=sys.stderr)
                time.sleep(0.5)
                continue
            seq = payload.get("frame")
            if seq in seen:
                # The endpoint serves the newest frame, so polling faster
                # than preview_max_fps returns the same one repeatedly.
                time.sleep(0.01)
                continue
            seen.add(seq)
            if payload.get("count", 0) < REFINED:
                no_face += 1
                continue
            handle.write(json.dumps(payload) + "\n")
            kept += 1
            if kept % 50 == 0:
                print(f"  {kept} frames", flush=True)
    print(f"{kept} frames with a tracked face -> {args.out}")
    if no_face:
        print(f"{no_face} frames had no face, or an unrefined landmark list")
    if kept < 100:
        print(
            "That is thin. Check somebody is actually in frame, and that "
            "preview_max_fps is not throttling below a few hertz.",
            file=sys.stderr,
        )


# -- analysis --------------------------------------------------------------


def analyse(args):
    frames = []
    with open(args.path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    if len(frames) < 10:
        sys.exit(f"{args.path}: only {len(frames)} frames, need at least 10")

    size = frames[0].get("size") or {}
    w = args.width or size.get("width") or 1280
    h = args.height or size.get("height") or 720
    if not (size.get("width") or args.width):
        print(
            f"note: no frame size in the capture, assuming {w}x{h}. Pass "
            "--width/--height if that is wrong; the aspect ratio changes "
            "the old algorithm's dy.\n"
        )

    series = {
        k: []
        for k in (
            "old_dx",
            "old_dy",
            "new_dx",
            "new_dy",
            "old_disp",
            "new_disp",
            "roll",
            "c1_x",
            "c1_y",
            "c5_x",
            "c5_y",
            "width_l",
            "iris_r",
        )
    }
    rolled = {"old": [], "new": []}

    for frame in frames:
        points = frame["points"]
        if len(points) < REFINED:
            continue
        old, new = [], []
        for _name, iris, ring, a_idx, b_idx in EYES:
            o = old_eye(points, iris, ring, a_idx, b_idx, w, h)
            n = new_eye(points, iris, ring, a_idx, b_idx, w, h)
            if o is None or n is None:
                break
            old.append(o)
            new.append(n)
        if len(old) != 2:
            continue

        series["old_dx"].append((old[0][0] + old[1][0]) / 2)
        series["old_dy"].append((old[0][1] + old[1][1]) / 2)
        series["new_dx"].append((new[0][0] + new[1][0]) / 2)
        series["new_dy"].append((new[0][1] + new[1][1]) / 2)
        series["old_disp"].append(abs(old[0][1] - old[1][1]))
        series["new_disp"].append(abs(new[0][1] - new[1][1]))
        series["roll"].append(head_roll_deg(points, w, h))

        x1, y1 = iris_centre_px(points, LEFT_IRIS, LEFT_RING, w, h, five=False)
        x5, y5 = iris_centre_px(points, LEFT_IRIS, LEFT_RING, w, h, five=True)
        series["c1_x"].append(x1)
        series["c1_y"].append(y1)
        series["c5_x"].append(x5)
        series["c5_y"].append(y5)
        series["width_l"].append(eye_width_px(points, LEFT_OUTER, LEFT_INNER, w, h))
        series["iris_r"].append(iris_radius_px(points, LEFT_IRIS, LEFT_RING, w, h))

        # A rigid rotation of the whole face changes no eye-in-head
        # geometry, so any change in the reading is the algorithm's fault.
        turned = rotate(points, args.roll_test, w, h)
        for label, fn in (("old", old_eye), ("new", new_eye)):
            vals = [fn(turned, iris, ring, a, b, w, h) for _n, iris, ring, a, b in EYES]
            if all(v is not None for v in vals):
                rolled[label].append((vals[0][1] + vals[1][1]) / 2)

    n = len(series["old_dy"])
    if n < 10:
        sys.exit(f"only {n} usable frames after filtering")

    mm_per_px = IRIS_MM / (2 * statistics.fmean(series["iris_r"]))
    print(f"=== {args.path}  ({n} frames, {w}x{h}) ===\n")

    print("Iris centre, one landmark against five")
    for axis in ("x", "y"):
        a = noise(series[f"c1_{axis}"])
        b = noise(series[f"c5_{axis}"])
        gain = a / b if b else float("nan")
        print(
            f"  {axis}: {a:6.3f} px -> {b:6.3f} px   "
            f"{gain:4.2f}x less noise  ({a * mm_per_px:.3f} -> "
            f"{b * mm_per_px:.3f} mm)"
        )

    # The two algorithms report dy on different scales -- the old one carries
    # a factor of the frame aspect ratio -- so their raw numbers are not
    # comparable. Nor is noise as a fraction of observed range: the roll
    # artifact inflates the old algorithm's range, which makes its noise look
    # like a smaller share of it and flatters exactly the thing being
    # measured. Converting both back to millimetres of iris travel through
    # each algorithm's own scaling is the only honest comparison.
    eye_px = statistics.fmean(series["width_l"])
    to_mm = {
        "old_dx": eye_px / 2 * mm_per_px,
        "old_dy": eye_px / 4 * mm_per_px * (h / w),
        "new_dx": eye_px / 2 * mm_per_px,
        "new_dy": eye_px / 4 * mm_per_px,
    }
    print("\nGaze noise floor, in millimetres of iris travel")
    print("  (converted through each algorithm's own scaling, because the")
    print("   old dy carries a factor of the aspect ratio)")
    for axis in ("dx", "dy"):
        o = noise(series[f"old_{axis}"]) * to_mm[f"old_{axis}"]
        nw = noise(series[f"new_{axis}"]) * to_mm[f"new_{axis}"]
        gain = o / nw if nw else float("nan")
        print(f"  {axis}: {o:.4f} mm -> {nw:.4f} mm   {gain:4.2f}x better")
    print("\n  raw, in each algorithm's own units, for reference:")
    for axis in ("dx", "dy"):
        print(
            f"    {axis}: old {noise(series[f'old_{axis}']):.4f}  "
            f"new {noise(series[f'new_{axis}']):.4f}  "
            f"(old dy scale is {w / h:.3f}x the new)"
        )

    print("\nEye disparity: the two eyes disagreeing vertically")
    print("  (no vertical vergence exists, so this is all instrument error)")
    for label in ("old", "new"):
        d = series[f"{label}_disp"]
        print(
            f"  {label}: mean {statistics.fmean(d):.4f}  median "
            f"{statistics.median(d):.4f}  p95 {pct(d, 95):.4f}  "
            f"p99 {pct(d, 99):.4f}  max {max(d):.4f}"
        )
    print(f"\n  suggested max_eye_disparity: {pct(series['new_disp'], 99):.3f}")
    print("  (the new p99, so roughly one frame in a hundred is rejected)")

    print("\nHead roll in this capture")
    r = series["roll"]
    print(
        f"  range {min(r):+.1f} to {max(r):+.1f} deg, " f"sd {statistics.stdev(r):.2f}"
    )
    print("  correlation of roll with reported dy:")
    print(f"    old {correlation(r, series['old_dy']):+.3f}")
    print(f"    new {correlation(r, series['new_dy']):+.3f}")
    if max(r) - min(r) < 8:
        print("  too little roll here to say much. Run the roll protocol.")

    print(f"\nSynthetic {args.roll_test:+.0f} deg rotation of the whole face")
    print("  (rigid, so it changes no real eye geometry: all artifact)")
    for label in ("old", "new"):
        if not rolled[label]:
            continue
        base = series[f"{label}_dy"][: len(rolled[label])]
        shift = statistics.fmean(
            b - a for a, b in zip(base, rolled[label], strict=True)
        )
        print(f"  {label}: dy moves by {shift:+.4f}")

    print("\nScale references")
    print(
        f"  eye width {statistics.fmean(series['width_l']):.1f} px, "
        f"iris diameter {2 * statistics.fmean(series['iris_r']):.1f} px, "
        f"so {mm_per_px:.4f} mm/px"
    )
    print(
        f"  eye-width noise {noise(series['width_l']):.3f} px, "
        f"iris-diameter noise {2 * noise(series['iris_r']):.3f} px"
    )
    print("  (whichever is quieter is the better scale reference)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="record raw landmarks from a monitor")
    cap.add_argument("--url", default="http://localhost:8080")
    cap.add_argument("--seconds", type=float, default=30.0)
    cap.add_argument("--out", default="landmarks.jsonl")
    cap.set_defaults(func=capture)

    ana = sub.add_parser("analyse", help="compare the algorithms on a capture")
    ana.add_argument("path")
    ana.add_argument("--width", type=int, default=None)
    ana.add_argument("--height", type=int, default=None)
    ana.add_argument(
        "--roll-test",
        type=float,
        default=20.0,
        help="degrees of synthetic rotation for the artifact test",
    )
    ana.set_defaults(func=analyse)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
