"""Command line for the simulated saw operator.

    python -m tools.operator_sim generate --scenario shift --out runs/shift
    python -m tools.operator_sim evaluate runs/shift
    python -m tools.operator_sim sweep runs/shift --grid filter_alpha=0.08,0.12,0.2
    python -m tools.operator_sim snapshot --scenario eyes-only --at 3.0 --out frame.png
    python -m tools.operator_sim live runs/shift -- --tui --recalibrate
    python -m tools.operator_sim placement --x=-0.15,0.1,0.025 --z=0.07,0.19,0.03
    python -m tools.operator_sim drift

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .behaviour import SCENARIOS
from .face import PRESETS


def _xyz(text: str) -> tuple:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z")
    return tuple(parts)


def _sim_args(p: argparse.ArgumentParser, seconds: float = 60.0) -> None:
    p.add_argument("--scenario", default="shift", choices=sorted(SCENARIOS))
    p.add_argument(
        "--seconds",
        type=float,
        default=seconds,
        help="scenario length, after calibration",
    )
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--appearance", default="default", choices=sorted(PRESETS))
    p.add_argument(
        "--ssaa", type=int, default=2, help="supersampling of the head (1 is fastest)"
    )
    p.add_argument(
        "--calibration-head",
        default="natural",
        choices=("natural", "still"),
        help="how the operator moves their head while calibrating",
    )
    p.add_argument(
        "--camera",
        type=_xyz,
        metavar="X,Y,Z",
        help="camera position in metres: x to the operator's right of the blade, "
        "y up from the floor, z from the teeth towards the operator",
    )
    p.add_argument(
        "--aim",
        type=_xyz,
        metavar="X,Y,Z",
        help="point the camera is aimed at (default: the operator's nose bridge)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.operator_sim",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="render a scenario and run FaceMesh on it")
    _sim_args(g)
    g.add_argument("--out", required=True, type=pathlib.Path)
    g.add_argument(
        "--workers", type=int, default=0, help="render processes (0: one per CPU)"
    )
    g.add_argument(
        "--video", action="store_true", help="also write an annotated video.mp4"
    )
    g.add_argument(
        "--save-frames",
        action="store_true",
        help="also write every colour frame (jpg) and depth frame (16-bit png, mm)",
    )

    e = sub.add_parser(
        "evaluate", help="replay the pipeline on generated runs and score it"
    )
    e.add_argument("runs", nargs="+", type=pathlib.Path)
    e.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a tuning parameter, e.g. --set filter_alpha=0.2",
    )
    e.add_argument("--json", type=pathlib.Path, help="write the full report here")

    s = sub.add_parser(
        "sweep", help="score a grid of tuning parameters on generated runs"
    )
    s.add_argument("runs", nargs="+", type=pathlib.Path)
    s.add_argument("--grid", action="append", required=True, metavar="NAME=V1,V2,...")
    s.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    s.add_argument("--csv", type=pathlib.Path)

    sn = sub.add_parser("snapshot", help="render single frames to look at")
    _sim_args(sn, seconds=10.0)
    sn.add_argument(
        "--at",
        type=float,
        nargs="+",
        default=[3.0],
        help="times, seconds from the start",
    )
    sn.add_argument("--out", type=pathlib.Path, default=pathlib.Path("operator.png"))
    sn.add_argument(
        "--depth", action="store_true", help="also write a false-colour depth image"
    )

    pl = sub.add_parser(
        "placement", help="score candidate camera mounts (no rendering)"
    )
    pl.add_argument("--scenario", default="shift", choices=sorted(SCENARIOS))
    pl.add_argument("--seconds", type=float, default=60.0)
    pl.add_argument("--seed", type=int, default=0)
    for axis, default in (("x", "-0.15,0.1,0.025"), ("y", "1.365,1.43,0.03"),
                          ("z", "0.07,0.19,0.03")):  # fmt: skip
        pl.add_argument(
            f"--{axis}",
            default=default,
            metavar="FROM,TO,STEP",
            help=f"candidate {axis} positions, metres (default {default})",
        )
    pl.add_argument("--top", type=int, default=20, help="rows to print")
    pl.add_argument("--csv", type=pathlib.Path)

    dr = sub.add_parser(
        "drift", help="how far the mount may move after calibration (no rendering)"
    )
    dr.add_argument("--scenario", default="shift", choices=sorted(SCENARIOS))
    dr.add_argument("--seconds", type=float, default=90.0)
    dr.add_argument("--seed", type=int, default=0)
    dr.add_argument("--camera", type=_xyz, metavar="X,Y,Z", help="where it was set up")

    pr = sub.add_parser("promo", help="render the promotional video")
    pr.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("docs/media/operator-promo.mp4"),
    )
    pr.add_argument("--seed", type=int, default=3)
    pr.add_argument("--workers", type=int, default=0)
    pr.add_argument(
        "--keep",
        type=pathlib.Path,
        help="keep intermediate renders here (and reuse them)",
    )

    lv = sub.add_parser("live", help="run the real monitor (demo.py) on a recorded run")
    lv.add_argument("run", type=pathlib.Path, help="a run generated with --save-frames")
    lv.add_argument(
        "--once", action="store_true", help="stop at the end instead of looping"
    )
    lv.add_argument(
        "demo_args", nargs=argparse.REMAINDER, help="arguments for demo.py, after a --"
    )
    return parser


def _overrides(items) -> dict:
    from .evaluate import parse_grid

    grid = parse_grid(items)
    return {k: v[0] for k, v in grid.items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        from .evaluate import generate

        out = generate(
            args.out, scenario=args.scenario, duration=args.seconds, fps=args.fps,
            seed=args.seed, appearance=args.appearance, workers=args.workers,
            calibration_head=args.calibration_head, camera=args.camera, aim=args.aim,
            ssaa=args.ssaa, video=args.video, save_frames=args.save_frames,
        )  # fmt: skip
        print(f"Wrote {out}")
        return 0

    if args.command in ("evaluate", "sweep"):
        import dataclasses

        from .evaluate import (
            Params,
            Run,
            format_table,
            parse_grid,
            score,
            summarise,
            sweep,
        )

        runs = [Run.load(p) for p in args.runs]
        base = dataclasses.replace(Params(), **_overrides(args.set))
        if args.command == "evaluate":
            reports = [score(run, base) for run in runs]
            for path, rep in zip(args.runs, reports, strict=True):
                print(f"== {path}")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in rep.items()
                            if k not in ("params", "calibration")
                        },
                        indent=2,
                    )
                )
            if len(reports) > 1:
                print("== all runs")
                print(json.dumps(summarise(reports), indent=2))
            if args.json:
                args.json.write_text(json.dumps(reports, indent=2))
            return 0
        rows = sweep(runs, parse_grid(args.grid), base)
        print(format_table(rows))
        if args.csv:
            import csv

            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        return 0

    if args.command == "placement":
        return _placement(args)

    if args.command == "promo":
        from .promo import render_promo

        paths = render_promo(
            args.out, seed=args.seed, workers=args.workers, keep=args.keep
        )
        for k, v in paths.items():
            print(
                f"{k}: {v} ({v.stat().st_size / 1e6:.2f} MB)"
                if v.exists()
                else f"{k}: -"
            )
        return 0

    if args.command == "drift":
        import dataclasses

        from .evaluate import format_table
        from .placement import drift_study
        from .scene import CameraMount

        mount = CameraMount()
        if args.camera is not None:
            mount = dataclasses.replace(mount, position=args.camera)
        rows = drift_study(
            mount=mount, scenario=args.scenario, duration=args.seconds, seed=args.seed
        )
        print(format_table(rows))
        return 0

    if args.command == "snapshot":
        return _snapshot(args)

    if args.command == "live":
        from .source import run_live

        demo_args = [a for a in args.demo_args if a != "--"]
        return run_live(demo_args, args.run, loop=not args.once)
    return 2


def _placement(args) -> int:
    import numpy as np

    from .evaluate import format_table
    from .placement import grid, study
    from .scene import CameraMount

    def rng(text):
        a, b, step = (float(v) for v in text.split(","))
        return [round(float(v), 4) for v in np.arange(a, b + step / 2, step)]

    candidates = grid(rng(args.x), rng(args.y), rng(args.z))
    current = tuple(CameraMount().position)
    if current not in candidates:
        candidates.append(current)
    print(f"{len(candidates)} mountable candidates", file=sys.stderr)
    rows = study(candidates, args.scenario, args.seconds, args.seed)
    for r in rows:
        r["default"] = (
            "*"
            if (r["x"], r["y"], r["z"]) == tuple(round(v, 3) for v in current)
            else ""
        )
    rows.sort(key=lambda r: r["face_deg"] + 0.5 * r["iris_deg"])
    print(format_table(rows[: args.top]))
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return 0


def _snapshot(args) -> int:
    import cv2
    import numpy as np

    from .sim import Simulation

    sim = Simulation(
        args.scenario, duration=args.seconds, fps=args.fps, seed=args.seed,
        appearance=args.appearance, ssaa=args.ssaa,
        calibration_head=args.calibration_head, camera=args.camera, aim=args.aim,
    )  # fmt: skip
    wanted = sorted(int(round(t * args.fps)) for t in args.at)
    out = args.out
    for k, idx in enumerate(wanted):
        state = None
        while sim._index <= idx:
            i, state = sim.next_state()
        frame = sim.render_state(i, state)
        path = out if len(wanted) == 1 else out.with_name(f"{out.stem}_{k}{out.suffix}")
        cv2.imwrite(str(path), frame.render.color)
        if args.depth:
            d = frame.render.depth_m
            vis = cv2.applyColorMap(
                np.clip(d / 2.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )
            vis[d == 0] = 0
            cv2.imwrite(str(path.with_name(path.stem + "_depth.png")), vis)
        tr = frame.truth
        print(
            f"{path}: t={tr['t']:.2f}s {tr['activity']} -> {tr['target']}  "
            f"zone u,v={tr['zone_u']:+.2f},{tr['zone_v']:+.2f}  "
            f"eye yaw/pitch={tr['eye_yaw_deg'][0]:+.1f}/{tr['eye_pitch_deg'][0]:+.1f}  "
            f"head yaw/pitch/roll={tr['head_yaw_deg']:+.1f}/"
            f"{tr['head_pitch_deg']:+.1f}/{tr['head_roll_deg']:+.1f}  "
            f"dist={tr['distance_m']:.2f}m"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
