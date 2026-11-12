"""A fixed benchmark for the gaze pipeline, and a scorecard to read it by.

A single simulated run is not evidence: where the four-corner calibration
happens to land moves the result more than most changes do. So the
benchmark is a fixed, versioned suite of cases that varies everything the
pipeline has to be robust to -- the operator's looks and height, how they
move their head, how carefully they commission, the camera's exact mount,
the scenario and the random seed -- and the scorecard pools over all of it,
with confidence intervals from resampling the cases.

What the scorecard puts first is what matters at the saw:

* **missed alarms**: real excursions from the blade that never escalated;
* **false alarms per hour**: what decides whether operators put up with it.

Then latency, time to clear, the time the face is tracked, and the gaze
error and in/out agreement behind them.

Two tiers:

``truth``
    Perfect landmarks, no rendering, no FaceMesh. Seconds per case. Tests
    the gaze model, calibration and policy, and is quick enough for CI.
``mediapipe``
    Renders each case once into a cache, runs FaceMesh, and replays on
    that. Slow the first time; every later run of the suite, for any change
    downstream of FaceMesh, replays the cached landmarks.

Comparisons are paired: two configurations, or this code against a saved
baseline, are scored on identical cases, and the scorecard reports the
per-case difference with its interval. Only a difference whose interval
excludes zero is worth believing.

    python -m tools.operator_sim benchmark --suite quick
    python -m tools.operator_sim benchmark --suite standard --out base.json
    python -m tools.operator_sim benchmark --suite standard --baseline base.json
    python -m tools.operator_sim benchmark --suite standard --set filter_alpha=0.2
    python -m tools.operator_sim benchmark --suite facemesh --landmarks mediapipe
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

from .behaviour import Traits
from .evaluate import Params, Run, generate, score, simulate_truth
from .face import PRESETS
from .placement import drifted
from .scene import CameraMount, Scene

SUITE_VERSION = 1

# How operators differ in moving their heads. "head_mover" turns the head
# for smaller shifts and further; "eye_mover" barely moves it.
TRAITS = {
    "typical": Traits(),
    "head_mover": Traits(
        head_threshold_deg=8.0, head_gain=0.9, max_eccentricity_deg=12.0
    ),
    "eye_mover": Traits(
        head_threshold_deg=22.0, head_gain=0.6, max_eccentricity_deg=28.0
    ),
}
APPEARANCES = ("default", "dark", "light", "medium")


@dataclass(frozen=True)
class Case:
    scenario: str
    duration: float
    seed: int
    appearance: str = "default"
    traits: str = "typical"
    calibration_head: str = "still"
    # Small random offset of the camera mount from the recommended one, as
    # a real installation would have: up to 15 mm and 3 degrees.
    mount_jitter: bool = False

    @property
    def id(self) -> str:
        return (
            f"{self.scenario}-{self.duration:g}s-s{self.seed}-{self.appearance}"
            f"-{self.traits}-{self.calibration_head}"
            + ("-jitter" if self.mount_jitter else "")
        )

    def key(self) -> str:
        """Content hash: what a cached rendering of this case is filed under."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha1(f"{SUITE_VERSION}:{blob}".encode()).hexdigest()[:12]

    def scene(self) -> Scene:
        mount = CameraMount()
        if self.mount_jitter:
            r = np.random.default_rng(self.seed + 7919)
            t = r.uniform(-0.015, 0.015, 3)
            a = r.uniform(-3.0, 3.0, 3)
            mount = drifted(mount, t, a)
        return Scene(mount=mount)

    def sim_kwargs(self) -> dict:
        return dict(
            duration=self.duration,
            seed=self.seed,
            appearance=self.appearance,
            calibration_head=self.calibration_head,
            traits=TRAITS[self.traits],
            scene=self.scene(),
        )


def _rotate(seq, i):
    return seq[i % len(seq)]


def suite(name: str) -> list[Case]:
    """The named suites. Changing one changes every baseline: bump
    SUITE_VERSION when you do, so cached renderings are not reused."""
    if name == "quick":
        # Under a minute, truth-only: for CI and for a fast sanity check.
        return [
            Case("shift", 40, 0),
            Case("shift", 40, 1, appearance="dark", traits="head_mover"),
            Case("shift", 40, 2, appearance="light", traits="eye_mover"),
            Case("glances", 30, 0),
            Case("edges", 30, 0),
            Case("flicker", 20, 0),
            Case("vor", 20, 0),
            Case("shift", 40, 3, calibration_head="natural"),
        ]
    if name == "standard":
        cases = []
        plan = [("shift", 60, 10), ("glances", 40, 4), ("edges", 40, 4),
                ("flicker", 30, 3), ("vor", 20, 3), ("distance", 40, 3)]  # fmt: skip
        k = 0
        for scenario, duration, seeds in plan:
            for seed in range(seeds):
                cases.append(
                    Case(
                        scenario,
                        duration,
                        seed,
                        appearance=_rotate(APPEARANCES, k),
                        traits=_rotate(list(TRAITS), k // 2),
                        mount_jitter=k % 2 == 1,
                    )
                )
                k += 1
        # Natural commissioning tracked separately: not what the checklist
        # asks for, but what happens when nobody reads it.
        for seed in range(4):
            cases.append(
                Case("shift", 60, 100 + seed, appearance=_rotate(APPEARANCES, seed),
                     calibration_head="natural")  # fmt: skip
            )
        return cases
    if name == "facemesh":
        # The rendered tier: small, because each case renders for minutes
        # the first time. Cached afterwards.
        return [
            Case("shift", 60, 0),
            Case(
                "shift",
                60,
                1,
                appearance="dark",
                traits="head_mover",
                mount_jitter=True,
            ),
            Case("shift", 60, 2, appearance="light", traits="eye_mover"),
            Case("shift", 60, 3, appearance="medium", mount_jitter=True),
            Case("glances", 40, 0),
            Case("edges", 40, 0, appearance="dark"),
            Case("flicker", 30, 0, appearance="light"),
            Case("vor", 20, 0),
        ]
    raise ValueError(f"Unknown suite {name!r}: quick, standard or facemesh")


# -- running -------------------------------------------------------------------


def load_case_run(case: Case, landmarks: str, cache: pathlib.Path, render: bool) -> Run:
    if landmarks == "truth":
        return simulate_truth(case.scenario, **case.sim_kwargs())
    path = cache / f"{case.id}-{case.key()}"
    if not (path / "meta.json").exists():
        if not render:
            raise FileNotFoundError(f"{path} is not rendered; run without --no-render")
        kw = case.sim_kwargs()
        generate(
            path,
            scenario=case.scenario,
            duration=kw["duration"],
            seed=kw["seed"],
            appearance=kw["appearance"],
            calibration_head=kw["calibration_head"],
            traits=kw["traits"],
            scene=kw["scene"],
        )
    return Run.load(path)


def run_suite(
    cases: list[Case],
    params: Params | None = None,
    landmarks: str = "truth",
    cache: pathlib.Path = pathlib.Path("runs/bench-cache"),
    render: bool = True,
    progress: bool = True,
) -> list[dict]:
    """Score every case. Returns one compact record per case."""
    params = params or Params()
    out = []
    started = time.monotonic()
    for k, case in enumerate(cases):
        run = load_case_run(case, landmarks, cache, render)
        rep = score(run, params, landmarks=landmarks, ideal=False)
        out.append(_record(case, rep))
        if progress:
            print(
                f"\r  {k + 1}/{len(cases)}  {case.id:<60s}"
                f" {time.monotonic() - started:6.0f}s",
                end="",
                file=sys.stderr,
                flush=True,
            )
    if progress:
        print(file=sys.stderr)
    return out


def _record(case: Case, rep: dict) -> dict:
    """The per-case numbers the scorecard pools."""
    rec = {
        "id": case.id,
        "case": dataclasses.asdict(case),
        "calibrated": rep["calibration_ok"],
    }
    if not rep["calibration_ok"]:
        return rec
    ev = rep.get("events") or {}
    for level in ("warning", "alert"):
        e = ev.get(level) or {}
        rec[level] = {
            k: e.get(k)
            for k in ("true_episodes", "detected", "missed", "false_episodes",
                      "false_seconds", "latencies_s", "clear_lags_s",
                      "minutes")  # fmt: skip
        }
    rec["tracked"] = rep.get("sample_rate")
    rec["gaze_median_deg"] = (rep.get("gaze_error") or {}).get("median_deg")
    rec["in_zone"] = (rep.get("in_zone") or {}).get("accuracy")
    rec["false_out"] = (rep.get("in_zone") or {}).get("false_out")
    rec["eye_in_head"] = rep.get("eye_in_head")
    return rec


# -- the scorecard -------------------------------------------------------------------


def _bootstrap(values_fn, n_cases, reps=2000, seed=0, ci=0.9):
    """Percentile interval of a statistic over cases resampled with replacement."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        idx = rng.integers(0, n_cases, n_cases)
        v = values_fn(idx)
        if v is not None and np.isfinite(v):
            stats.append(v)
    if not stats:
        return None
    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return [round(float(lo), 3), round(float(hi), 3)]


def scorecard(records: list[dict]) -> dict:
    """Pooled metrics over the calibrated cases, each with a 90% interval."""
    ok = [r for r in records if r.get("calibrated")]
    n = len(ok)
    card = {
        "cases": len(records),
        "calibration_failures": len(records) - n,
        "calibration_failed_ids": [r["id"] for r in records if not r.get("calibrated")],
    }
    if n == 0:
        return card

    def arr(path):
        out = []
        for r in ok:
            v = r
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            out.append(np.nan if v is None else float(v))
        return np.asarray(out)

    def pooled_rate(num_path, den_path, scale=1.0):
        num, den = arr(num_path), arr(den_path)

        def f(idx):
            d = np.nansum(den[idx])
            return np.nansum(num[idx]) / d * scale if d > 0 else None

        return f

    def metric(fn):
        return {"value": _r(fn(np.arange(n))), "ci90": _bootstrap(fn, n)}

    for level in ("warning", "alert"):
        card[f"missed_{level}s"] = {
            "missed": int(np.nansum(arr((level, "missed")))),
            "of": int(np.nansum(arr((level, "true_episodes")))),
            "rate": metric(pooled_rate((level, "missed"), (level, "true_episodes"))),
        }
        card[f"false_{level}s_per_hour"] = metric(
            pooled_rate((level, "false_episodes"), (level, "minutes"), 60.0)
        )

    lat = [x for r in ok for x in (r["warning"].get("latencies_s") or [])]
    clr = [x for r in ok for x in (r["warning"].get("clear_lags_s") or [])]
    card["warning_latency_s"] = _summary(lat)
    card["time_to_clear_s"] = _summary(clr)
    for key in ("tracked", "in_zone", "gaze_median_deg"):
        vals = arr((key,))

        def mean(idx, vals=vals):
            v = vals[idx]
            v = v[np.isfinite(v)]
            return float(np.mean(v)) if len(v) else None

        card[f"{key}_mean"] = metric(mean)
    card["by_scenario"] = _by_group(ok, "scenario")
    card["by_traits"] = _by_group(ok, "traits")
    card["by_commissioning"] = _by_group(ok, "calibration_head")
    card["by_appearance"] = _by_group(ok, "appearance")
    card["by_mount"] = _by_group(ok, "mount_jitter")
    return card


def _summary(xs):
    if not xs:
        return None
    xs = np.asarray(xs, float)
    return {"median": _r(np.median(xs)), "p90": _r(np.percentile(xs, 90)), "n": len(xs)}


def _by_group(ok, field):
    groups = {}
    for r in ok:
        groups.setdefault(str(r["case"][field]), []).append(r)
    out = {}
    for name, rs in sorted(groups.items()):
        missed = sum(r["warning"]["missed"] or 0 for r in rs)
        true = sum(r["warning"]["true_episodes"] or 0 for r in rs)
        false = sum(r["warning"]["false_episodes"] or 0 for r in rs)
        minutes = sum(r["warning"]["minutes"] or 0 for r in rs)
        out[name] = {
            "cases": len(rs),
            "missed_warnings": f"{missed}/{true}",
            "false_warnings_per_hour": _r(false / minutes * 60) if minutes else None,
            "in_zone_mean": _r(
                np.mean([r["in_zone"] for r in rs if r["in_zone"] is not None])
            ),
        }
    return out


def compare(base: list[dict], new: list[dict]) -> dict:
    """Paired differences, new minus base, over the cases both calibrated."""
    b = {r["id"]: r for r in base if r.get("calibrated")}
    pairs = [(b[r["id"]], r) for r in new if r.get("calibrated") and r["id"] in b]
    n = len(pairs)
    out = {
        "paired_cases": n,
        "calibration_failures": {
            "base": sum(1 for r in base if not r.get("calibrated")),
            "new": sum(1 for r in new if not r.get("calibrated")),
        },
    }
    if n == 0:
        return out

    def diff(get):
        d = np.array(
            [
                (get(y) if get(y) is not None else np.nan)
                - (get(x) if get(x) is not None else np.nan)
                for x, y in pairs
            ]
        )

        def f(idx):
            v = d[idx]
            v = v[np.isfinite(v)]
            return float(np.mean(v)) if len(v) else None

        return {"mean_diff": _r(f(np.arange(n))), "ci90": _bootstrap(f, n)}

    def per_hour(level):
        def g(r):
            m = r[level]["minutes"]
            return r[level]["false_episodes"] / m * 60 if m else None

        return g

    out["missed_warnings"] = diff(lambda r: r["warning"]["missed"])
    out["missed_alerts"] = diff(lambda r: r["alert"]["missed"])
    out["false_warnings_per_hour"] = diff(per_hour("warning"))
    out["false_alerts_per_hour"] = diff(per_hour("alert"))
    out["in_zone"] = diff(lambda r: r["in_zone"])
    out["gaze_median_deg"] = diff(lambda r: r["gaze_median_deg"])
    out["tracked"] = diff(lambda r: r["tracked"])
    return out


def _r(x, nd=3):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def format_scorecard(card: dict) -> str:
    lines = []
    w = card.get("missed_warnings") or {}
    a = card.get("missed_alerts") or {}

    def ci(m):
        if not m or m.get("value") is None:
            return "-"
        c = m.get("ci90")
        return f"{m['value']:.3g}" + (f"  [{c[0]:.3g}, {c[1]:.3g}]" if c else "")

    lines.append(
        f"cases {card['cases']}, calibration failures {card['calibration_failures']}"
    )
    if w:
        lines.append(
            f"missed warnings     {w['missed']} of {w['of']}   rate {ci(w['rate'])}"
        )
        lines.append(
            f"missed alerts       {a['missed']} of {a['of']}   rate {ci(a['rate'])}"
        )
        lines.append(f"false warnings/hour {ci(card['false_warnings_per_hour'])}")
        lines.append(f"false alerts/hour   {ci(card['false_alerts_per_hour'])}")
        lat, clr = card.get("warning_latency_s"), card.get("time_to_clear_s")
        if lat:
            lines.append(
                f"warning latency     median {lat['median']} s,"
                f" p90 {lat['p90']} s (n={lat['n']})"
            )
        if clr:
            lines.append(
                f"time to clear       median {clr['median']} s,"
                f" p90 {clr['p90']} s (n={clr['n']})"
            )
        lines.append(f"face tracked        {ci(card['tracked_mean'])}")
        lines.append(f"in-zone agreement   {ci(card['in_zone_mean'])}")
        lines.append(f"gaze error, median  {ci(card['gaze_median_deg_mean'])} deg")
        for group in ("scenario", "traits", "commissioning", "appearance", "mount"):
            table = card.get(f"by_{group}") or {}
            if len(table) < 2 and group != "scenario":
                continue
            lines.append("")
            lines.append(
                f"{group:<14} {'cases':>5} {'missed':>8} {'false/h':>8} {'in-zone':>8}"
            )
            for name, g in table.items():
                lines.append(
                    f"{name:<14} {g['cases']:>5} {g['missed_warnings']:>8}"
                    f" {_fmt(g['false_warnings_per_hour']):>8}"
                    f" {_fmt(g['in_zone_mean']):>8}"
                )
    if card.get("calibration_failed_ids"):
        lines.append("")
        lines.append("did not calibrate: " + ", ".join(card["calibration_failed_ids"]))
    return "\n".join(lines)


def format_comparison(cmp: dict) -> str:
    lines = [
        f"paired cases {cmp['paired_cases']}, calibration failures"
        f" base {cmp['calibration_failures']['base']}"
        f" -> new {cmp['calibration_failures']['new']}"
    ]
    for key in ("missed_warnings", "missed_alerts", "false_warnings_per_hour",
                "false_alerts_per_hour", "in_zone", "gaze_median_deg",
                "tracked"):  # fmt: skip
        d = cmp.get(key)
        if not d or d["mean_diff"] is None:
            continue
        c = d["ci90"]
        sig = ""
        if c and (c[0] > 0 or c[1] < 0):
            sig = "  *"
        lines.append(
            f"{key:<24} {d['mean_diff']:+.3g}   [{c[0]:+.3g}, {c[1]:+.3g}]{sig}"
            if c
            else f"{key:<24} {d['mean_diff']:+.3g}"
        )
    lines.append("(new minus base, per case; * = 90% interval excludes zero)")
    return "\n".join(lines)


def _fmt(v):
    return "-" if v is None else f"{v:.3g}"


__all__ = ["Case", "suite", "run_suite", "scorecard", "compare", "PRESETS"]
