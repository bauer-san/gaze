"""tools/iris_gain.py recovers a known iris gain from synthetic captures."""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest
from test_headpose import HGT, W, face, rot

from gaze_monitor import gaze as G
from gaze_monitor import headpose as H

spec = importlib.util.spec_from_file_location(
    "iris_gain", pathlib.Path(__file__).resolve().parents[1] / "tools" / "iris_gain.py"
)
IG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IG)

RINGS = (G.LEFT_IRIS_RING, G.RIGHT_IRIS_RING)
IRISES = (G.LEFT_IRIS, G.RIGHT_IRIS)


def frame(head_yaw, head_pitch, eye_yaw, eye_pitch, gain, rng, t):
    """One captured frame: the head turned by (head_yaw, head_pitch), the eye
    rotated in it by (eye_yaw, eye_pitch), seen by a tracker that registers
    ``gain`` of the geometric iris offset, plus a little noise."""
    lm, _ = face(rot(head_yaw, head_pitch, 0))
    gx, gy = IG.geometric_per_unit()
    dx = gain[0] * gx * math.sin(math.radians(eye_yaw)) + rng.normal(0, 0.005)
    dy = gain[1] * gy * math.sin(math.radians(eye_pitch)) + rng.normal(0, 0.005)
    for (x, y), iris, ring in zip(
        G.project_gaze(lm, dx, dy, W, HGT), IRISES, RINGS, strict=True
    ):
        lm[iris, :2] = (x, y)
        for k, idx in enumerate(ring):
            a = k * math.pi / 2
            lm[idx, :2] = (x + 6 * math.cos(a) / W, y + 6 * math.sin(a) / HGT)
    return {"points": lm.tolist(), "size": {"width": W, "height": HGT}, "t": t}


def session(tmp_path, gain, head_moves=False, seed=0):
    rng = np.random.default_rng(seed)
    meta = {"subject": "synthetic", "distance_m": 0.6, "hold_s": 2.0, "discard_s": 0.5}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    for rep in (1, 2):
        for name, yaw, pitch in IG.MARKS:
            # A head that drifts a few degrees between holds: the eye makes
            # up the difference, so the target stays fixated.
            hy, hp = (rng.uniform(-4, 4), rng.uniform(-4, 4)) if head_moves else (0, 0)
            lines = [
                json.dumps(frame(hy, 20 + hp, yaw - hy, pitch - hp, gain, rng, k / 30))
                for k in range(60)
            ]
            (tmp_path / f"r{rep}-{name}.jsonl").write_text("\n".join(lines))
    return tmp_path


@pytest.mark.parametrize("gain", [(1.0, 1.0), (0.5, 0.08)])
def test_recovers_the_gain(tmp_path, gain):
    _, holds, _ = IG.load_session(session(tmp_path, gain))
    fit = IG.fit_gains(holds)
    assert fit["x"]["gain"] == pytest.approx(gain[0], abs=0.05)
    assert fit["y"]["gain"] == pytest.approx(gain[1], abs=0.05)


def test_head_movement_is_subtracted(tmp_path):
    _, holds, _ = IG.load_session(session(tmp_path, (1.0, 1.0), head_moves=True))
    fit = IG.fit_gains(holds)
    assert fit["head_movement_deg_max"] > 2
    # FaceMesh-depth head pose, so a few percent, not exact.
    assert fit["y"]["gain"] == pytest.approx(1.0, abs=0.1)
    assert fit["x"]["gain"] == pytest.approx(1.0, abs=0.1)


def test_verdicts():
    def v(g, n):
        return IG.verdict({"y": {"gain": g, "noise_deg": n}})

    assert v(0.8, 1.0).startswith("VIABLE")
    assert v(0.8, 6.0).startswith("MARGINAL")
    assert v(0.3, 1.0).startswith("MARGINAL")
    assert v(0.08, 1.0).startswith("CONFIRMED")
    assert v(-0.7, 1.0).startswith("INVERTED")


def test_motion_reads_a_perfect_vor(tmp_path):
    rng = np.random.default_rng(1)
    rows = IG.measure_frames(
        [
            frame(
                0, 20 + 12 * math.sin(k / 8), 0, -12 * math.sin(k / 8), (1, 1), rng, k
            )
            for k in range(120)
        ]
    )
    m = IG.analyse_motion(rows)["pitch"]
    assert m["vor_gain"] == pytest.approx(1.0, abs=0.15)
    assert m["gaze_sd_deg"] < 0.3 * m["head_sd_deg"]


def test_mark_descriptions_give_offsets():
    assert IG.describe("up20", 0, -20, 0.6) == "UP 20 (21.8 cm from the centre)"
    assert IG.describe("right10", -10, 0, 0.6) == "RIGHT 10 (10.6 cm from the centre)"
    assert H.CANONICAL_EYE_WIDTH > 0
