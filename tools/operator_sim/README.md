# Simulated saw operator

A rendered operator standing at a 14" floor-standing meat band saw, seen by
the monitor's own camera, for tuning and evaluating the gaze pipeline
against ground truth that no real camera can give.

Each simulated frame is a 1280×720 BGR image plus an aligned depth image in
metres: exactly what `RealSenseSource` delivers. It comes with a record of
what the operator was actually doing: where the line of sight met the
attention zone, how each eye was rotated in the head, the head's pose, the
distance to the camera, whether the eyes were open, and the true pixel
position of every one of FaceMesh's 478 landmarks.

```bash
# Render a minute of work, run FaceMesh on it, keep everything for replay
python -m tools.operator_sim generate --scenario shift --seconds 60 --out runs/shift --video

# Replay the real pipeline on it and score it against the truth
python -m tools.operator_sim evaluate runs/shift

# Tune: every combination, judged on identical frames
python -m tools.operator_sim sweep runs/shift runs/glances \
    --grid filter_alpha=0.08,0.12,0.2,0.35 --grid max_eye_disparity=0,0.3

# Look at a frame
python -m tools.operator_sim snapshot --scenario eyes-only --at 2 5 8 --out op.png --depth

# Point the real monitor (demo.py) at a recorded run
python -m tools.operator_sim generate --scenario shift --out runs/live --save-frames
python -m tools.operator_sim live runs/live -- --tui --recalibrate
```

Run these from the repository root. `generate` needs the deployment stack
(`requirements.txt`: MediaPipe and OpenCV). Everything else, the renderer
included, needs only numpy. `generate` renders on one process per CPU and
manages about 5 frames a second on four cores, so a one-minute scenario takes
about seven minutes. `--ssaa 1` is roughly twice as fast, at the cost of
aliased iris edges.

## The operator

**The face** is MediaPipe's canonical face model (Apache-2.0, see
[`assets/NOTICE`](assets/NOTICE)). It has the proportions FaceMesh was
trained on, and its 468 vertices *are* the FaceMesh landmarks, so ground
truth for the landmark detector comes for free. It is textured procedurally:
skin, lips, eyebrows, a lash line and a lid crease, the contrast cues
FaceMesh locks onto. The model's flat eye openings are cut out and replaced
by:

* **Eyeballs** that rotate independently of the head, 24 mm across, with an
  11.7 mm iris, a pupil, a limbal ring, and a specular highlight off a
  separate corneal curvature. The iris landmarks are the true limbus.
* **Eyelids** that blink, with the fast-close, slow-open profile, and that
  **follow vertical gaze**. The upper lid follows by about 0.7 of the eye's
  elevation and the lower by about 0.45. For an operator looking down at a
  blade this is not a detail: the lid comes down over the iris, and that is
  a large part of why downward gaze is hard to measure.

A skull with hair, a white cap or a hairnet, ears, a neck, a torso in a
white coat and apron, and arms reaching for the work complete the operator.
`--appearance` picks one of four presets (`default`, `light`, `medium`,
`dark`), which vary skin, iris and hair colour, headwear, face width and
stature.

**The motion** is driven by gaze, the thing people actually control:

| | |
| --- | --- |
| Saccades | Main-sequence duration (21 ms + 2.2 ms/°) and the skewed velocity profile |
| Eye–head coordination | Shifts under 15° use the eyes alone. Larger ones bring the head in 20–50 ms after the eyes, an order of magnitude more slowly, stopping with the eyes up to 10° off-centre |
| Vestibulo-ocular reflex | Once gaze lands it stays on the target while the head is still moving, so the eyes counter-rotate in the head |
| Head re-centring | During a held fixation, the head slowly takes up part of a large eye eccentricity, with gaze held by VOR |
| Vergence | Each eye aims at the fixated point itself |
| Fixational movements | Drift and corrective microsaccades |
| Blinks | About 17 a minute, some partial, plus blinks accompanying large gaze shifts |
| Posture | Sway, breathing, forward lean over the table, steps |
| Oculomotor range | Hard limits. When the eyes pin, the truth reports where they actually point, not the intended target |

Individual differences in head movement are a `Traits` object. Some people
move their heads, others their eyes, and an eye-in-head measurement is
sensitive to which.

## The saw and the camera

Gross dimensions come from the general-arrangement drawing (1:8): 1.80 m
overall, table at 0.915 m, underside of the upper wheel housing at 1.34 m,
housing 0.48 m wide. The cutting side of the blade runs down the left of the
head as the operator sees it, the column is on the right, and the sliding
carriage is to the left of the blade. The geometry is in
[`scene.py`](scene.py); nothing of the drawing itself is reproduced.

The camera is a D435i at 1280×720 (70.4° × 43.3°), mounted under the front
edge of the upper wheel housing just right of the blade guide, aimed up at
the operator's eyes. That gives the 0.55–0.65 m working distance the
pipeline was tuned at. It is also why a camera there works at all: the
operator looks *down* at the blade, so a camera below eye level sees the face
nearly frontally. `CameraMount` moves it.

The **attention zone** runs from the bottom of the upper blade guide down the
blade to the table, and across the strip of table in front of the teeth:
24 cm wide, tilted towards the operator, about 17° × 16° as seen from the
eyes.

The sensor model is deliberately unflattering. The colour image gets optical
blur, shot and read noise, and 8-bit quantisation. Depth error grows with the
square of distance (about 1 mm at the working distance), quantised to the
1 mm depth unit. It also has the stereo occlusion shadow down the left of
everything nearer than its background, and random dropouts.

## Scenarios

| Scenario | What the operator does | What it is for |
| --- | --- | --- |
| `shift` | Realistic work generated from a task model: watching the cut with small re-fixations, checking the product, reaching for the carriage, unloading to the tray, the controls, talking to coworkers, the clock, the floor, the camera, occasional steps | Overall behaviour, false alarm rate |
| `glances` | Timed excursions out of the zone (0.5, 1.5, 2.5, 4 s) | Alarm latency and missed alarms |
| `eyes-only` | Head held still; the eyes alone go to targets in and out of the zone | Eye movement independent of the head |
| `vor` | Gaze held on the cut while the head turns | Whether head motion reads as gaze motion |
| `fixation` | Still, staring at the cut | The noise floor, as in `tools/gaze_noise.py` |
| `roll` | Staring at the cut, tilting the head ±14° | Roll leaking into the reading |
| `distance` | Watching the cut while stepping back and forth | Depth compensation |

Every scenario starts with commissioning: the operator fixates the four zone
corners in order, and the evaluator "presses c" a second into each
fixation, pressing again if the calibrator rejects one.
`--calibration-head still` models an operator who keeps their head still
while calibrating. The default is natural eye–head coordination, which turns
the head for the diagonal corner-to-corner shifts.

## What `evaluate` reports

`evaluate` replays everything downstream of FaceMesh with the package's own
classes: `GazeMeasurer`, the disparity gate, the EMA, `GazeCalibrator` and
`AttentionMonitor`. That takes milliseconds per minute of footage. It is
scored against an **oracle**: the same `AttentionMonitor`, with the same
thresholds, fed the operator's true line of sight. A disagreement is
therefore always the measurement's fault, never the state machine's policy.

| Key | |
| --- | --- |
| `landmarks` | FaceMesh against the rendered truth, in pixels: iris centres, eye corners, all 468 |
| `eye_in_head` | How well the raw `(dx, dy)` track the eyes' true rotation in the head: an affine fit, residual in degrees. Independent of calibration and filtering. **The number to watch when changing the landmark geometry** |
| `gaze_error` | Calibrated zone position against the truth, in degrees, smoothed and unsmoothed |
| `in_zone` | Per-frame agreement on in/out: `false_out` is a false alarm in the making, `false_in` is an excursion missed |
| `events` | Alarm episodes at WARNING and ALERT: detected, missed, latency, false episodes per minute |
| `depth` | Valid depth readings and their error |
| `by_activity` | All of the above, split by what the operator was doing |
| `ideal_landmarks` | The same run replayed on **perfect** landmarks |

The last one is how to read the rest. With perfect landmarks, whatever error
remains belongs to the gaze model: calibration, eye-in-head as a stand-in for
gaze, the filter. The difference between the two runs belongs to the
landmark detector. `evaluate.simulate_truth()` produces a truth-only run, with
no rendering and no FaceMesh, in seconds. Use it for experiments on the model
alone.

## What it has shown so far

These are from short runs. Treat them as what to look at, not as results.

* **The geometry checks out.** With perfect landmarks, a still-headed
  calibration and head-still eye movements, `(dx, dy)` track the true eye
  rotation to about 0.5° in both axes. The calibrated zone position comes in
  at a 2–2.5° median error, mostly EMA lag around saccades. The truth, the
  renderer and the replay agree with each other, and `tests/` pins this.
* **FaceMesh is the larger error, and vertically it is nearly blind** on
  these frames. On the same frames, MediaPipe's landmarks give an
  eye-in-head residual of about 3.5–5° in yaw and 5° in pitch, with pitch
  correlating only about 0.5 with the true eye elevation. Most of the yaw
  error comes from the **eye corners** (about 4 px median error), not the
  iris (about 1 px). Most of the pitch error comes from the iris. Whether a
  real face does better is the open question: compare `landmarks` and
  `eye_in_head` with `tools/gaze_noise.py` captures from the hardware.
* **Eye-in-head is not gaze.** In the `vor` scenario the line of sight never
  leaves the zone, yet the pipeline calls it out of zone much of the time,
  even with perfect landmarks: it measures the eye in the head, and when the
  head turns with gaze held, the eye turns the other way. The same mechanism
  makes calibration depend on how the operator moved their head while
  commissioning. An operator who turns their head towards each corner
  calibrates a zone that is too small. Head pose is the obvious next input,
  and the simulator's `head_*_deg` truth is there to fit it against.

## Limitations

* FaceMesh behaves differently on a rendered face than on a real one, in
  ways the simulator cannot settle. Absolute landmark numbers are a
  comparison point, not a prediction: compare a run with the noise tool's
  hardware captures before trusting a conclusion about landmarks. The
  ideal-landmark replay has no such caveat.
* No cast shadows, no glasses glare (the safety-glasses option draws frames
  only), no motion blur, no auto-exposure.
* The `live` command plays back a recorded run at its frame rate. Rendering
  on the fly is far too slow for a monitor whose timers are in wall-clock
  seconds.
