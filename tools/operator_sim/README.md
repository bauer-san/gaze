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
white coat and long apron, legs, boots and gloves complete the operator.
While cutting, the right hand drives the pusher plate's grip and the left
holds the carriage's hand rail. The plate bears on the back of the product
and rides with it, so both hands stay behind it as the carriage makes its
cutting stroke: a slow push through the blade and a quicker draw back.
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

The camera is a D435i at 1280×720 (70.4° × 43.3°), on the mount the
placement study below recommends: a short bracket off the front-left corner
of the upper wheel housing, aimed up at the operator's face. `--camera
X,Y,Z` (and `--aim`) moves it for any command that renders, and
`CameraMount` in code.

### Where to put the camera

`python -m tools.operator_sim placement` scores a grid of candidate mounts
from ground truth, without rendering, in under a minute. Each candidate is
aimed at the operator's median eye position while they watch the cut (the
aim barely matters: see the drift study below). It is
scored on:

* how head-on it sees the face
* how obliquely it sees the iris
* whether it is below or above the line of sight
* iris size in pixels
* how often the saw, the product or the operator's own arms block an eye

Candidates are restricted to what standoffs or a bracket can reach: within
0.15 m of the housing, and never into the **product envelope**. The housing's
underside is the saw's maximum cut height, and product on the carriage, and
the pieces coming off the blade, pass beneath it across the whole table. So
nothing can hang below it there.

**Where the operator stands decides the distances.** The operator stands
close in, with the front of their body a hand's width behind the carriage's
hand rail, leaning over the table. That is where the product is: the cut is
about 0.35 m beyond the table's front edge. Their eyes end up about 0.4 m in
front of the blade and 1.57 m up, roughly 0.65 m from the cut, which is the
working distance the pipeline was tuned at. (An earlier version of the
simulator stood the operator about 0.2 m further back. The FaceMesh
comparisons made then are withdrawn rather than kept with numbers that no
longer apply.)

Scored from ground truth over a minute of the `shift` scenario:

| Mount (from the blade, as the operator sees it) | Face off-axis | Iris seen at | Camera to eyes | Iris | Eye blocked |
| --- | --- | --- | --- | --- | --- |
| Previous: 100 mm right, 52.2 in, 15 mm in front of the housing (hangs into the product envelope) | 13.3° | 25.1° | 0.45 m | 24 px | 0% |
| **Recommended: 50 mm left, 53.7 in, 20 mm in front of the housing** | 13.7° | 26.6° | 0.41 m | 26 px | 0% |
| 50 mm left, 53.7 in, 70 mm in front | 11.4° | 23.2° | 0.37 m | 29 px | 0% |
| In line with the blade, 53.7 in, 70 mm in front | 8.6° | 22.3° | 0.36 m | 29 px | 0% |
| Guide-post side, 53 in, 110 mm behind the teeth | 20.0° | 33.8° | 0.58 m | 18 px | 40% |

Side to side, at the recommended height and depth:

| Camera, from the blade | 150 mm left | 100 mm left | 50 mm left | in line | 50 mm right | 100 mm right | 150 mm right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Face off-axis | 23.4° | 17.5° | 13.7° | 11.5° | 12.0° | 16.1° | 21.6° |

What decides it:

* **Side to side** is forgiving within about 50 mm either way of the blade.
  The operator stands about 40 mm left of the blade, with their face turned
  about 5° to their right, towards the cut, so the best line is roughly
  in line with the blade. Anywhere within 50 mm of it is within about 2° of
  the best; past 100 mm the face turns away fast. 50 mm left puts the camera
  at the housing's left edge, where a bracket or suction arm can reach, and
  out of the operator's line of sight to the top of a raised blade guide.
  The *previous* position, 100 mm right, is about as good on angle; what
  rules it out is that it hung 15 mm below the housing, into the product.
* **Height** is fixed by the product clearance. The camera's centre is 25 mm
  above the housing's underside, so its body clears the maximum cut height
  by about 10 mm. Every mount allowed there is *above* the operator's line of
  sight to the cut, by 20–25°, so it looks down onto the upper eyelid, which
  follows the eye down. That is the price of the clearance, and the main
  reason vertical gaze is the weaker axis.
* **Forward** is a trade between angle and **framing**. At 0.36–0.4 m the
  operator's face nearly fills the D435's 42° vertical field. With the camera
  70 mm proud of the housing, the chin and forehead come within about 2° of
  the frame's edges as the operator moves. 20 mm proud buys 5–7° of margin
  for a couple of degrees of angle, and it keeps the camera out of the
  operator's line of sight to the top of a raised blade guide. Turning the
  camera on its side (portrait) would give the face 70° of vertical field;
  the pipeline would need to rotate the frame back before FaceMesh, which
  it does not do today.
* The **guide-post side** is further away, sees the face from the side, and
  the guide and its arm block an eye in about 30% of frames.

The recommended mount, in the drawing's terms:

| | |
| --- | --- |
| Camera centre height | 53.7 in (1.365 m) from the floor; body bottom 0.4 in above the housing's underside |
| Side to side | At the housing's left edge: about 2 in to the operator's left of the blade |
| Front to back | About 0.8 in in front of the housing's front face |
| Aim | Up about 27°, and about 4° to the right, at the bridge of the operator's nose |

Aim at the nose rather than the eyes: that centres the whole face, which
FaceMesh needs.

### Commissioning and head movement

From where the operator stands, the zone's bottom corners are as much as 27°
to either side of their eyes. Asked to "look at each corner", most people
turn their head for shifts that size. The eye model measures the eye *in
the head*, so a head turned during commissioning shrinks its calibrated
zone. The head model adds the head's pose and does not care. With perfect
landmarks, so that only this changes (two seeds each, 40 s):

| Commissioning | Scenario | Eye model: gaze error / in-zone | Head model: gaze error / in-zone |
| --- | --- | --- | --- |
| Head held still | `eyes-only` | 2.5–2.8° / 90–98% | 2.8–3.3° / 90–96% |
| Natural head movement | `eyes-only` | 19° / 60%, other seed failed | 2.9–5.2° / 83–88% |
| Head held still | `shift` | 1.8–4.1° / 96–97% | 3.0° / 90–97% |
| Natural head movement | `shift` | 12° / 63%, other seed failed | 1.9–2.8° / 98% |

"Failed" means the calibrator rejected the corners as "too much movement":
the head was still turning while the operator held the fixation.

With the head model, commissioning no longer depends on the operator
keeping their head still, given landmarks that track the eye; see the
MediaPipe results below for why that is not yet the case. With the eye model it does, and the
instruction has to be *keep your head still and look with your eyes only*.
`--calibration-head still` models that operator; the default is natural
eye–head coordination.

### Mounting it with the articulated suction arm

The demonstration mount is a 4.5 in (114 mm) pump suction cup with a
dual-ball-head arm. The D435i's 1/4-20 thread goes on the arm's 1/4 in
mount, and at 72 g the camera is far inside the mount's rating.

* **Cup:** on the housing's flat stainless front face, its centre about
  3 in in from the left edge and 3 in up from the underside. That keeps the
  cup's 2¼ in radius on flat metal, clear of the lower-left chamfer and the
  edge. The cup goes on clean, dry, degreased metal; the pump's indicator
  shows when it is losing vacuum.
* **Arm:** reaches about 3 in left, 2 in down and 1 in out from the cup to
  put the camera at the recommended position, well within its reach.
* **Check it is clear:** with the arm locked, lower the blade guide all the
  way and slide the tallest product the saw takes under the housing. The
  camera must stay clear of both.
* **Tether it.** A suction cup on a vibrating machine, above an open blade
  and food product, should not be the only thing holding the camera. Run a
  short lanyard from the camera or arm to a fixed point on the housing. Check
  the vacuum indicator at each start-up, and after a washdown, which a cup
  will not survive reliably.

**How much the mount may move after calibration**, from
`python -m tools.operator_sim drift`: 90 s of `shift`, commissioned with the
head still, perfect landmarks so that only the camera changes. The two gaze
models behave differently here, and it matters for the mount:

| Mount moves after calibration by | Eye model: zone shift (h, v) | Head model: zone shift (h, v) |
| --- | --- | --- |
| tips down 1° | 0.0°, 0.0° | 0.0°, +1.2° |
| tips down 3° | 0.0°, 0.0° | +0.1°, +3.7° |
| turns 3° | −0.1°, 0.0° | −2.0°, −0.1° |
| rolls 5° | 0.0°, 0.0° | +2.5°, −0.5° |
| sags 15 mm | +0.2°, −0.2° | +0.1°, 0.0° |
| sags 30 mm | +0.1°, −0.8° | 0.0°, −0.5° |
| slides 15 mm sideways | +0.2°, 0.0° | 0.0°, +0.1° |

(The shift is the change in the mean gaze error's bias, against the mount
not moving.)

* **Eye model:** turning the camera changes nothing. The measurement is the
  iris's position relative to the eye corners, and turning the camera moves
  both together.
* **Head model:** gaze is measured relative to the camera, so a camera that
  tips, turns or rolls after calibration shifts the whole zone by about
  that angle. A sag or slide that does not rotate it barely matters.

So with the head model, **recalibrate after anything that rotates the
camera by more than a degree or so**: a knock, a re-aim, a ball head that
creeps. The D435i has an IMU that measures gravity, and folding it in could
cancel tip and roll automatically; that is a candidate next improvement.
With either model, a mount that tips further than the frame's 5–7° of margin
loses the operator's chin or forehead, and FaceMesh loses the face.

The **attention zone** runs from the bottom of the upper blade guide down the
blade to the table, and across the strip of table in front of the teeth:
24 cm wide and tilted towards the operator. Seen from the eyes it is about
21° × 18° across its middle, and much wider at its near edge, whose corners
are up to 27° to either side.

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
while calibrating, which the eye model needs (see
[Commissioning and head movement](#commissioning-and-head-movement)). The
default is natural eye–head coordination, which turns the head for the
wider shifts.

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

One seed per scenario (seed 0), default parameters, the `default`
appearance, the recommended mount and the realistic stance. Each run is
scored twice, on FaceMesh's landmarks and on perfect ones. Treat these as
what to look at, not as results.

| | `shift` 60 s, head-still commissioning | | `shift` 60 s, natural commissioning | | `vor` 20 s, head-still commissioning | |
| --- | --- | --- | --- | --- | --- | --- |
| Landmarks | FaceMesh | perfect | FaceMesh | perfect | FaceMesh | perfect |
| Eye-in-head residual, yaw / pitch | 3.4° / 5.8° | 0.8° / 0.8° | 4.1° / 4.3° | 0.9° / 0.7° | 3.8° / 4.4° | 0.5° / 0.7° |
| Pitch correlation with the true eye elevation | 0.32 | 0.99 | 0.25 | 0.99 | 0.71 | 0.99 |
| Zone position, median error | 5.5° | 4.3° | 16.4° | 9.3° | 8.3° | 8.5° |
| In-zone agreement | 75% | 87% | 43% | 74% | 79% | 100% |
| True warnings detected | 2 of 3 | 2 of 3 | 3 of 3 | 3 of 3 | none to detect | none to detect |
| False warnings per minute | 0.9 | 0 | 3.7 | 2.8 | 0 | 0 |

Landmark error against the rendered truth was about the same in all three
runs: iris centres 1.5 px median, eye corners 5.8–6.6 px, all 468 points
4.9–5.4 px. The face was found in every frame.

In `vor` the line of sight never leaves the zone. With the operator this
close, the zone is wide enough (about 21° across) that the 15° head sweep
stays inside it, so there are no false warnings. The 8.5° zone error with
perfect landmarks is the eye counter-rotating in the head.

The warning missed in both head-still runs is a 0.8 s look at the product,
which sits just outside the zone's edge; with 4–5° of zone error a target
that close to the boundary is ambiguous. FaceMesh's one false warning in
that run is a single 12 s stretch of watching the cut read as out of zone.

* **The geometry checks out.** With perfect landmarks, `(dx, dy)` track the
  true eye rotation to under 1° in both axes, in every run. The truth, the
  renderer and the replay agree with each other, and `tests/` pins this.
* **Commissioning with the head still matters as much as the detector.**
  Natural head movement during commissioning triples the median zone error
  with FaceMesh (5.5° to 16.4°) and quadruples the false warnings. It costs
  more than FaceMesh's own error does, on the same run.
* **FaceMesh is the larger error, and vertically it is nearly blind** on
  these frames: an eye-in-head residual of 3.4–4.1° in yaw and 4.3–5.8° in
  pitch, with pitch correlating only 0.25–0.7 with the true eye elevation
  against 0.99 for perfect landmarks. Most of the yaw error comes from the
  **eye corners** (about 6 px median error), not the iris (about 1.5 px).
  Whether a real face does better is the open question: compare `landmarks`
  and `eye_in_head` with `tools/gaze_noise.py` captures from the hardware.
* **Eye-in-head is not gaze.** The pipeline measures the eye in the head,
  so when the head turns with gaze held, the reading moves. In `vor` that
  costs 8.5° of zone error even with perfect landmarks, and after a large
  head turn it takes a couple of seconds of the head settling for the
  reading to come back. The same mechanism is why commissioning needs the
  head still. Head pose is the obvious next input, and the simulator's
  `head_*_deg` truth is there to fit it against.

## Benchmarking

One simulated run is not evidence: where the four-corner calibration happens
to land moves the result more than most changes do. `benchmark` scores the
pipeline on a fixed, versioned suite of cases that varies what it has to be
robust to, and pools the result into a scorecard with 90% intervals from
resampling the cases:

* the operator's looks and height (the four appearance presets);
* how they move their head: `typical`, a `head_mover` who turns the head
  for small shifts, and an `eye_mover` who barely does;
* how they commission: head still, as the checklist asks, or naturally;
* the camera's exact mount: up to 15 mm and 3° from the recommended one;
* the scenario, including two stress cases: `edges`, fixations just inside
  and just outside each edge of the zone, and `flicker`, rapid glances away
  that are each shorter than the grace period;
* the random seed.

The scorecard leads with what matters at the saw: **missed alarms** (real
excursions that never escalated) and **false alarms per hour**, then
latency, time to clear, tracking, and the gaze error behind them, broken
down by scenario, operator type, commissioning, appearance and mount.

```bash
python -m tools.operator_sim benchmark --suite quick              # 30 s, perfect landmarks
python -m tools.operator_sim benchmark --suite standard --out new.json
python -m tools.operator_sim benchmark --suite standard \
    --baseline tools/operator_sim/baselines/standard-truth.json   # paired comparison
python -m tools.operator_sim benchmark --suite standard --set filter_alpha=0.2
python -m tools.operator_sim benchmark --suite facemesh --landmarks mediapipe
```

| Suite | Cases | Landmarks | Time |
| --- | --- | --- | --- |
| `quick` | 8 | perfect | about 30 s |
| `standard` | 31 | perfect | about 2 min |
| `facemesh` | 8 | FaceMesh on rendered frames | about an hour the first time, cached after |

The perfect-landmark tier tests the gaze model, calibration and policy with
the detector taken out. The FaceMesh tier renders each case once into
`--cache`, and every later run replays the cached landmarks, so a change
downstream of FaceMesh costs seconds to evaluate there too.

**Comparisons are paired.** `--baseline` scores the same cases and reports,
for each metric, the mean per-case difference and its 90% interval. Only a
difference whose interval excludes zero (marked `*`) is worth believing.
The baselines in [`baselines/`](baselines/) are the pipeline as it is on
`main`, with the default head model (`*-truth.json`), and the eye model for
comparison (`*-truth-eye.json`). Re-running unchanged code reproduces them
exactly.

The benchmark can see a change, and that conclusions depend on the model.
A faster filter (`--set filter_alpha=0.3` against the default 0.12) added
33 false warnings an hour [+10, +60] on the eye model. On the head model the
same change slightly *helps*: in-zone agreement +2.2 points [+1.3, +3.2] and
gaze error −0.27° [−0.37, −0.19], with no significant change in alarms. So
re-tune after changing the model, never carry settings across.

### Results, with perfect landmarks

`standard` suite, 31 cases. The eye model is the iris's offset within the
eye; the head model adds head pose from the depth camera:

| | Eye model | Head model |
| --- | --- | --- |
| Missed warnings | 16 of 72 (22%) | **6 of 80 (7.5%, interval 4–12%)** |
| Missed alerts | 1 of 6 | **0 of 7** |
| False warnings per hour | 87 | **32 (17–49)** |
| In-zone agreement | 80% | **92%** |
| Gaze error, median | 9.0° | **3.9°** |
| Did not calibrate | 2 of 31 | **0 of 31** |

Paired over the 29 cases both calibrated, the head model has fewer missed
warnings (−0.38 per case [−0.76, −0.10]), fewer false warnings (−61 an hour
[−111, −17]) and fewer false alerts (−32 an hour [−64, −3]).

| Head model, by operator | Missed warnings | False warnings/hour |
| --- | --- | --- |
| `eye_mover` | 1 of 22 | 11 |
| `typical` | 2 of 35 | 25 |
| `head_mover` | 3 of 23 | 59 |

| Head model, by scenario | Missed warnings | False warnings/hour |
| --- | --- | --- |
| `edges` | 2 of 17 | 0 |
| `flicker` | none to miss | 0 |
| `glances` | 0 of 20 | 0 |
| `shift` | 4 of 43 | 41 |
| `distance` | none to miss | 90 |
| `vor` | none to miss | 0 |

All of this is with a perfect landmark detector. What the head model fixed,
and what is left:

* **Operators who look away with their head are no longer missed.** On the
  eye model, a head mover's excursions went unnoticed half the time (11 of
  23), with 134 false warnings an hour, because turning the head with the
  eyes centred barely changes eye-in-head. On the head model, 3 of 23 and
  59 an hour.
* **Rapid short glances no longer pile up into false alarms.** `flicker`
  went from about 350 false warnings an hour to none.
* **Commissioning no longer depends on holding the head still.** Both
  calibration failures on the eye model were naturally commissioned cases.
* **Still to fix:** false warnings in the `shift`, mostly head movers (59
  an hour), and in `distance` (90 an hour), where stepping back and forth
  is still read as gaze moving. The distance scaling is a rough
  approximation, and is the next thing to look at.

### Results, with MediaPipe landmarks

The `facemesh` suite, 8 cases rendered and run through MediaPipe, reverses
that. Commissioned with the head still:

| | Eye model | Head model |
| --- | --- | --- |
| Missed warnings | 3 of 21 | 0 of 21 |
| False alerts per hour | 111 | 74 |
| In-zone agreement | **77%** | 55% |
| Gaze error, median | **9.5°** | 20.6° |
| Time for an alarm to clear, median | **0.9 s** | 5.2 s |

The head model's lack of misses is not a strength: it reads "out of zone"
so much of the time that nothing gets missed, and its alarms take seconds
to clear. `facemesh-natural`, the same cases commissioned with natural head
movement, is worse for both models (in-zone agreement 45% and 42%).

The cause is MediaPipe's iris, not the head pose:

* Head pose from MediaPipe's landmarks is good: against the pose from
  perfect landmarks, a gain of 0.9–0.97 on each axis and about 1° of noise.
* The iris offset is not. Against the true eye rotation it moves with about
  half the expected gain horizontally and about **8%** vertically, with
  noise worth about 25° of vertical eye rotation.
* A zone 16° tall is taken mostly with the eyes, whether or not the head is
  held still, because people do not turn their heads for shifts that small.
  So the head model's calibrated zone comes out 2–4° tall where it should
  be about 16°. The eye model escapes this because its calibration rescales
  to whatever the iris reports.
* Correcting the iris's gain from the vestibulo-ocular reflex (eyes
  counter-rotating while the head moves) was tried and does not work: the
  right correction ranges from 0.4 to 2.2 between faces and lighting.

Whether real eyes do better than rendered ones is the open question, and
only a recording from the real camera can answer it. Until then `eye` is
the default. Baselines: `facemesh-mediapipe*.json`,
`facemesh-natural-mediapipe*.json`.

## The promo video

```bash
python -m tools.operator_sim promo --out docs/media/operator-promo.mp4
```

renders [`docs/media/operator-promo.mp4`](../../docs/media/operator-promo.mp4)
(1080p, 30 s), a copy sized for a text message beside it (`-sms.mp4`, 640×360
and under 1 MB, which is about the most carriers pass as MMS), and a poster
frame. It takes about 15 minutes on four cores. `--keep DIR` saves the
intermediate renders and reuses them, so a change to the captions or
overlays re-encodes in a minute or two.

The shot is a 3/4 view of the operator at the saw. The camera that does the
monitoring is not drawn, since in production it is built into the saw head,
but what it sees is: the inset is its frame, with FaceMesh's landmarks.

**The status badge is not scripted.** The face camera's frames are rendered,
FaceMesh runs on them, and the package's own measurer, calibrator and state
machine decide the state, exactly as `evaluate` replays them. The script
(watch the cut, glance at the tray for half a second, turn to talk to a
coworker, go back to the cut) is chosen to show the grace period, the
warning and alert thresholds, and the alarm clearing. The operator
commissions with their head still, as recommended above. The video is cut
from one continuous run and is not trimmed: what the pipeline did is what it
shows.

The saw is drawn from the general-arrangement drawing and the Hi-Yield 14's
published features (peaked upper housing, stainless doors), with no markings
or logo.

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
