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

### Commission with the head still

From where the operator stands, the zone's bottom corners are as much as 27°
to either side of their eyes. Asked to "look at each corner", most people
turn their head for shifts that size, and the pipeline measures the eye
*in the head*: a head turned during commissioning shrinks the calibrated
zone. With perfect landmarks, so that only this changes:

| Commissioning | Scenario | Gaze error, median | In-zone agreement |
| --- | --- | --- | --- |
| Head held still, eyes only | `eyes-only` | 2.7–3.4° | 89–97% |
| Natural head movement | `eyes-only` | 20° | 57% |
| Head held still | `shift` | 2.6–4.0° | 97–98% |
| Natural head movement | `shift` | 12.5° | 59% |

Two seeds each, 40 s. With natural head movement, one of the two seeds
never calibrated at all: the calibrator rejected the corners as "too much
movement", because the head was still turning while the operator held the
fixation. The rows give the seed that did calibrate.

So the instruction at commissioning should be *keep your head still and
look at each corner with your eyes*. `--calibration-head still` models that
operator; the default is natural eye–head coordination.

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
head still, perfect landmarks so that only the camera changes:

| Mount moves after calibration by | Gaze error, median | In-zone agreement | False warnings/min |
| --- | --- | --- | --- |
| nothing | 5.40° | 77.2% | 1.28 |
| tips down 3° | 5.41° | 77.2% | 1.28 |
| turns 3° | 5.34° | 77.4% | 1.28 |
| rolls 5° | 5.40° | 77.2% | 1.28 |
| sags 15 mm | 5.46° | 81.0% | 0.64 |
| sags 30 mm | 5.56° | 87.2% | 0.64 |
| slides 15 mm sideways | 5.35° | 75.3% | 1.28 |

Turning the camera changes nothing measurable. The gaze measurement is the
iris's position relative to the eye corners, and turning the camera moves
both together. Moving it changes the viewpoint slightly, which moves
individual frames across the zone's edge, but not the error. So a ball head
that creeps does not quietly invalidate the calibration. What does limit it
is the frame: with only 5–7° of margin, a mount that tips further than that
loses the operator's chin or forehead, and FaceMesh loses the face.
Recalibrate after anyone re-aims the arm. A small droop does not need it.

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
while calibrating, as they should (see above). The default is natural
eye–head coordination, which turns the head for the wider shifts.

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

One seed per scenario, default parameters, `default` appearance, natural
calibration, and the *previous* camera position. The recommended mount does
better with FaceMesh in the loop; see the placement table above. Treat them
as what to look at, not as results.

| | `shift`, 60 s | | `vor`, 20 s | |
| --- | --- | --- | --- | --- |
| Landmarks | FaceMesh | perfect | FaceMesh | perfect |
| Eye-in-head residual, yaw / pitch | 3.3° / 4.9° | 1.2° / 1.0° | 4.0° / 5.6° | 0.4° / 0.5° |
| Zone position, median error | 12.4° | 4.5° | 9.3° | 14.9° |
| In-zone agreement | 54% | 90% | 72% | 37% |
| True warnings detected | 5 of 5 | 5 of 5 | none to detect | none to detect |
| False warnings per minute | 4.7 | 0 | 6.0 | 24 |

In `vor` the gaze never leaves the zone, so every warning there is false.
FaceMesh scores *better* than perfect landmarks there only because its noise
happens to land some frames back inside the zone.

* **The geometry checks out.** With perfect landmarks, a still-headed
  calibration and head-still eye movements, `(dx, dy)` track the true eye
  rotation to about 0.5° in both axes. The calibrated zone position comes in
  at a 2–2.5° median error, mostly EMA lag around saccades. The truth, the
  renderer and the replay agree with each other, and `tests/` pins this.
* **FaceMesh is the larger error, and vertically it is nearly blind** on
  these frames. On the same frames, MediaPipe's landmarks give an
  eye-in-head residual of about 3.5–5° in yaw and 5° in pitch, with pitch
  correlating only 0.25–0.6 with the true eye elevation. Most of the yaw
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
