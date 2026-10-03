# A wider field of view, and body pose

Status: tabled. Nothing here is built. Recorded because the reasoning took a
while to arrive at and the measurements were made on the actual hardware, so
rediscovering them later would be wasted effort.

The question was whether this device could watch more than the operator's
eyes: head, torso, arms and hands, classifying dangerous working behaviour
rather than only inattention. And if so, whether that wants a vision language
model or something smaller.

## What the hardware has spare

Measured on the Jetson Orin Nano Super with the monitor running normally at
30.0 fps and no dropped frames:

| | in use | spare |
| --- | --- | --- |
| CPU | ~8% of 6 cores | ~5.5 cores |
| GPU | 0 to 5% | effectively all |
| RAM | 2.7 GB of 7.6 | 4.9 GB |
| Power | 6.6 W | ~18 W of budget |

MediaPipe on aarch64 runs through XNNPACK on the CPU, so the GPU is idle
while the monitor works. Compute is not the constraint.

## Why latency decides this, and not compute

A band saw blade runs somewhere around 1000 to 1400 m/min, so 17 to 23 m/s. A
hand moves at 1 to 2 m/s deliberately and can exceed 5 m/s on a startle. With
a hazard distance of 10 to 20 cm that is 50 to 100 ms of warning.

A 3B-class vision language model on this board, quantised to int4, is
realistically 0.5 to 1.5 s per frame for a short answer. That is ten to
thirty times too slow, and a smaller model does not close the gap.

Nor does anything else. Add the saw's own spin-down, which for a braked band
saw is a second or more, and no camera system prevents hand-to-blade contact.
That is what guards and interlocks are for. A camera's value is upstream, in
changing the behaviour that eventually produces the injury, which is what the
existing attention monitor already does with a three second `brake_after`.
Any body-pose work belongs in the same supervisory role. Presenting it as
reflex protection would be selling something the physics does not support.

There is a second problem independent of speed. Vision language models are
weak at precise spatial relations. Ask one whether a person is operating a
band saw and it is reliable. Ask whether the left hand is within 15 cm of the
blade and it is not. Geometry on landmarks is good at exactly that, and it is
the only approach with a number attached to its answer.

## The depth camera is the asset here

A danger zone defined in the image is a polygon, and a hand a metre behind
the blade that happens to overlap it in 2D is a false stop. With depth the
zone is a volume and that whole class of false positive disappears. A
pose-plus-depth geometric check is both faster and more trustworthy than any
semantic model, and the sensor is already on the machine.

## Tiers, if this is ever picked up

1. **Geometry, 30 Hz, no learning.** MediaPipe Pose and Hands, landmarks
   projected into the depth frame, a calibrated keep-out volume around the
   blade, hysteresis over a window. The same shape as `away_seconds` and
   `brake_after` in `attention.py`. This is the only tier that may go near
   the permit.

2. **Pose-sequence classifier, 10 to 15 Hz.** A window of normalised
   landmarks, say 2 s at 15 fps, into a small temporal CNN or an ST-GCN.
   100k to 1M parameters, under a millisecond of inference. The compute is
   free; the landmark extraction in tier 1 is the entire cost. This is where
   reaching across the blade, clearing offcuts by hand while running, and
   working without a push stick would live.

3. **Vision language model, 0.2 Hz, advisory only.** Never in the permit
   path. Triggered only on frames the lower tiers flag as odd but not clearly
   dangerous. A 3B model at int4 fits the spare RAM. Its output annotates the
   incident log and raises a supervisor alert.

The architectural line is the one the code already draws. `safety.py` renews
a permit every 100 ms and goes silent when the vision result is older than
500 ms. Nothing with multi-second, non-deterministic latency can meet that
contract, so a model of that kind belongs beside `machine.py`, in the
read-only half of the system.

## Pixels on target, measured

From the D435i's own colour intrinsics rather than the datasheet, because the
sensor is 16:9 and a 4:3 mode is produced by cropping the sides:

| Mode | Horizontal FOV | At 0.65 m | face (15 cm) | iris (11.7 mm) | hand (9 cm) |
| --- | --- | --- | --- | --- | --- |
| 640x480 | 55.7 deg | 69 cm wide | 140 px | 11 px | 84 px |
| 1280x720 | 70.4 deg | 92 cm wide | 209 px | 16 px | 126 px |

Vertical field is 43.3 degrees in both.

This is the tension that makes the idea awkward on one sensor. Widening the
field to take in arms and hands shrinks the face, and the face is where the
gaze measurement comes from. Moving to 1280x720 was worth doing on its own
merits and has been done, but it does not resolve the conflict so much as buy
a little room: a hand at 126 px is at the edge of what MediaPipe Hands wants
for finger landmarks, and that is at 0.65 m with the operator close in.

If one sensor turns out not to serve both, the answer is two: one wide for
pose, one tight on the blade zone for hands. More plumbing, no new ideas.

## What would actually block tier 2

Labelled examples of dangerous behaviour, not compute. They are rare and the
interesting ones cannot be staged safely.

Two things make it tractable. Pose landmarks barely change according to
whether the blade is moving, so dangerous configurations can be staged with
the machine locked out and the classifier cannot tell the difference. It sees
skeletons. That covers most of a training set safely.

And this is the one genuinely good use for a vision language model here, run
offline rather than on the device: pass recorded footage through it to propose
labels for a human to confirm. That buys the semantic breadth without putting
its latency or its spatial sloppiness anywhere near a control decision.

Mind the base rate when it comes to that. At 99% per-frame specificity on a
30 fps stream, a per-frame decision produces a false stop every three
seconds. Window plus hysteresis is the fix, and `attention.py` already has
the pattern.

## The one experiment worth running first

Add MediaPipe Pose and Hands alongside the existing FaceMesh at 1280x720 and
log landmark confidence and timing, wired to nothing. That answers the only
genuinely uncertain question, which is whether one camera can serve both face
and hands at the working distance. The model choices after that are easy.
