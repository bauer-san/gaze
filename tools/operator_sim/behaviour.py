"""How the operator moves: where they look, and how head, eyes and body get there.

The controlled variable is gaze -- the line of sight in the world -- because
that is what people control. The head and eyes are two effectors that share
the job, and most of what makes the motion realistic is how they share it:

* **Saccades** move gaze between targets in a few tens of milliseconds, on
  the main sequence (duration grows linearly with amplitude), with the skewed
  velocity profile real saccades have.
* **Eye-head coordination.** Small shifts are made with the eyes alone. For
  larger ones the head joins in, starting a little after the eyes and moving
  an order of magnitude more slowly, and it does not go all the way: the
  eyes are left some way off-centre in the head. How far the head goes is a
  per-operator trait.
* **The vestibulo-ocular reflex.** Once gaze lands, it stays on the target
  while the head is still moving, so the eyes counter-rotate in the head.
  This is the case where eye-in-head and gaze come apart, and it is the one
  a head-mounted gaze model gets wrong.
* **Vergence.** Each eye is aimed at the fixated point itself, so the two
  eyes converge on a near target.
* **Fixational eye movements**: slow drift and occasional microsaccades.
* **Blinks**: spontaneous, at about 15 to 20 a minute, plus blinks that
  accompany large gaze shifts, with the fast-close, slow-open lid profile.
* **Posture**: sway, breathing, forward lean over the table, steps.

A scenario is a list of :class:`Segment`, each one "look at this, like
this, for this long". The stochastic ``shift`` scenario generates them from
a model of the cutting task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .mathutil import (
    OUNoise,
    direction_from_yaw_pitch,
    head_movement_duration,
    min_jerk,
    normalize,
    rot_x,
    rot_y,
    rot_z,
    saccade_duration,
    saccade_profile,
    yaw_pitch,
)
from .scene import Scene

# The head frame looks along +z; the operator faces -z in the world.
R_BASE = rot_y(math.pi)

# Pivot of the head on the neck (roughly the atlanto-occipital joint), in the
# head frame: below and behind the canonical model's origin.
HEAD_PIVOT = np.array([0.0, -0.072, -0.035])

# Oculomotor range. People rarely use more than about 30 degrees; these are
# the hard stops the eye cannot pass.
EYE_YAW_LIMIT = math.radians(42.0)
EYE_UP_LIMIT = math.radians(32.0)
EYE_DOWN_LIMIT = math.radians(48.0)
# Where the operator stands, (x, z) of the feet: a little left of the blade,
# with the front of their body a hand's width behind the carriage's hand
# rail, leaning over the table to the work. Close in, because that is where
# the product is: the cut is about 0.35 m beyond the table's front edge.
STANCE = (-0.04, 0.64)

# Neck rotation range; beyond it the torso has to turn.
HEAD_YAW_LIMIT = math.radians(75.0)


@dataclass
class Segment:
    """One instruction: look at ``target`` for ``duration`` seconds.

    ``target`` is a name from :meth:`Scene.targets`, ``"zone:u,v"`` for a
    point in the attention zone, or an explicit world point.

    ``head`` is how the head takes part: ``natural`` (the default eye-head
    coordination), ``still`` (eyes only; the head holds where it is) or
    ``face`` (turn the head to face the target).

    ``scan`` is the radius in metres of small re-fixations around the target,
    which is how people actually watch an extended thing like a cut.
    """

    target: str | tuple
    duration: float
    head: str = "natural"
    scan: float = 0.0
    stance: tuple | None = None
    hands: str = "work"
    activity: str = ""
    calib_corner: int | None = None
    # Overrides of the head's own motion, for the roll and head-sweep tests.
    head_roll_deg: float | None = None
    head_sweep_deg: tuple | None = None  # (yaw amplitude, pitch amplitude, period)
    lean_deg: float | None = None


@dataclass(frozen=True)
class Traits:
    """Individual differences in how an operator moves."""

    # Head contribution grows linearly with the part of a gaze shift beyond
    # the threshold: head = head_gain * (shift - threshold). Some people are
    # "head movers" (high gain, low threshold), others move mostly their eyes.
    head_gain: float = 0.75
    # Gaze shifts smaller than this are made with the eyes alone. Most people
    # make shifts under about 15-20 degrees with the eyes alone; lower it to
    # model a "head mover".
    head_threshold_deg: float = 15.0
    # However large the shift, the head goes far enough that the eyes end up
    # at most this far off-centre in the head.
    max_eccentricity_deg: float = 20.0
    # During a held fixation the head carries on round to re-centre the
    # eyes once they have been more than ``recentre_deg`` off-centre for a
    # moment, until they are ``recentre_to_deg`` off. VOR holds gaze
    # meanwhile. After a big turn back to the work, this is the head
    # catching up with the eyes.
    #
    # This matters more than it looks. An eye-in-head measurement cannot
    # tell a re-centred head from a change of gaze, and an operator who
    # re-centres on each corner during commissioning calibrates a zone that
    # is far too small. Lower recentre_deg to model that operator.
    recentre_deg: float = 15.0
    recentre_to_deg: float = 2.0
    # People looking down keep their eyes lowered in the head rather than
    # bowing the head all the way: the comfortable split.
    eye_down_bias_deg: float = 12.0
    blink_rate_per_min: float = 17.0
    lean_deg: float = 8.0
    sway_m: float = 0.008
    head_jitter_deg: float = 0.6
    roll_bias_deg: float = 3.0
    reaction_s: float = 0.18


@dataclass
class State:
    """Everything needed to pose and render the operator at one instant."""

    t: float
    head_R: np.ndarray  # head frame -> world
    head_pivot: np.ndarray  # world
    neck_base: np.ndarray
    hips: np.ndarray
    stance: np.ndarray  # (x, z) of the feet, without sway
    carriage: float  # carriage travel towards the blade, metres
    torso_R: np.ndarray
    fixation: np.ndarray  # world point both eyes are aimed at
    cyclopean: np.ndarray  # world, midpoint of the eyes
    eye_dirs_head: np.ndarray  # (2, 3) visual axes in the head frame
    eye_centres_world: np.ndarray  # (2, 3)
    blink: float
    pupil_radius: float
    hands: np.ndarray  # (2, 3) world, left then right
    segment: Segment
    saccade: bool
    head_angles: tuple  # yaw, pitch, roll in radians (positive pitch is down)
    eye_angles: np.ndarray  # (2, 2) yaw, pitch of each eye in the head (rad)


class Behaviour:
    """Steps the operator through a scenario at a fixed frame rate."""

    def __init__(
        self,
        scene: Scene,
        segments: list[Segment],
        eye_centres_head: np.ndarray,
        stature: float = 1.75,
        traits: Traits | None = None,
        seed: int = 0,
        stance: tuple = STANCE,
    ):
        self.scene = scene
        self.targets = scene.targets()
        self.segments = list(segments)
        self.eye_centres_head = np.asarray(eye_centres_head, float)
        self.stature = stature
        self.traits = traits or Traits()
        self.rng = np.random.default_rng(seed)
        r = self.rng
        tr = self.traits

        self.t = 0.0
        self.seg_index = 0
        self.seg_start = 0.0
        self.total = sum(s.duration for s in self.segments)

        # Posture.
        self.stance = np.array(stance, float)
        self.step = None  # (from, to, t0, dur)
        self.sway = OUNoise(r, tr.sway_m, 1.8, dim=2)
        self.lean_noise = OUNoise(r, math.radians(1.5), 3.0)
        self.lean = math.radians(tr.lean_deg)
        self.torso_yaw = 0.0

        # Head.
        self.head = np.array([0.0, math.radians(35.0), 0.0])  # yaw, pitch, roll
        self.head_plan = None  # (from, to, t0, dur)
        self.head_jitter = OUNoise(r, math.radians(tr.head_jitter_deg), 0.35, dim=3)
        self.roll_drift = OUNoise(r, math.radians(tr.roll_bias_deg), 6.0)

        # Gaze.
        self.gaze_target = self._resolve(
            self.segments[0].target if self.segments else "cut"
        )
        self.gaze_point = self.gaze_target.copy()
        self.saccade_plan = None  # (start point, end point, t0, dur)
        self.pending = None  # (t_start, target point, head mode)
        self.drift = OUNoise(r, math.radians(0.08), 0.6, dim=2)
        self.micro_offset = np.zeros(2)
        self.next_micro = self._exp(0.8)
        self.next_scan = None
        self._fixation_start = 0.0
        self._recentre_wait = 1.0

        # Blinks.
        self.blink_t0 = None
        self.blink_depth = 1.0
        self.next_blink = self._exp(60.0 / tr.blink_rate_per_min)
        self.pupil = OUNoise(r, 0.00015, 1.5)

        # Hands.
        self.hand_pos, self.hand_on_carriage = self._hand_targets("work")
        self.hand_plan = None
        # Carriage travel towards the blade, metres, and where in the
        # cutting stroke the operator is, seconds.
        self.carriage = 0.0
        self._stroke = 0.0
        self._start_segment(0, first=True)

    # -- helpers ---------------------------------------------------------------

    def _exp(self, mean: float) -> float:
        return float(self.rng.exponential(mean))

    def _resolve(self, target) -> np.ndarray:
        if isinstance(target, str):
            if target.startswith("zone:"):
                u, v = (float(x) for x in target[5:].split(","))
                return self.scene.zone.point(u, v)
            return self.targets[target].copy()
        return np.asarray(target, float)

    def _hand_targets(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Wrist positions for a mode, with the carriage at rest, and which
        hands are holding the carriage (and so travel with it).

        Wrists, not palms: the palm sits about 75 mm beyond the wrist along
        the forearm, so these put the palms on the grips. Both working hands
        stay behind the pusher plate: the right drives the plate's grip,
        which pushes the product through the blade, and the left holds the
        carriage's hand rail. Neither reaches over the plate.
        """
        rail_left = [-0.38, 1.03, 0.48]
        rail_right = [-0.10, 1.03, 0.48]
        pusher = [-0.13, 1.12, 0.26]
        if mode == "tray":
            return np.array([rail_left, [0.62, 0.97, 0.30]]), np.array([1.0, 0.0])
        if mode == "rest":
            return np.array([rail_left, rail_right]), np.array([1.0, 1.0])
        return np.array([rail_left, pusher]), np.array([1.0, 1.0])

    @property
    def segment(self) -> Segment:
        return self.segments[min(self.seg_index, len(self.segments) - 1)]

    @property
    def finished(self) -> bool:
        return self.seg_index >= len(self.segments)

    def _start_segment(self, index: int, first: bool = False) -> None:
        self.seg_index = index
        self.seg_start = self.t
        if self.finished:
            return
        seg = self.segments[index]
        target = self._resolve(seg.target)
        if first:
            self.gaze_target = target
            self.gaze_point = target.copy()
            yaw, pitch = self._target_angles(target)
            self.head[0] = yaw
            self.head[1] = pitch - math.radians(self.traits.eye_down_bias_deg)
        else:
            self.pending = (
                self.t + self.traits.reaction_s * self.rng.uniform(0.7, 1.3),
                target,
            )
        if seg.stance is not None:
            to = np.array(seg.stance, float)
            if np.linalg.norm(to - self.stance) > 1e-3:
                self.step = (
                    self.stance.copy(),
                    to,
                    self.t + 0.1,
                    0.5 + 1.2 * np.linalg.norm(to - self.stance),
                )
        hands, on = self._hand_targets(seg.hands)
        self.hand_plan = (
            (self.hand_pos.copy(), self.hand_on_carriage.copy()),
            (hands, on),
            self.t + 0.1,
            0.6,
        )
        self.next_scan = self.t + 0.4 if seg.scan > 0 else None

    # -- kinematics --------------------------------------------------------------

    def _body(self):
        """Hips, torso rotation, neck base and head pivot in the world."""
        s = self.stature
        stance = self.stance + self.sway.state
        hips = np.array([stance[0], 0.53 * s, stance[1]])
        seg = self.segment
        lean = (
            math.radians(seg.lean_deg) if seg.lean_deg is not None else self.lean
        ) + float(self.lean_noise.state[0])
        torso_R = rot_y(self.torso_yaw) @ rot_x(-lean)
        breathe = 0.003 * math.sin(2 * math.pi * 0.25 * self.t)
        neck_base = hips + torso_R @ np.array([0.0, 0.30 * s + breathe, 0.0])
        # The neck leans forward more than the torso when looking down.
        neck_R = rot_y(self.torso_yaw) @ rot_x(-(lean + 0.5 * max(self.head[1], 0.0)))
        pivot = neck_base + neck_R @ np.array([0.0, 0.10 * s / 1.75, 0.0])
        return hips, torso_R, neck_base, pivot

    def _head_R(self, angles) -> np.ndarray:
        yaw, pitch, roll = angles
        return R_BASE @ rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)

    def _eye_world(self, head_R, pivot) -> np.ndarray:
        origin = pivot - head_R @ HEAD_PIVOT
        return self.eye_centres_head @ head_R.T + origin

    def _target_angles(self, point) -> tuple[float, float]:
        """Yaw and pitch of ``point`` from the eyes, in the head's base frame."""
        _, _, _, pivot = self._body()
        eyes = self._eye_world(self._head_R(self.head), pivot).mean(axis=0)
        d = R_BASE.T @ normalize(point - eyes)
        return yaw_pitch(d)

    # -- control -----------------------------------------------------------------

    def _plan_gaze_shift(
        self, target: np.ndarray, mode: str, scan: bool = False
    ) -> None:
        """Start a saccade to ``target`` and, if needed, a head movement.

        ``scan`` marks a small re-fixation within the same target, which
        does not restart the head's settling: re-centring waits on the last
        real change of target, not on every glance across the cut.
        """
        _, _, _, pivot = self._body()
        head_R = self._head_R(self.head)
        cyc = self._eye_world(head_R, pivot).mean(axis=0)
        g0 = normalize(self.gaze_point - cyc)
        g1 = normalize(target - cyc)
        amp = math.degrees(math.acos(float(np.clip(g0 @ g1, -1, 1))))
        self.saccade_plan = (
            self.gaze_point.copy(),
            target.copy(),
            self.t,
            saccade_duration(amp),
        )
        if not scan:
            self._fixation_start = self.t
            self._recentre_wait = float(self.rng.uniform(0.5, 1.2))
        self.gaze_target = target
        self.micro_offset[:] = 0.0

        tr = self.traits
        yaw_t, pitch_t = yaw_pitch(R_BASE.T @ g1)
        new = self.head.copy()
        if mode == "face":
            new[0] = yaw_t
            new[1] = pitch_t - math.radians(4.0)
        elif mode == "natural":
            new = self._head_goal(yaw_t, pitch_t, tr.head_threshold_deg)
        new[0] = float(np.clip(new[0], -HEAD_YAW_LIMIT, HEAD_YAW_LIMIT))
        if mode != "still" and np.max(np.abs(new[:2] - self.head[:2])) > math.radians(
            0.5
        ):
            amp_h = math.degrees(float(np.max(np.abs(new[:2] - self.head[:2]))))
            # The head starts 20-50 ms after the eyes.
            self.head_plan = (
                self.head.copy(),
                new,
                self.t + self.rng.uniform(0.02, 0.05),
                head_movement_duration(amp_h) * self.rng.uniform(0.9, 1.15),
            )
        # Large gaze shifts are often accompanied by a blink.
        if amp > 25 and self.blink_t0 is None and self.rng.random() < 0.4:
            self._blink(partial=False)

    def _head_goal(
        self, yaw_t: float, pitch_t: float, threshold_deg: float
    ) -> np.ndarray:
        """Where the head goes for a gaze target at (yaw_t, pitch_t).

        Below the threshold the head stays put. Above it the head moves by
        ``head_gain`` times the excess, and always far enough that the eyes
        end no more than ``max_eccentricity_deg`` off-centre. The linear
        growth above a threshold is the classic description of head
        contribution to combined gaze shifts; taking a fixed fraction of the
        whole shift instead moves the head far too much for the 15-25 degree
        shifts that commissioning the zone involves.
        """
        tr = self.traits
        new = self.head.copy()
        pitch_goal = pitch_t - math.radians(tr.eye_down_bias_deg)
        cap = math.radians(tr.max_eccentricity_deg)
        for k, goal in ((0, yaw_t), (1, pitch_goal)):
            delta = goal - self.head[k]
            excess = abs(delta) - math.radians(threshold_deg)
            if excess > 0:
                move = max(tr.head_gain * excess, abs(delta) - cap)
                new[k] = self.head[k] + math.copysign(move, delta)
        return new

    def _maybe_recentre(self, seg: Segment) -> None:
        """Slow head re-centring during a held fixation (gaze held by VOR)."""
        if (
            seg.head != "natural"
            or self.head_plan is not None
            or self.saccade_plan is not None
            # A new instruction whose gaze shift has not started yet: the
            # current target is the old one, and re-centring on it now
            # would turn the head the wrong way.
            or self.pending is not None
        ):
            return
        if self.t - self._fixation_start < self._recentre_wait:
            return
        yaw_t, pitch_t = self._target_angles(self.gaze_target)
        tr = self.traits
        new = self.head.copy()
        pitch_goal = pitch_t - math.radians(tr.eye_down_bias_deg)
        for k, goal in ((0, yaw_t), (1, pitch_goal)):
            ecc = goal - self.head[k]
            if abs(ecc) > math.radians(tr.recentre_deg):
                new[k] = goal - math.copysign(math.radians(tr.recentre_to_deg), ecc)
        if np.max(np.abs(new[:2] - self.head[:2])) < math.radians(1.0):
            return
        amp = math.degrees(float(np.max(np.abs(new[:2] - self.head[:2]))))
        self.head_plan = (
            self.head.copy(),
            new,
            self.t,
            1.6 * head_movement_duration(amp),
        )
        self._recentre_wait = float(self.rng.uniform(0.6, 1.5)) + 10.0

    def _blink(self, partial: bool) -> None:
        self.blink_t0 = self.t
        self.blink_depth = self.rng.uniform(0.55, 0.8) if partial else 1.0

    def _blink_closure(self) -> float:
        if self.blink_t0 is None:
            return 0.0
        s = self.t - self.blink_t0
        close, hold, open_ = 0.075, 0.04, 0.17
        if s < close:
            c = (s / close) ** 2
        elif s < close + hold:
            c = 1.0
        elif s < close + hold + open_:
            c = 1.0 - min_jerk((s - close - hold) / open_)
        else:
            self.blink_t0 = None
            return 0.0
        return c * self.blink_depth

    def step_to(self, t: float) -> State:
        """Advance to time ``t`` (seconds) and return the state there."""
        dt = max(t - self.t, 0.0)
        self.t = t
        tr = self.traits

        while not self.finished and self.t - self.seg_start >= self.segment.duration:
            self._start_segment(self.seg_index + 1)
        seg = self.segment

        # Noise processes.
        self.sway.step(dt)
        self.lean_noise.step(dt)
        jitter = self.head_jitter.step(dt)
        roll_bias = float(self.roll_drift.step(dt)[0])
        drift = self.drift.step(dt)
        pupil = 0.0018 + float(self.pupil.step(dt)[0])

        # Steps.
        if self.step is not None:
            a, b, t0, dur = self.step
            s = (self.t - t0) / dur
            if s >= 0:
                self.stance = a + (b - a) * min_jerk(s)
                if s >= 1:
                    self.step = None

        # Pending gaze shift (reaction time after a new instruction).
        if self.pending is not None and self.t >= self.pending[0]:
            _, target = self.pending
            self.pending = None
            self._plan_gaze_shift(target, seg.head)

        # Scanning sub-fixations around an extended target.
        if (
            self.next_scan is not None
            and self.t >= self.next_scan
            and self.saccade_plan is None
            and self.head_plan is None
            and self.pending is None
        ):
            base = self._resolve(seg.target)
            off = self.rng.normal(0.0, seg.scan, size=3) * np.array([1.0, 0.6, 0.3])
            self._plan_gaze_shift(
                base + off, "still" if seg.head == "still" else "natural", scan=True
            )
            self.next_scan = self.t + self.rng.uniform(0.25, 0.9)

        # Head.
        self._maybe_recentre(seg)
        if self.head_plan is not None:
            a, b, t0, dur = self.head_plan
            s = (self.t - t0) / dur
            if s >= 0:
                self.head = a + (b - a) * min_jerk(s)
                if s >= 1:
                    self.head_plan = None
        head = self.head.copy()
        head += jitter
        if seg.head_roll_deg is not None:
            head[2] = math.radians(seg.head_roll_deg) * math.sin(
                2 * math.pi * (self.t - self.seg_start) / 4.0
            )
        else:
            head[2] = roll_bias + jitter[2]
        if seg.head_sweep_deg is not None:
            ay, ap, period = seg.head_sweep_deg
            ph = 2 * math.pi * (self.t - self.seg_start) / period
            head[0] += math.radians(ay) * math.sin(ph)
            head[1] += math.radians(ap) * math.sin(2 * ph)
        # The torso follows a large head turn, slowly.
        self.torso_yaw += (0.35 * self.head[0] - self.torso_yaw) * min(1.0, dt / 0.8)

        hips, torso_R, neck_base, pivot = self._body()
        head_R = self._head_R(head)
        eyes = self._eye_world(head_R, pivot)
        cyc = eyes.mean(axis=0)

        # Gaze: a saccade in flight, or locked on the target (VOR + pursuit).
        saccading = False
        if self.saccade_plan is not None:
            p0, p1, t0, dur = self.saccade_plan
            s = (self.t - t0) / dur
            d0 = normalize(p0 - cyc)
            d1 = normalize(p1 - cyc)
            ang = math.acos(float(np.clip(d0 @ d1, -1, 1)))
            k = saccade_profile(s)
            if ang > 1e-6:
                # Slerp the direction; interpolate the distance.
                d = (math.sin((1 - k) * ang) * d0 + math.sin(k * ang) * d1) / math.sin(
                    ang
                )
            else:
                d = d1
            dist = (1 - k) * np.linalg.norm(p0 - cyc) + k * np.linalg.norm(p1 - cyc)
            self.gaze_point = cyc + d * dist
            saccading = s < 1
            if s >= 1:
                self.saccade_plan = None
                self.gaze_point = p1.copy()
        else:
            # Microsaccades: small, corrective, about one a second.
            if self.t >= self.next_micro:
                self.micro_offset = -drift * 0.8 + self.rng.normal(
                    0, math.radians(0.12), 2
                )
                self.next_micro = self.t + 0.3 + self._exp(0.9)
            self.gaze_point = self.gaze_target.copy()

        fix = self.gaze_point
        if not saccading:
            # Fixational drift and the last microsaccade, as an angular offset.
            d = normalize(fix - cyc)
            yaw, pitch = yaw_pitch(d)
            off = drift + self.micro_offset
            fix = cyc + direction_from_yaw_pitch(
                yaw + off[0], pitch + off[1]
            ) * np.linalg.norm(fix - cyc)

        # Each eye aims at the fixated point itself: vergence for free.
        dirs = []
        eye_angles = []
        for c in eyes:
            v = head_R.T @ normalize(fix - c)
            ey, ep = yaw_pitch(v)
            ey = float(np.clip(ey, -EYE_YAW_LIMIT, EYE_YAW_LIMIT))
            ep = float(np.clip(ep, -EYE_UP_LIMIT, EYE_DOWN_LIMIT))
            dirs.append(direction_from_yaw_pitch(ey, ep))
            eye_angles.append((ey, ep))
        dirs = np.array(dirs)
        # Pinned at the oculomotor limit, the eyes are not on the target:
        # report where they actually point, not where the operator meant to.
        world = dirs @ head_R.T
        actual = normalize(world.mean(axis=0))
        aim = normalize(fix - cyc)
        if float(actual @ aim) < math.cos(math.radians(0.2)):
            fix = cyc + actual * float(np.linalg.norm(fix - cyc))

        # Blinks.
        if self.blink_t0 is None and self.t >= self.next_blink:
            self._blink(partial=self.rng.random() < 0.2)
            self.next_blink = self.t + 0.6 + self._exp(60.0 / tr.blink_rate_per_min)
        blink = self._blink_closure()

        # Hands.
        if self.hand_plan is not None:
            (a, fa), (b, fb), t0, dur = self.hand_plan
            s = (self.t - t0) / dur
            if s >= 0:
                k = min_jerk(s)
                self.hand_pos = a + (b - a) * k
                self.hand_on_carriage = fa + (fb - fa) * k
                if s >= 1:
                    self.hand_plan = None
        # The cutting stroke: push the carriage through the blade, draw it
        # back, again. Only while working; otherwise it stays where it is.
        if seg.hands == "work":
            self._stroke += dt
            self.carriage = cutting_stroke(self._stroke)
        hands = self.hand_pos.copy()
        hands[:, 2] -= self.hand_on_carriage * self.carriage

        return State(
            t=self.t,
            head_R=head_R,
            head_pivot=pivot,
            neck_base=neck_base,
            hips=hips,
            stance=self.stance.copy(),
            carriage=self.carriage,
            torso_R=torso_R,
            fixation=fix,
            cyclopean=cyc,
            eye_dirs_head=dirs,
            eye_centres_world=eyes,
            blink=blink,
            pupil_radius=pupil,
            hands=hands,
            segment=seg,
            saccade=saccading,
            head_angles=tuple(float(x) for x in head),
            eye_angles=np.array(eye_angles),
        )


def cutting_stroke(
    t: float, length: float = 0.10, push: float = 3.0, back: float = 1.5
):
    """Carriage travel over a repeating cutting stroke, metres.

    A slow, steady push through the blade, then a quicker draw back.
    """
    period = push + back
    s = t % period
    if s < push:
        return length * min_jerk(s / push)
    return length * (1.0 - min_jerk((s - push) / back))


# -- scenarios -------------------------------------------------------------


def calibration_segments(hold: float = 2.5, head: str = "natural") -> list[Segment]:
    """Look at each zone corner in turn, in the calibration routine's order.

    The fixation is held, as the commissioning instructions ask. ``head`` is
    how the operator gets there: ``natural`` eye-head coordination, which
    turns the head for the longer diagonal shifts, or ``still``, an operator
    who keeps their head where it is and moves only their eyes.
    ``calib_corner`` marks the segments so the evaluator knows when to
    "press c".
    """
    corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    segs = [Segment("zone:0.5,0.5", 1.5, activity="calibrate")]
    for i, (u, v) in enumerate(corners):
        segs.append(
            Segment(
                f"zone:{u},{v}",
                hold,
                head=head,
                activity="calibrate",
                calib_corner=i,
                hands="rest",
            )
        )
    segs.append(Segment("cut", 1.0, activity="calibrate"))
    return segs


# Activity model for a shift at the saw: (name, weight, duration range s,
# segment template). Watching the cut dominates; glances away are short.
TASK_MODEL = [
    ("watch_cut", 0.50, (2.0, 9.0), dict(target="cut", scan=0.025, hands="work")),
    (
        "watch_guide",
        0.06,
        (0.6, 2.0),
        dict(target="blade_guide", scan=0.01, hands="work"),
    ),
    (
        "check_product",
        0.12,
        (0.4, 1.6),
        dict(target="product", scan=0.03, hands="work"),
    ),
    ("reach_carriage", 0.06, (0.8, 2.0), dict(target="carriage", hands="carriage")),
    ("unload_tray", 0.09, (1.0, 3.5), dict(target="tray", scan=0.04, hands="tray")),
    ("controls", 0.03, (0.5, 1.2), dict(target="controls", hands="work")),
    (
        "coworker",
        0.04,
        (1.5, 4.0),
        dict(target="coworker_left", head="face", hands="rest"),
    ),
    (
        "coworker_r",
        0.03,
        (1.0, 3.0),
        dict(target="coworker_right", head="face", hands="rest"),
    ),
    ("look_down", 0.03, (0.5, 1.5), dict(target="floor", hands="rest")),
    ("clock", 0.02, (0.6, 1.5), dict(target="clock", hands="work")),
    ("glance_camera", 0.02, (0.3, 1.0), dict(target="camera", hands="work")),
]


def shift_segments(duration: float, rng: np.random.Generator) -> list[Segment]:
    """A stochastic stretch of work at the saw."""
    names = [m[0] for m in TASK_MODEL]
    weights = np.array([m[1] for m in TASK_MODEL])
    weights = weights / weights.sum()
    segs = []
    t = 0.0
    last = None
    stance = np.array(STANCE)
    while t < duration:
        i = int(rng.choice(len(names), p=weights))
        name, _, (lo, hi), tmpl = TASK_MODEL[i]
        if name == last and name != "watch_cut":
            continue
        # Every glance away is followed by a return to the cut.
        d = float(rng.uniform(lo, hi))
        seg = Segment(duration=d, activity=name, **tmpl)
        if rng.random() < 0.08:
            # Occasionally shift position at the table: a step back, closer,
            # or sideways along it. Exercises the depth compensation.
            # Never closer than the rail allows.
            stance = np.array(STANCE) + np.abs(rng.normal(0, [0.06, 0.08])) * [
                rng.choice([-1, 1]),
                1,
            ]
            seg = replace(seg, stance=tuple(stance))
        segs.append(seg)
        t += d
        if name != "watch_cut":
            back = float(rng.uniform(1.5, 6.0))
            segs.append(Segment("cut", back, scan=0.025, activity="watch_cut"))
            t += back
        last = name
    return segs


def fixation_segments(duration: float) -> list[Segment]:
    """Stand still and stare at the cut: the noise-floor protocol."""
    return [Segment("cut", duration, head="still", activity="fixation", hands="work")]


def roll_segments(duration: float) -> list[Segment]:
    """Keep staring at the cut and tilt the head slowly side to side."""
    return [
        Segment(
            "cut",
            duration,
            head="still",
            activity="roll",
            head_roll_deg=14.0,
            hands="work",
        )
    ]


def head_sweep_segments(duration: float) -> list[Segment]:
    """Gaze held on the cut while the head turns: pure VOR.

    True gaze never leaves the zone, so every alarm here is the pipeline
    mistaking eye-in-head rotation for a change of gaze.
    """
    return [
        Segment(
            "cut",
            duration,
            head="still",
            activity="vor",
            head_sweep_deg=(15.0, 6.0, 5.0),
        )
    ]


def eyes_only_segments(duration: float, rng: np.random.Generator) -> list[Segment]:
    """Head held still while the eyes alone move between targets."""
    segs = []
    t = 0.0
    pool = [
        "cut",
        "product",
        "blade_guide",
        "zone:0.0,0.0",
        "zone:1.0,1.0",
        "controls",
        "tray",
    ]
    while t < duration:
        name = pool[int(rng.integers(len(pool)))]
        d = float(rng.uniform(0.8, 2.5))
        segs.append(Segment(name, d, head="still", activity=f"eyes:{name}"))
        t += d
    return segs


def distance_segments(duration: float) -> list[Segment]:
    """Watch the cut while stepping back and forth from the table."""
    segs = []
    x, z = STANCE
    stances = [(x, z), (x, z + 0.18), (x, z + 0.03), (x + 0.09, z + 0.08), (x, z)]
    per = duration / len(stances)
    for s in stances:
        segs.append(Segment("cut", per, scan=0.02, stance=s, activity="distance"))
    return segs


def glances_segments(duration: float, rng: np.random.Generator) -> list[Segment]:
    """Controlled excursions of known length out of the zone and back.

    For measuring alarm latency: each excursion is a clean step out of the
    zone to the tray, the carriage or a coworker, of a duration straddling
    the warning and alert thresholds.
    """
    segs = []
    t = 0.0
    away = ["tray", "carriage", "coworker_left", "controls", "floor"]
    lengths = [0.5, 1.5, 2.5, 4.0]
    k = 0
    while t < duration:
        segs.append(Segment("cut", 4.0, scan=0.02, activity="watch_cut"))
        target = away[k % len(away)]
        d = lengths[k % len(lengths)]
        segs.append(Segment(target, d, activity=f"glance:{target}"))
        t += 4.0 + d
        k += 1
    return segs


SCENARIOS = {
    "shift": "A realistic stretch of work, generated from the task model",
    "fixation": "Head still, staring at the cut: the noise floor",
    "roll": "Staring at the cut while tilting the head side to side",
    "vor": "Gaze fixed on the cut while the head turns: eye-in-head is not gaze",
    "eyes-only": "Head still; eyes alone move between targets in and out of the zone",
    "distance": "Watching the cut while stepping back and forth",
    "glances": "Timed glances away, for alarm latency",
}


def build_segments(
    name: str,
    duration: float,
    seed: int = 0,
    calibrate: bool = True,
    calibration_head: str = "natural",
) -> list[Segment]:
    rng = np.random.default_rng(seed + 1000)
    builders = {
        "shift": lambda: shift_segments(duration, rng),
        "fixation": lambda: fixation_segments(duration),
        "roll": lambda: roll_segments(duration),
        "vor": lambda: head_sweep_segments(duration),
        "eyes-only": lambda: eyes_only_segments(duration, rng),
        "distance": lambda: distance_segments(duration),
        "glances": lambda: glances_segments(duration, rng),
    }
    if name not in builders:
        raise ValueError(
            f"Unknown scenario {name!r}. Expected one of {', '.join(SCENARIOS)}"
        )
    segs = calibration_segments(head=calibration_head) if calibrate else []
    return segs + builders[name]()


@dataclass
class Clock:
    """Frame timing. Kept separate so a source can run faster than real time."""

    fps: float = 30.0
    frame: int = field(default=0)

    def next(self) -> float:
        t = self.frame / self.fps
        self.frame += 1
        return t
