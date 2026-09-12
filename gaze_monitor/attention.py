"""Attention monitoring: is the operator looking at the danger zone?

The four-corner calibration defines the **attention area** -- the region the
operator must be watching, in practice the blade and the cut immediately
around it. This module turns a stream of gaze samples into a state:

    ATTENTIVE -> WARNING -> ALERT

with a fault state for when the camera stops producing usable frames.

The design bias throughout is to fail *loud*. Losing the operator's face,
losing the camera, or getting an unusable depth reading all escalate rather
than silently holding ATTENTIVE, because a monitor that goes quiet when it
breaks is worse than no monitor at all.

Everything here is pure: time is injected, so the state machine is tested
without sleeping.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

# Corner order used by the calibration routine and by AttentionZone.
CORNER_NAMES = ("Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right")

# A zone whose span is narrower than this in gaze-displacement units is not a
# usable calibration -- the four corners were effectively the same point.
MIN_ZONE_SPAN = 1e-3


class AttentionState(str, Enum):
    ATTENTIVE = "attentive"
    WARNING = "warning"
    ALERT = "alert"
    FAULT = "fault"

    @property
    def is_alarming(self) -> bool:
        return self in (AttentionState.ALERT, AttentionState.FAULT)


@dataclass(frozen=True)
class AttentionZone:
    """The calibrated attention area, in normalised iris-displacement units.

    The bounds are stored signed rather than sorted: a mirrored camera gives
    dx_min > dx_max, and dividing by the signed span handles that without a
    separate flip flag.
    """

    dx_min: float
    dx_max: float
    dy_min: float
    dy_max: float
    calib_z: float = 0.0
    margin: float = 0.1

    def __post_init__(self) -> None:
        if abs(self.dx_max - self.dx_min) < MIN_ZONE_SPAN:
            raise ValueError("Degenerate attention zone: horizontal span is ~0")
        if abs(self.dy_max - self.dy_min) < MIN_ZONE_SPAN:
            raise ValueError("Degenerate attention zone: vertical span is ~0")
        if self.margin < 0.0:
            raise ValueError("margin must be >= 0")

    @classmethod
    def from_corners(
        cls,
        corners: Sequence[tuple[float, float]],
        calib_z: float = 0.0,
        margin: float = 0.1,
    ) -> AttentionZone:
        """Build a zone from four corners in CORNER_NAMES order."""
        if len(corners) != 4:
            raise ValueError(f"Expected 4 corners, got {len(corners)}")
        tl, tr, bl, br = corners
        return cls(
            dx_min=(tl[0] + bl[0]) / 2.0,
            dx_max=(tr[0] + br[0]) / 2.0,
            dy_min=(tl[1] + tr[1]) / 2.0,
            dy_max=(bl[1] + br[1]) / 2.0,
            calib_z=calib_z,
            margin=margin,
        )

    def project(self, dx: float, dy: float, z_m: float = 0.0) -> tuple[float, float]:
        """Map a gaze sample to zone-relative coordinates.

        (0, 0) is the top-left calibrated corner and (1, 1) the bottom-right.
        Values outside 0..1 mean the operator is looking outside the zone.

        When both the sample and the calibration carry a depth reading, the
        displacement is scaled by the distance ratio: the same eye rotation
        subtends a larger area further from the camera, so without this an
        operator who steps back reads as looking wider than they are.
        """
        scale = 1.0
        if self.calib_z > 0.0 and z_m > 0.0:
            scale = z_m / self.calib_z
        u = (dx * scale - self.dx_min) / (self.dx_max - self.dx_min)
        v = (dy * scale - self.dy_min) / (self.dy_max - self.dy_min)
        return u, v

    def contains(self, dx: float, dy: float, z_m: float = 0.0) -> bool:
        """True if the gaze falls inside the zone, plus the margin."""
        u, v = self.project(dx, dy, z_m)
        if not (math.isfinite(u) and math.isfinite(v)):
            return False
        lo, hi = -self.margin, 1.0 + self.margin
        return lo <= u <= hi and lo <= v <= hi

    def to_dict(self) -> dict:
        return {
            "dx_min": self.dx_min,
            "dx_max": self.dx_max,
            "dy_min": self.dy_min,
            "dy_max": self.dy_max,
            "calib_z": self.calib_z,
            "margin": self.margin,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttentionZone:
        try:
            return cls(
                dx_min=float(data["dx_min"]),
                dx_max=float(data["dx_max"]),
                dy_min=float(data["dy_min"]),
                dy_max=float(data["dy_max"]),
                calib_z=float(data.get("calib_z", 0.0)),
                margin=float(data.get("margin", 0.1)),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed attention zone: {exc}") from exc


@dataclass(frozen=True)
class AttentionStatus:
    """The monitor's verdict for one frame."""

    state: AttentionState
    away_seconds: float = 0.0
    in_zone: bool = False
    tracked: bool = False
    position: tuple[float, float] | None = None

    @property
    def is_alarming(self) -> bool:
        return self.state.is_alarming


TransitionHook = Callable[[AttentionState, AttentionState, "AttentionStatus"], None]


@dataclass
class AttentionMonitor:
    """Escalates from ATTENTIVE to ALERT while the operator looks away.

    Thresholds:

    * ``warn_after``  -- seconds outside the zone before warning. Glancing at
      the feed table is normal; this is the grace period that stops the
      monitor crying wolf.
    * ``alert_after`` -- seconds outside the zone before alerting.
    * ``clear_after`` -- seconds the gaze must be *continuously* back inside
      before the alarm clears. Without this the state flickers on every
      blink-length detection dropout right at the zone edge.
    * ``fault_after`` -- seconds of unusable camera frames before FAULT.
    """

    zone: AttentionZone | None = None
    warn_after: float = 1.0
    alert_after: float = 3.0
    clear_after: float = 0.4
    fault_after: float = 1.0
    on_transition: TransitionHook | None = None

    state: AttentionState = field(default=AttentionState.ATTENTIVE, init=False)
    _away_since: float | None = field(default=None, init=False)
    _returned_since: float | None = field(default=None, init=False)
    _camera_bad_since: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.warn_after <= self.alert_after:
            raise ValueError("Require 0 <= warn_after <= alert_after")

    def reset(self, now: float = 0.0) -> None:
        self.state = AttentionState.ATTENTIVE
        self._away_since = None
        self._returned_since = None
        self._camera_bad_since = None

    def update(
        self,
        now: float,
        gaze=None,
        camera_ok: bool = True,
    ) -> AttentionStatus:
        """Advance the state machine by one frame.

        ``gaze`` is a :class:`gaze_monitor.gaze.GazeSample` or ``None`` when the
        operator's eyes were not measurable this frame. Not measurable counts
        as looking away: from a single camera, "turned away from the blade" and
        "not tracking" are the same observation, and the safe reading is the
        pessimistic one.
        """
        if not camera_ok:
            return self._camera_fault(now)
        self._camera_bad_since = None

        if self.zone is None:
            # Uncalibrated: nothing to be attentive *to*. Hold, do not alarm.
            return self._settle(now, AttentionState.ATTENTIVE, 0.0, False, False, None)

        tracked = gaze is not None
        position: tuple[float, float] | None = None
        in_zone = False
        if tracked:
            position = self.zone.project(gaze.dx, gaze.dy, gaze.z_m)
            in_zone = self.zone.contains(gaze.dx, gaze.dy, gaze.z_m)

        if in_zone:
            return self._handle_in_zone(now, tracked, position)
        return self._handle_out_of_zone(now, tracked, position)

    # -- internals ---------------------------------------------------------

    def _camera_fault(self, now: float) -> AttentionStatus:
        if self._camera_bad_since is None:
            self._camera_bad_since = now
        bad_for = now - self._camera_bad_since
        if bad_for >= self.fault_after:
            return self._settle(
                now, AttentionState.FAULT, self._away(now), False, False, None
            )
        # Inside the grace window hold the previous verdict rather than
        # claiming either health or failure on one dropped frame.
        return self._status(self.state, self._away(now), False, False, None)

    def _handle_in_zone(self, now, tracked, position) -> AttentionStatus:
        if self._away_since is None:
            return self._settle(
                now, AttentionState.ATTENTIVE, 0.0, True, tracked, position
            )

        if self._returned_since is None:
            self._returned_since = now
        if now - self._returned_since >= self.clear_after:
            self._away_since = None
            self._returned_since = None
            return self._settle(
                now, AttentionState.ATTENTIVE, 0.0, True, tracked, position
            )

        # Back in the zone but not yet long enough to clear: freeze the away
        # timer and hold the current state.
        return self._status(self.state, self._away(now), True, tracked, position)

    def _handle_out_of_zone(self, now, tracked, position) -> AttentionStatus:
        self._returned_since = None
        if self._away_since is None:
            self._away_since = now
        away = now - self._away_since

        if away >= self.alert_after:
            state = AttentionState.ALERT
        elif away >= self.warn_after:
            state = AttentionState.WARNING
        else:
            state = AttentionState.ATTENTIVE
        return self._settle(now, state, away, False, tracked, position)

    def _away(self, now: float) -> float:
        return 0.0 if self._away_since is None else now - self._away_since

    def _status(self, state, away, in_zone, tracked, position) -> AttentionStatus:
        return AttentionStatus(
            state=state,
            away_seconds=away,
            in_zone=in_zone,
            tracked=tracked,
            position=position,
        )

    def _settle(self, now, state, away, in_zone, tracked, position) -> AttentionStatus:
        status = self._status(state, away, in_zone, tracked, position)
        if state is not self.state:
            previous, self.state = self.state, state
            if self.on_transition is not None:
                self.on_transition(previous, state, status)
        return status
