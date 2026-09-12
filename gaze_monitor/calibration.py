"""Four-corner calibration of the attention area, and its on-disk record.

The operator looks at each corner of the region they must watch -- the blade
and the cut around it -- and the mean iris displacement at each corner becomes
one edge of an :class:`~gaze_monitor.attention.AttentionZone`.

This is a commissioning step, not a per-session one: it is done once when the
machine is installed and then loaded from disk on every start, until someone
asks for a factory reset. That makes the persistence format part of the
contract, so it is versioned and written atomically.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .attention import CORNER_NAMES, AttentionZone

log = logging.getLogger(__name__)

# Bump when the on-disk shape changes. A record from a different version is
# refused rather than guessed at: silently reinterpreting old calibration
# numbers would move the danger zone without telling anyone.
CALIBRATION_VERSION = 1

# Environment override, so the container can point at a mounted volume.
CALIBRATION_ENV_VAR = "GAZE_CALIBRATION_FILE"


def default_calibration_path() -> pathlib.Path:
    """Where calibration lives unless told otherwise."""
    override = os.environ.get(CALIBRATION_ENV_VAR)
    if override:
        return pathlib.Path(override).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return pathlib.Path(base).expanduser() / "gaze_monitor" / "calibration.json"


@dataclass(frozen=True)
class CalibrationRecord:
    """A stored calibration, plus enough provenance to audit it."""

    zone: AttentionZone
    corners: list[tuple[float, float]]
    source: str = "unknown"
    created_utc: str = ""
    frame_size: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "version": CALIBRATION_VERSION,
            "created_utc": self.created_utc,
            "source": self.source,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "corner_order": list(CORNER_NAMES),
            "corners": [list(c) for c in self.corners],
            "zone": self.zone.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationRecord:
        version = data.get("version")
        if version != CALIBRATION_VERSION:
            raise ValueError(
                f"Calibration file is version {version!r}, this build expects "
                f"{CALIBRATION_VERSION}. Re-run calibration."
            )
        frame_size = data.get("frame_size")
        return cls(
            zone=AttentionZone.from_dict(data["zone"]),
            corners=[tuple(c) for c in data.get("corners", [])],
            source=data.get("source", "unknown"),
            created_utc=data.get("created_utc", ""),
            frame_size=tuple(frame_size) if frame_size else None,
        )


def save_calibration(record: CalibrationRecord, path: pathlib.Path) -> None:
    """Write a calibration record atomically.

    A half-written file would be indistinguishable from a corrupt one on the
    next start, and on an appliance that reboots with the machine that is a
    realistic way to lose the commissioning data. Write to a temporary file in
    the same directory, then rename.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record.to_dict(), indent=2, sort_keys=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        pathlib.Path(tmp_name).unlink(missing_ok=True)
        raise
    log.info("Calibration saved to %s", path)


def load_calibration(path: pathlib.Path) -> CalibrationRecord | None:
    """Load a calibration record, or ``None`` if there is nothing usable.

    A missing file is the normal first-run state. A corrupt or
    wrong-version file is logged and treated as missing, so the system
    falls back to asking for calibration rather than refusing to start.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return CalibrationRecord.from_dict(json.load(handle))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("Ignoring unusable calibration at %s: %s", path, exc)
        return None


def factory_reset(path: pathlib.Path) -> bool:
    """Delete the stored calibration. Returns True if a file was removed."""
    path = pathlib.Path(path)
    if not path.exists():
        return False
    path.unlink()
    log.info("Calibration cleared: %s", path)
    return True


@dataclass
class GazeCalibrator:
    """Collects four corner fixations and turns them into an AttentionZone.

    ``sigma_threshold`` rejects a corner the operator was not actually holding
    still on. ``min_samples`` rejects one measured from too few frames: with a
    single sample the standard deviation is 0 and the jitter check passes
    trivially, which is exactly the case it exists to catch.
    """

    sample_duration: float = 1.0
    sigma_threshold: float = 0.05
    min_samples: int = 5

    is_collecting: bool = field(default=False, init=False)
    start_time: float = field(default=0.0, init=False)
    samples: list[tuple[float, float, float]] = field(default_factory=list, init=False)
    corners: list[tuple[float, float]] = field(default_factory=list, init=False)
    calib_z: float = field(default=0.0, init=False)

    # -- collection --------------------------------------------------------

    @property
    def corner_index(self) -> int:
        return len(self.corners)

    @property
    def current_corner_name(self) -> str:
        return CORNER_NAMES[min(self.corner_index, len(CORNER_NAMES) - 1)]

    @property
    def is_finished(self) -> bool:
        return len(self.corners) >= len(CORNER_NAMES)

    def start_collection(self, now: float) -> None:
        self.is_collecting = True
        self.start_time = now
        self.samples = []

    def collect(self, now: float, dx: float, dy: float, z_m: float = 0.0) -> int:
        """Record one sample. Returns collection progress as a percentage.

        Only call this for frames where the gaze was actually measured;
        feeding it zeros for untracked frames drags the corner mean toward
        the origin and inflates the jitter estimate.
        """
        if not self.is_collecting:
            return 0
        self.samples.append((dx, dy, z_m))
        elapsed = now - self.start_time
        if self.sample_duration <= 0:
            return 100
        return int(max(0.0, min(1.0, elapsed / self.sample_duration)) * 100)

    def should_finish(self, now: float) -> bool:
        return self.is_collecting and (now - self.start_time) >= self.sample_duration

    def validate_and_save(self) -> tuple[bool, str]:
        """Accept or reject the collected fixation."""
        self.is_collecting = False
        count = len(self.samples)

        if count < self.min_samples:
            return False, (
                f"Only {count} samples (need {self.min_samples}). "
                "Is the camera seeing your eyes?"
            )

        arr = np.asarray(self.samples, dtype=float)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)

        if stds[0] > self.sigma_threshold or stds[1] > self.sigma_threshold:
            return False, (
                f"Too much movement (sx:{stds[0]:.3f}, sy:{stds[1]:.3f}). "
                "Hold still and try again."
            )

        self.corners.append((float(means[0]), float(means[1])))

        # Depth is taken from the first corner only: it anchors the distance
        # the calibration was performed at, and all four are the same standing
        # position anyway. Zero means the camera gave no usable reading, in
        # which case depth compensation stays off.
        if len(self.corners) == 1:
            valid_z = arr[:, 2][arr[:, 2] > 0.0]
            self.calib_z = float(np.median(valid_z)) if valid_z.size else 0.0

        return True, f"{CORNER_NAMES[len(self.corners) - 1]} accepted"

    # -- result ------------------------------------------------------------

    def build_zone(self, margin: float = 0.1) -> AttentionZone:
        if not self.is_finished:
            raise RuntimeError(
                f"Need {len(CORNER_NAMES)} corners, have {len(self.corners)}"
            )
        return AttentionZone.from_corners(
            self.corners, calib_z=self.calib_z, margin=margin
        )

    def build_record(
        self,
        source: str = "unknown",
        margin: float = 0.1,
        frame_size: tuple[int, int] | None = None,
    ) -> CalibrationRecord:
        return CalibrationRecord(
            zone=self.build_zone(margin=margin),
            corners=list(self.corners),
            source=source,
            created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            frame_size=frame_size,
        )

    def reset(self) -> None:
        self.is_collecting = False
        self.start_time = 0.0
        self.samples = []
        self.corners = []
        self.calib_z = 0.0
