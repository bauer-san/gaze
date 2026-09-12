"""Tests for four-corner calibration and its persistence.

Calibration is a commissioning step that is meant to survive reboots, so the
on-disk format gets as much attention as the collection logic.
"""

import json

import pytest

from gaze_monitor.attention import CORNER_NAMES, AttentionZone
from gaze_monitor.calibration import (
    CALIBRATION_ENV_VAR,
    CALIBRATION_VERSION,
    CalibrationRecord,
    GazeCalibrator,
    default_calibration_path,
    factory_reset,
    load_calibration,
    save_calibration,
)

CORNERS = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]


def collect_corner(cal, dx, dy, z=1.0, n=20, jitter=0.0, t0=0.0):
    """Drive one full corner fixation through the calibrator."""
    cal.start_collection(t0)
    for i in range(n):
        wobble = jitter if i % 2 else -jitter
        cal.collect(t0 + i * 0.05, dx + wobble, dy + wobble, z)
    return cal.validate_and_save()


def calibrated():
    cal = GazeCalibrator(sample_duration=1.0, sigma_threshold=0.05)
    for dx, dy in CORNERS:
        ok, msg = collect_corner(cal, dx, dy)
        assert ok, msg
    return cal


# --- collection -----------------------------------------------------------


def test_steady_fixation_is_accepted():
    cal = GazeCalibrator()
    ok, msg = collect_corner(cal, 0.1, 0.05)
    assert ok, msg
    assert cal.corner_index == 1


def test_jittery_fixation_is_rejected():
    cal = GazeCalibrator(sigma_threshold=0.01)
    ok, msg = collect_corner(cal, 0.1, 0.05, jitter=0.5)
    assert not ok
    assert "movement" in msg
    assert cal.corner_index == 0  # not advanced


def test_too_few_samples_is_rejected():
    """One sample has zero stddev, so the jitter check passes trivially."""
    cal = GazeCalibrator(min_samples=5)
    cal.start_collection(0.0)
    cal.collect(0.0, 0.1, 0.1, 1.0)
    ok, msg = cal.validate_and_save()
    assert not ok
    assert "samples" in msg


def test_rejection_does_not_leave_collection_open():
    cal = GazeCalibrator(min_samples=5)
    cal.start_collection(0.0)
    cal.validate_and_save()
    assert not cal.is_collecting


def test_collect_ignored_when_not_collecting():
    cal = GazeCalibrator()
    assert cal.collect(0.0, 0.1, 0.1, 1.0) == 0
    assert cal.samples == []


def test_progress_advances_and_clamps():
    cal = GazeCalibrator(sample_duration=1.0)
    cal.start_collection(0.0)
    assert cal.collect(0.0, 0.1, 0.1) == 0
    assert cal.collect(0.5, 0.1, 0.1) == 50
    assert cal.collect(9.0, 0.1, 0.1) == 100  # clamped


def test_should_finish_after_duration():
    cal = GazeCalibrator(sample_duration=1.0)
    cal.start_collection(10.0)
    assert not cal.should_finish(10.5)
    assert cal.should_finish(11.0)


def test_corner_names_advance_in_order():
    cal = GazeCalibrator()
    seen = []
    for dx, dy in CORNERS:
        seen.append(cal.current_corner_name)
        collect_corner(cal, dx, dy)
    assert seen == list(CORNER_NAMES)
    assert cal.is_finished


def test_calib_z_taken_from_first_corner_only():
    cal = GazeCalibrator()
    collect_corner(cal, -0.5, -0.5, z=1.5)
    collect_corner(cal, 0.5, -0.5, z=9.9)
    assert cal.calib_z == pytest.approx(1.5)


def test_calib_z_zero_when_no_depth_readings():
    """Without depth the zone must not apply a bogus compensation factor."""
    cal = GazeCalibrator()
    collect_corner(cal, -0.5, -0.5, z=0.0)
    assert cal.calib_z == 0.0


def test_build_zone_requires_all_corners():
    cal = GazeCalibrator()
    collect_corner(cal, -0.5, -0.5)
    with pytest.raises(RuntimeError):
        cal.build_zone()


def test_build_zone_matches_corners():
    zone = calibrated().build_zone()
    assert zone.dx_min == pytest.approx(-0.5)
    assert zone.dx_max == pytest.approx(0.5)
    assert zone.dy_min == pytest.approx(-0.5)
    assert zone.dy_max == pytest.approx(0.5)


def test_reset_clears_everything():
    cal = calibrated()
    cal.reset()
    assert cal.corner_index == 0
    assert not cal.is_finished
    assert cal.calib_z == 0.0


# --- persistence ----------------------------------------------------------


def test_record_roundtrips_through_disk(tmp_path):
    path = tmp_path / "nested" / "calibration.json"
    record = calibrated().build_record(source="realsense", frame_size=(640, 480))
    save_calibration(record, path)

    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.zone == record.zone
    assert loaded.source == "realsense"
    assert loaded.frame_size == (640, 480)
    assert loaded.created_utc


def test_saved_file_records_version_and_corner_order(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(calibrated().build_record(), path)
    data = json.loads(path.read_text())
    assert data["version"] == CALIBRATION_VERSION
    assert data["corner_order"] == list(CORNER_NAMES)


def test_missing_file_loads_as_none(tmp_path):
    assert load_calibration(tmp_path / "absent.json") is None


def test_corrupt_file_loads_as_none(tmp_path):
    """A corrupt file must degrade to "needs calibration", not to a crash."""
    path = tmp_path / "calibration.json"
    path.write_text("{not json at all")
    assert load_calibration(path) is None


def test_wrong_version_is_refused(tmp_path):
    """Old numbers must never be reinterpreted -- that silently moves the zone."""
    path = tmp_path / "calibration.json"
    record = calibrated().build_record().to_dict()
    record["version"] = CALIBRATION_VERSION + 1
    path.write_text(json.dumps(record))
    assert load_calibration(path) is None


def test_degenerate_stored_zone_is_refused(tmp_path):
    path = tmp_path / "calibration.json"
    data = calibrated().build_record().to_dict()
    data["zone"]["dx_max"] = data["zone"]["dx_min"]
    path.write_text(json.dumps(data))
    assert load_calibration(path) is None


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(calibrated().build_record(), path)
    save_calibration(calibrated().build_record(), path)
    assert [p.name for p in tmp_path.iterdir()] == ["calibration.json"]


def test_save_overwrites_previous_calibration(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(calibrated().build_record(), path)

    wider = CalibrationRecord(
        zone=AttentionZone.from_corners(
            [(-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0)]
        ),
        corners=[],
    )
    save_calibration(wider, path)
    assert load_calibration(path).zone.dx_max == pytest.approx(2.0)


# --- factory reset --------------------------------------------------------


def test_factory_reset_removes_the_file(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(calibrated().build_record(), path)
    assert factory_reset(path) is True
    assert not path.exists()
    assert load_calibration(path) is None


def test_factory_reset_on_missing_file_is_not_an_error(tmp_path):
    assert factory_reset(tmp_path / "absent.json") is False


# --- default location -----------------------------------------------------


def test_env_var_overrides_default_path(monkeypatch, tmp_path):
    target = tmp_path / "mounted" / "calibration.json"
    monkeypatch.setenv(CALIBRATION_ENV_VAR, str(target))
    assert default_calibration_path() == target


def test_default_path_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv(CALIBRATION_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_calibration_path() == tmp_path / "gaze_monitor" / "calibration.json"
