"""Tests for configuration merging and validation."""

import pathlib

import pytest

from gaze_monitor.config import (
    ConfigError,
    MonitorConfig,
    build_config,
    load_config_file,
)


def test_defaults_are_usable():
    cfg = MonitorConfig()
    assert cfg.source == "realsense"
    assert cfg.warn_after <= cfg.alert_after


def test_cli_overrides_file_overrides_default():
    cfg = build_config(
        file_values={"source": "webcam", "alert_after": 5.0},
        cli_values={"alert_after": 2.0},
    )
    assert cfg.source == "webcam"  # from file
    assert cfg.alert_after == 2.0  # CLI wins
    assert cfg.warn_after == MonitorConfig().warn_after  # default


def test_none_cli_values_do_not_override():
    """argparse uses None for 'flag not supplied'."""
    cfg = build_config(file_values={"source": "kinect"}, cli_values={"source": None})
    assert cfg.source == "kinect"


def test_unknown_keys_are_warned_not_fatal(caplog):
    cfg = build_config(file_values={"source": "webcam", "typoed_key": 1})
    assert cfg.source == "webcam"
    assert "typoed_key" in caplog.text


def test_calibration_file_becomes_a_path():
    cfg = build_config(cli_values={"calibration_file": "/tmp/x/calib.json"})
    assert isinstance(cfg.calibration_file, pathlib.Path)


@pytest.mark.parametrize(
    "values",
    [
        {"filter_alpha": 0.0},
        {"filter_alpha": 1.5},
        {"warn_after": 5.0, "alert_after": 1.0},
        {"alert_after": -1.0},
        {"zone_margin": -0.1},
        {"sample_duration": 0.0},
        {"screen_w": 0},
        {"headless": True, "fullscreen": True},
    ],
)
def test_invalid_configurations_are_rejected(values):
    with pytest.raises(ConfigError):
        build_config(file_values=values)


def test_missing_config_file_is_not_an_error(tmp_path):
    assert load_config_file(tmp_path / "absent.yaml") == {}


def test_empty_config_file_is_not_an_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    assert load_config_file(path) == {}


def test_config_file_must_be_a_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError):
        load_config_file(path)


def test_example_config_is_valid():
    """The shipped example must actually load -- it is the documented start."""
    values = load_config_file(pathlib.Path("config.example.yaml"))
    cfg = build_config(file_values=values)
    assert cfg.alert_after > 0
