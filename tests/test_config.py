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
        # A brake threshold with nowhere to send it looks armed and is not.
        {"brake_after": 3.0},
        {"brake_after": -1.0},
        {"brake_after": 3.0, "safety_host": "10.0.0.5", "safety_kind": "holding"},
        {"safety_port": 0},
        {"safety_unit_id": 248},
        {"safety_address": -1},
        {"safety_interval": 0.0},
        # Staleness must exceed the renewal interval or nothing is ever sent.
        {"safety_stale_after": 0.1, "safety_interval": 0.1},
    ],
)
def test_invalid_configurations_are_rejected(values):
    with pytest.raises(ConfigError):
        build_config(file_values=values)


def test_the_safety_output_is_off_by_default():
    """Nothing reaches a machine controller unless someone configured it."""
    cfg = build_config()
    assert cfg.brake_after == 0.0
    assert cfg.safety_host == ""
    assert cfg.safety_auto_arm is False


def test_a_configured_safety_output_validates():
    cfg = build_config(
        file_values={
            "brake_after": 3.0,
            "safety_host": "127.0.0.1",
            "safety_port": 5020,
            "safety_kind": "register",
            "safety_address": 40,
        }
    )
    assert cfg.brake_after == 3.0
    assert cfg.safety_kind == "register"


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


# -- transports and the machine reader --------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        {"safety_transport": "carrier-pigeon"},
        # A serial transport with only a host set: configured-looking, and
        # writing nowhere.
        {"brake_after": 3.0, "safety_transport": "rtu", "safety_host": "10.0.0.5"},
        {"brake_after": 3.0, "safety_transport": "relay"},
        {"safety_relay_protocol": "morse"},
        {"safety_relay_channel": -1},
        {"safety_gpio_line": ""},
        {"safety_gpio_line": "   "},
        {"safety_baud": 0},
        {"safety_parity": "Z"},
        {"safety_stopbits": 3},
        {"safety_bytesize": 9},
        {"machine_unit_id": 248},
        {"machine_interval": 0.0},
        {"machine_parity": "X"},
        # One serial device cannot be both the permit output and the reader.
        {"machine_port": "/dev/ttyUSB0", "safety_serial_port": "/dev/ttyUSB0"},
    ],
)
def test_invalid_transport_configurations_are_rejected(values):
    with pytest.raises(ConfigError):
        build_config(values)


def test_the_error_names_the_field_the_transport_actually_needs():
    """Naming safety_host when the transport is serial is how people end up
    setting both and still having nothing written."""
    with pytest.raises(ConfigError, match="safety_serial_port"):
        build_config({"brake_after": 3.0, "safety_transport": "relay"})
    with pytest.raises(ConfigError, match="safety_host"):
        build_config({"brake_after": 3.0, "safety_transport": "tcp"})


def test_a_usb_relay_output_validates():
    cfg = build_config(
        {
            "brake_after": 3.0,
            "safety_transport": "relay",
            "safety_serial_port": "/dev/ttyACM0",
            "safety_relay_protocol": "lcus",
            "safety_relay_channel": 1,
        }
    )
    assert cfg.safety_transport == "relay"
    assert cfg.safety_relay_channel == 1


def test_the_machine_reader_is_off_by_default():
    cfg = build_config()
    assert cfg.machine_port == ""
    # -1 by default: the drives this was written against document an address
    # one higher than the one that goes on the wire.
    assert cfg.machine_address_offset == -1


def test_the_reader_and_the_permit_may_use_different_serial_ports():
    cfg = build_config(
        {
            "brake_after": 3.0,
            "safety_transport": "relay",
            "safety_serial_port": "/dev/ttyACM0",
            "machine_port": "/dev/ttyUSB0",
        }
    )
    assert cfg.machine_port != cfg.safety_serial_port


# -- the live preview -------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        {"preview_port": 70000},
        {"preview_draw": "wireframe"},
        {"preview_quality": 0},
        {"preview_quality": 101},
        {"preview_max_fps": -1.0},
        # Both want a listening socket; the loser just is not there.
        {"preview_port": 9091, "metrics_port": 9091},
    ],
)
def test_invalid_preview_configurations_are_rejected(values):
    with pytest.raises(ConfigError):
        build_config(values)


def test_the_preview_is_off_by_default():
    """It serves live images of a person with no authentication. Anything but
    off by default would be wrong."""
    cfg = build_config()
    assert cfg.preview_port == 0
    # The tesselation, because all of its cost lands on an HTTP thread and
    # the lighter overlays read as scattered dots on a black background.
    assert cfg.preview_draw == "mesh"


def test_a_lan_reachable_preview_validates():
    cfg = build_config({"preview_port": 8080, "preview_bind": "0.0.0.0"})
    assert cfg.preview_port == 8080
    assert cfg.preview_bind == "0.0.0.0"


def test_the_preview_and_the_metrics_exporter_may_share_a_host():
    cfg = build_config({"preview_port": 8080, "metrics_port": 9091})
    assert cfg.preview_port != cfg.metrics_port


def test_the_preview_draws_on_black_by_default():
    """So enabling it serves geometry rather than a picture of a person, and
    showing the camera has to be asked for."""
    assert build_config().preview_background == "black"


def test_an_unknown_preview_background_is_rejected():
    with pytest.raises(ConfigError, match="preview_background"):
        build_config({"preview_background": "greenscreen"})


def test_an_unknown_preview_gaze_mode_is_refused():
    with pytest.raises(ConfigError, match="preview_gaze"):
        build_config({"preview_gaze": "telepathy"})
