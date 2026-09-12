"""Import-level smoke tests.

The pure-logic modules must import without OpenCV or MediaPipe, so the test
suite (and CI) can run without the vision stack. The camera and UI modules are
checked only when that stack happens to be installed.
"""

import importlib

import pytest

PURE_MODULES = [
    "kinect_gaze",
    "kinect_gaze.attention",
    "kinect_gaze.calibration",
    "kinect_gaze.config",
    "kinect_gaze.gaze",
]


@pytest.mark.parametrize("name", PURE_MODULES)
def test_pure_modules_import_without_opencv(name):
    assert importlib.import_module(name)


def test_demo_cli_builds_without_opencv():
    """--help and --factory-reset must work on a box with no vision stack."""
    demo = importlib.import_module("demo")
    args = demo.build_parser().parse_args(["--source", "webcam"])
    assert args.source == "webcam"
    assert args.factory_reset is False


@pytest.mark.parametrize("name", ["kinect_gaze.capture", "kinect_gaze.ui"])
def test_vision_modules_import_when_stack_present(name):
    pytest.importorskip("cv2")
    assert importlib.import_module(name)


def test_every_source_name_builds_a_backend():
    """A name offered by the CLI must be one create_source can actually build."""
    pytest.importorskip("cv2")
    capture = importlib.import_module("kinect_gaze.capture")
    from kinect_gaze.config import SOURCE_NAMES

    for name in SOURCE_NAMES:
        source = capture.create_source(name)
        assert isinstance(source, capture.CameraSource)
    assert isinstance(
        capture.create_source("gst:fakesrc ! appsink"), capture.CameraSource
    )


def test_unknown_source_raises():
    pytest.importorskip("cv2")
    capture = importlib.import_module("kinect_gaze.capture")
    with pytest.raises(capture.UnknownSourceError):
        capture.create_source("not-a-camera")


def test_gstreamer_support_detection_returns_bool():
    """Parsed out of cv2.getBuildInformation(), whose column widths vary."""
    pytest.importorskip("cv2")
    from kinect_gaze.capture import has_gstreamer_support

    assert isinstance(has_gstreamer_support(), bool)
