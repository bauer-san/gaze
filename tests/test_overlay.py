"""Drawing the mesh onto a frame.

Only the parts that need MediaPipe's connection sets are skipped when it is
absent, which on a CI runner is all of them but the ones below. What is
covered here is the arithmetic that turns normalised landmarks into pixels,
because that is where an off-by-one becomes a crash on a real face rather
than on a synthetic one.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from gaze_monitor.overlay import render  # noqa: E402


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), np.uint8)


def _jpeg(data: bytes) -> bool:
    # SOI and EOI markers: proof it encoded rather than returned something.
    return data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


def test_a_frame_with_no_face_still_encodes():
    """An empty room is the normal state, not an error, and the preview has
    to keep showing the room."""
    assert _jpeg(render(_frame(), None))


def test_an_empty_landmark_array_is_treated_as_no_face():
    assert _jpeg(render(_frame(), np.zeros((0, 3), np.float32)))


def test_landmarks_outside_the_frame_are_clipped_not_crashed():
    """MediaPipe returns normalised coordinates that can fall outside [0, 1]
    when a face is half out of shot, which is exactly when someone is looking
    at the preview to find out why."""
    wild = (np.random.rand(478, 3).astype(np.float32) - 0.5) * 10
    assert _jpeg(render(_frame(), wild, "none"))


def test_the_gaze_landmarks_are_drawn_without_mediapipe():
    """The six points that actually drive the decision are drawn from indices
    this package owns, so "none" needs nothing but OpenCV."""
    frame = _frame()
    before = frame.copy()
    render(frame, np.full((478, 3), 0.5, np.float32), "none")
    assert not np.array_equal(frame, before), "nothing was drawn"


def test_quality_changes_the_encoded_size():
    lm = np.random.rand(478, 3).astype(np.float32)
    small = render(_frame(), lm, "none", quality=20)
    large = render(_frame(), lm, "none", quality=95)
    assert len(small) < len(large)


def test_rendering_draws_into_the_frame_it_is_given():
    """Documented behaviour, and load-bearing: the preview hands over a copy
    it already made, and a second one per frame would be waste."""
    frame = _frame()
    render(frame, np.full((478, 3), 0.5, np.float32), "none")
    assert frame.any(), "render did not touch the caller's frame"


@pytest.mark.parametrize("mode", ["irises", "contours", "mesh"])
def test_the_mediapipe_overlays_encode(mode):
    pytest.importorskip("mediapipe")
    lm = np.random.rand(478, 3).astype(np.float32)
    assert _jpeg(render(_frame(), lm, mode))


def test_extract_pulls_normalised_points_out_of_the_protobuf():
    from gaze_monitor.overlay import extract

    class _Landmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Face:
        landmark = [_Landmark(0.1, 0.2, 0.3), _Landmark(0.4, 0.5, 0.6)]

    points = extract(_Face())
    assert points.shape == (2, 3)
    assert points.dtype == np.float32
    assert points[1].tolist() == pytest.approx([0.4, 0.5, 0.6])
