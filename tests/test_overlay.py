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

import cv2  # noqa: E402

import gaze_monitor.overlay as overlay_mod  # noqa: E402
from gaze_monitor.overlay import render  # noqa: E402


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), np.uint8)


def _jpeg(data: bytes) -> bool:
    # SOI and EOI markers: proof it encoded rather than returned something.
    return data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


def test_a_frame_with_no_face_still_encodes():
    """An empty room is the normal state, not an error, and the preview has
    to keep showing the room."""
    assert _jpeg(render(_frame(), (480, 640), None))


def test_an_empty_landmark_array_is_treated_as_no_face():
    assert _jpeg(render(_frame(), (480, 640), np.zeros((0, 3), np.float32)))


def test_landmarks_outside_the_frame_are_clipped_not_crashed():
    """MediaPipe returns normalised coordinates that can fall outside [0, 1]
    when a face is half out of shot, which is exactly when someone is looking
    at the preview to find out why."""
    wild = (np.random.rand(478, 3).astype(np.float32) - 0.5) * 10
    assert _jpeg(render(_frame(), (480, 640), wild, "none"))


def test_the_gaze_landmarks_are_drawn_without_mediapipe():
    """The six points that actually drive the decision are drawn from indices
    this package owns, so "none" needs nothing but OpenCV."""
    frame = _frame()
    before = frame.copy()
    render(frame, (480, 640), np.full((478, 3), 0.5, np.float32), "none")
    assert not np.array_equal(frame, before), "nothing was drawn"


def test_quality_changes_the_encoded_size():
    lm = np.random.rand(478, 3).astype(np.float32)
    small = render(_frame(), (480, 640), lm, "none", quality=20)
    large = render(_frame(), (480, 640), lm, "none", quality=95)
    assert len(small) < len(large)


def test_rendering_draws_into_the_frame_it_is_given():
    """Documented behaviour, and load-bearing: the preview hands over a copy
    it already made, and a second one per frame would be waste."""
    frame = _frame()
    render(frame, (480, 640), np.full((478, 3), 0.5, np.float32), "none")
    assert frame.any(), "render did not touch the caller's frame"


@pytest.mark.parametrize("mode", ["irises", "contours", "mesh"])
def test_the_mediapipe_overlays_encode(mode):
    pytest.importorskip("mediapipe")
    lm = np.random.rand(478, 3).astype(np.float32)
    assert _jpeg(render(_frame(), (480, 640), lm, mode))


def test_the_heavier_overlays_actually_draw_more():
    """Encoding successfully is not the same as drawing something.

    The first version of the image-build assertion put every landmark at 0.5,
    which collapses all 2600 tesselation segments onto a single pixel. All
    four modes then returned byte-identical output and the check passed while
    proving nothing. Spread points and an ordering assertion is what catches
    an overlay that has quietly stopped drawing.
    """
    import cv2

    pytest.importorskip("mediapipe")
    points = (np.random.default_rng(0).random((478, 3)) * 0.6 + 0.2).astype(np.float32)

    def lit(mode):
        jpeg = render(None, (480, 640), points, mode)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return int((image.max(axis=2) > 25).sum())

    counts = {mode: lit(mode) for mode in ("none", "irises", "contours", "mesh")}
    assert counts["contours"] > counts["none"], counts
    assert counts["mesh"] > counts["contours"], counts
    assert counts["mesh"] > 5 * counts["none"], counts


def test_landmarks_stacked_on_one_pixel_do_not_look_like_a_mesh():
    """The degenerate case that fooled the first assertion, pinned so nobody
    writes a test on top of it again. Every point identical means every
    segment has zero length, so there is nothing to see whatever the mode."""
    import cv2

    stacked = np.full((478, 3), 0.5, np.float32)
    image = cv2.imdecode(
        np.frombuffer(render(None, (480, 640), stacked, "none"), np.uint8),
        cv2.IMREAD_COLOR,
    )
    lit = int((image.max(axis=2) > 25).sum())
    # Six overlapping dots at the centre, and nothing else anywhere.
    assert 0 < lit < 500, lit
    assert not image[:200].any(), "something was drawn away from the centre"


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


# -- drawing on black -------------------------------------------------------


def test_no_frame_draws_on_black():
    """The default, and the reason the endpoint is defensible: the camera
    image is never copied out of the capture loop, so there is no picture of
    anybody to serve even if somebody reaches the port."""
    jpeg = render(None, (480, 640), np.full((478, 3), 0.5, np.float32), "none")
    assert _jpeg(jpeg)


def test_the_black_canvas_is_the_size_it_was_told():
    import cv2

    decoded = cv2.imdecode(
        np.frombuffer(render(None, (240, 320), None), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape == (240, 320, 3)


def test_landmarks_land_where_they_were_put_and_nowhere_else():
    """Every landmark at 0.5 puts all six gaze points on the centre pixel, so
    the centre must be lit and a far corner must still be black. Comparing
    total brightness instead would be misleading: the "no face" caption lights
    more pixels than six small dots do."""
    import cv2

    drawn = cv2.imdecode(
        np.frombuffer(
            render(None, (480, 640), np.full((478, 3), 0.5, np.float32), "none"),
            np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )
    assert drawn[240, 320].any(), "the landmarks at the centre were not drawn"
    assert not drawn[470:, 600:].any(), "the background did not stay black"


# -- binocular markers -----------------------------------------------------
#
# The complaint these answer: in the preview the two irises moved
# independently and unnaturally, which real eyes do not do. They were
# MediaPipe's raw per-eye landmarks, inferred from separate eye crops with no
# binocular constraint in the model. The estimate had already averaged and
# smoothed them; the preview just was not showing it.


def _eyes(dx_left=0.0, dx_right=0.0, dy=0.0):
    """Landmarks with both eyes level and the irises placed by hand."""
    lms = np.zeros((478, 3), np.float32)
    lms[overlay_mod.LEFT_EYE_OUTER] = (0.40, 0.50, 0.0)
    lms[overlay_mod.LEFT_EYE_INNER] = (0.46, 0.50, 0.0)
    lms[overlay_mod.RIGHT_EYE_INNER] = (0.54, 0.50, 0.0)
    lms[overlay_mod.RIGHT_EYE_OUTER] = (0.60, 0.50, 0.0)
    width = 0.06
    lms[overlay_mod.LEFT_IRIS] = (0.43 + dx_left * width / 2, 0.50 + dy, 0.0)
    lms[overlay_mod.RIGHT_IRIS] = (0.57 + dx_right * width / 2, 0.50 + dy, 0.0)
    return lms


def _iris_marks(jpeg):
    """Where the bright iris markers landed, as (x, y) pixel columns/rows.

    Matched with a tolerance, because JPEG is lossy and an exact colour
    comparison finds nothing at any quality below 100.
    """
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    want = np.array(overlay_mod.IRIS_COLOUR, np.int16)
    mask = np.all(np.abs(image.astype(np.int16) - want) <= 50, axis=-1)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    left = xs < image.shape[1] // 2
    out = []
    for side in (left, ~left):
        if side.any():
            out.append((float(xs[side].mean()), float(ys[side].mean())))
    return out


def test_fused_markers_move_together_however_the_raw_irises_disagree():
    """The whole point. Two raw irises pulled hard in opposite directions
    still render as one shared estimate, because that is what the monitor
    acts on."""
    size = (480, 640)
    disagreeing = _eyes(dx_left=0.8, dx_right=-0.8)
    marks = _iris_marks(
        render(None, size, disagreeing, mode="none", quality=100, gaze=(0.0, 0.0))
    )
    assert len(marks) == 2
    left, right = marks
    # Both placed at their own eye centre, so each sits the same distance
    # inside its eye. Symmetric about the face midline, which is (w - 1) / 2
    # because that is the scaling the markers go through.
    midline = (size[1] - 1) / 2
    assert abs((midline - left[0]) - (right[0] - midline)) < 1.5
    assert abs(left[1] - right[1]) < 1.5, "a shared dy must put both at one height"


def test_the_fused_markers_follow_the_estimate_not_the_landmarks():
    size = (480, 640)
    still = _eyes()
    centred = _iris_marks(
        render(None, size, still, mode="none", quality=100, gaze=(0.0, 0.0))
    )
    looking = _iris_marks(
        render(None, size, still, mode="none", quality=100, gaze=(0.6, 0.0))
    )
    # Raw landmarks identical in both; only the estimate changed.
    assert centred[0][0] < looking[0][0]
    assert centred[1][0] < looking[1][0]
    moved_left = looking[0][0] - centred[0][0]
    moved_right = looking[1][0] - centred[1][0]
    assert abs(moved_left - moved_right) < 1.5, "eyes must move by equal amounts"


def test_raw_mode_still_shows_what_mediapipe_actually_said():
    """Diagnosing tracking needs the unfused landmarks, so the mode stays.

    Checked against the fused rendering of the same frame rather than against
    symmetry: two eyes disagreeing by equal and opposite amounts is
    convergence, and convergence is symmetric about the midline, so symmetry
    does not distinguish the two modes.
    """
    size = (480, 640)
    disagreeing = _eyes(dx_left=0.8, dx_right=-0.8)
    common = dict(mode="none", quality=100, gaze=(0.0, 0.0))
    fused = _iris_marks(render(None, size, disagreeing, **common))
    raw = _iris_marks(render(None, size, disagreeing, gaze_mode="raw", **common))
    assert len(raw) == 2
    # Fused ignores the landmarks and places both at the eye centres; raw
    # puts them where MediaPipe said. Those are different places.
    assert abs(raw[0][0] - fused[0][0]) > 10
    assert abs(raw[1][0] - fused[1][0]) > 10


def test_with_no_estimate_the_raw_landmarks_are_drawn_anyway():
    """No markers at all would read as a tracking failure when the truth is
    only that this frame had no usable sample."""
    marks = _iris_marks(
        render(None, (480, 640), _eyes(), mode="none", quality=100, gaze=None)
    )
    assert len(marks) == 2
