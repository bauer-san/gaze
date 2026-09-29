"""The live landmark preview.

Driven against a real HTTP server on a real socket, because the parts most
likely to be wrong are the parts a mock would paper over: the multipart
framing, whether a viewer disconnecting is noticed, and whether the thing that
must never happen -- a viewer slowing the capture loop -- actually cannot.

Rendering is injected, so none of this needs OpenCV.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from gaze_monitor.preview import BOUNDARY, ONESHOT_WARM_SECONDS, Preview, _Slot


def _cfg(**overrides):
    values = dict(
        preview_port=0,
        preview_bind="127.0.0.1",
        preview_quality=80,
        preview_max_fps=0.0,
        preview_draw="contours",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _Frame:
    """Stands in for a numpy frame: the preview only ever copies it."""

    def __init__(self, tag: str = "f") -> None:
        self.tag = tag
        self.copies = 0

    def copy(self):
        self.copies += 1
        clone = _Frame(self.tag)
        return clone


def _renderer(calls=None):
    def render(frame, landmarks, mode, quality):
        if calls is not None:
            calls.append((frame.tag, mode, quality))
        return b"JPEG:" + frame.tag.encode()

    return render


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -- staying out of the way ------------------------------------------------


def test_no_port_means_inert():
    p = Preview(_cfg(), renderer=_renderer())
    assert p.enabled is False
    assert p.start() is False
    assert p.wants_frames(1.0) is False
    p.stop()  # must not raise


def test_an_unwatched_preview_never_touches_the_frame():
    """The capture loop drives a safety output. A preview nobody is looking
    at has to cost it nothing at all."""
    p = Preview(_cfg(preview_port=1), renderer=_renderer())
    frame = _Frame()
    p.offer(frame, None, 100.0)
    assert frame.copies == 0
    assert p.frames_offered == 0


def test_a_preview_without_a_renderer_refuses_to_start(caplog):
    p = Preview(_cfg(preview_port=_free_port()))
    assert p.start() is False
    assert p.enabled is False
    assert "no renderer" in caplog.text


# -- the slot --------------------------------------------------------------


def test_the_slot_overwrites_rather_than_queueing():
    """A queue would let a slow viewer build a backlog and, worse, give it a
    way to push back on the thread producing frames."""
    slot = _Slot()
    for i in range(5):
        slot.put(i)
    seq, item = slot.latest()
    assert item == 4
    assert seq == 5


def test_waiting_returns_when_a_new_frame_lands():
    slot = _Slot()
    slot.put("first")
    seq, _ = slot.latest()
    threading.Timer(0.05, lambda: slot.put("second")).start()
    new_seq, item = slot.wait(seq, timeout=2.0)
    assert item == "second"
    assert new_seq != seq


def test_waiting_gives_up_on_timeout_rather_than_hanging():
    slot = _Slot()
    slot.put("only")
    seq, _ = slot.latest()
    started = time.monotonic()
    got_seq, item = slot.wait(seq, timeout=0.1)
    assert item is None
    assert got_seq == seq
    assert time.monotonic() - started < 1.0


def test_closing_releases_everyone_waiting():
    """Otherwise a stream thread blocks through shutdown, and the permit
    output's own shutdown queues behind it."""
    slot = _Slot()
    slot.put("x")
    seq, _ = slot.latest()
    threading.Timer(0.05, slot.close).start()
    _, item = slot.wait(seq, timeout=2.0)
    assert item is None


# -- encoding ---------------------------------------------------------------


def test_one_encode_serves_every_viewer():
    """Two browsers on the same frame must not cost two JPEG encodes."""
    calls = []
    p = Preview(_cfg(preview_port=1), renderer=_renderer(calls))
    p._viewers = 1
    p.offer(_Frame("a"), None, 100.0)
    first = p.next_jpeg(0, timeout=1.0)
    second = p.next_jpeg(0, timeout=1.0)
    assert first[1] == second[1] == b"JPEG:a"
    assert len(calls) == 1
    assert p.frames_encoded == 1


def test_the_renderer_is_given_the_configured_mode_and_quality():
    calls = []
    p = Preview(
        _cfg(preview_port=1, preview_draw="mesh", preview_quality=55),
        renderer=_renderer(calls),
    )
    p._viewers = 1
    p.offer(_Frame("a"), None, 100.0)
    p.next_jpeg(0, timeout=1.0)
    assert calls == [("a", "mesh", 55)]


def test_the_frame_is_copied_before_it_is_handed_over():
    """The capture backend reuses its buffer. Encoding a frame that is being
    overwritten underneath produces a torn image at best."""
    p = Preview(_cfg(preview_port=1), renderer=_renderer())
    p._viewers = 1
    frame = _Frame()
    p.offer(frame, None, 100.0)
    assert frame.copies == 1


# -- one-shot requests keep the pipeline warm -------------------------------


def test_a_snapshot_request_wakes_an_idle_pipeline():
    """With nobody streaming the loop stops offering frames, so a snapshot has
    to ask for one rather than serve whatever was last left lying around."""
    p = Preview(_cfg(preview_port=1), renderer=_renderer())
    assert p.wants_frames(time.monotonic()) is False

    result = {}
    threading.Thread(
        target=lambda: result.update(jpeg=p.snapshot(timeout=2.0)), daemon=True
    ).start()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not p.wants_frames(time.monotonic()):
        time.sleep(0.01)
    assert p.wants_frames(time.monotonic()) is True, "the request did not warm it"

    p.offer(_Frame("live"), None, time.monotonic())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and "jpeg" not in result:
        time.sleep(0.01)
    assert result["jpeg"] == b"JPEG:live"


def test_the_warm_period_expires():
    p = Preview(_cfg(preview_port=1), renderer=_renderer())
    p._warm()
    now = time.monotonic()
    assert p.wants_frames(now) is True
    assert p.wants_frames(now + ONESHOT_WARM_SECONDS + 1.0) is False


# -- over a real socket -----------------------------------------------------


@pytest.fixture
def served():
    """A preview on a real port, with a frame already in the slot."""
    p = Preview(
        _cfg(preview_port=_free_port(), preview_bind="127.0.0.1"),
        renderer=_renderer(),
    )
    assert p.start() is True
    base = f"http://127.0.0.1:{p.port}"
    try:
        yield p, base
    finally:
        p.stop()


def _feed(preview, tag="a", stop=None):
    """Keep offering frames, as the capture loop would."""

    def loop():
        while stop is None or not stop.is_set():
            preview.offer(_Frame(tag), [[0.1, 0.2, 0.3]], time.monotonic())
            time.sleep(0.02)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def test_the_index_page_is_served(served):
    _, base = served
    with urllib.request.urlopen(base + "/", timeout=5) as response:
        body = response.read().decode()
    assert response.status == 200
    assert "/stream.mjpg" in body


def test_an_unknown_path_is_a_404(served):
    _, base = served
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/nope", timeout=5)
    assert exc.value.code == 404


def test_a_snapshot_returns_a_jpeg(served):
    preview, base = served
    stop = threading.Event()
    _feed(preview, "snap", stop)
    try:
        with urllib.request.urlopen(base + "/frame.jpg", timeout=5) as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read() == b"JPEG:snap"
    finally:
        stop.set()


def test_landmarks_are_served_as_numbers(served):
    preview, base = served
    stop = threading.Event()
    _feed(preview, "lm", stop)
    try:
        with urllib.request.urlopen(base + "/landmarks.json", timeout=5) as response:
            payload = json.loads(response.read())
    finally:
        stop.set()
    assert payload["count"] == 1
    assert payload["points"] == [[0.1, 0.2, 0.3]]
    assert payload["age_seconds"] >= 0.0


def test_a_cold_pipeline_answers_503_rather_than_hanging(served):
    """Nothing is feeding it, so there is genuinely no frame. Saying so beats
    holding the connection open until somebody gives up."""
    _, base = served
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/frame.jpg", timeout=10)
    assert exc.value.code == 503


def test_the_stream_is_multipart_and_carries_frames(served):
    preview, base = served
    stop = threading.Event()
    _feed(preview, "vid", stop)
    try:
        response = urllib.request.urlopen(base + "/stream.mjpg", timeout=5)
        assert BOUNDARY in response.headers["Content-Type"]
        # Enough bytes for two parts, so the framing between them is covered.
        chunk = response.read(220)
        response.close()
    finally:
        stop.set()
    assert chunk.count(f"--{BOUNDARY}".encode()) >= 2
    assert b"Content-Type: image/jpeg" in chunk
    assert b"JPEG:vid" in chunk


def test_viewers_are_counted_and_released(served):
    preview, base = served
    stop = threading.Event()
    _feed(preview, "count", stop)
    try:
        assert preview.viewers == 0
        response = urllib.request.urlopen(base + "/stream.mjpg", timeout=5)
        response.read(80)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and preview.viewers == 0:
            time.sleep(0.01)
        assert preview.viewers == 1, "a connected stream was not counted"

        response.close()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and preview.viewers > 0:
            time.sleep(0.02)
        assert preview.viewers == 0, "a disconnected viewer was never released"
    finally:
        stop.set()


def test_the_capture_loop_is_not_blocked_by_a_viewer_that_never_reads():
    """The property the whole design exists for.

    A client that opens the stream and then stops reading must not slow the
    thread offering frames. The slot overwrites, so there is no queue to fill
    and no backpressure to apply.
    """
    preview = Preview(
        _cfg(preview_port=_free_port(), preview_bind="127.0.0.1"),
        renderer=_renderer(),
    )
    assert preview.start() is True
    try:
        # A raw socket that requests the stream and then never reads it.
        sock = socket.create_connection(("127.0.0.1", preview.port), timeout=5)
        sock.sendall(b"GET /stream.mjpg HTTP/1.0\r\n\r\n")

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and preview.viewers == 0:
            time.sleep(0.01)
        assert preview.viewers == 1

        started = time.monotonic()
        for _ in range(200):
            preview.offer(_Frame(), None, time.monotonic())
        elapsed = time.monotonic() - started

        assert preview.frames_offered == 200
        # 200 offers is 200 slot writes and 200 cheap copies. Anything near a
        # second means a viewer has managed to apply backpressure.
        assert elapsed < 1.0, f"offering frames took {elapsed:.2f}s"
        sock.close()
    finally:
        preview.stop()
