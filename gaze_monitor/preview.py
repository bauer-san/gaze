"""Live landmark preview over HTTP. A demonstration and commissioning aid.

Off unless ``preview_port`` is set, and deliberately so: this serves live
images of whoever is standing in front of the camera, to anyone who can reach
the port, with no authentication of any kind. It exists to prove during
commissioning that the mesh is tracking, and to put the tracking on a second
screen during a demonstration. It is not something to leave running on an
installed machine.

The design constraint that shapes everything here is that **a viewer must not
be able to slow the monitor down**. The capture loop drives an attention state
that can stop a machine, and a browser on a bad link must not get near it. So
the loop does no drawing and no encoding. It copies the frame and the landmark
array into a single slot that overwrites, and every expensive thing happens on
an HTTP thread:

* copying a 640x480 frame costs about 0.08 ms
* encoding one as JPEG costs about 3 to 8 ms

Which is why one of those happens in the loop and the other does not. When
nobody is watching, ``wants_frames`` is false and not even the copy happens.

A slow client drops frames. The slot holds one frame and overwrites it, so
there is no queue to grow and no backpressure to apply -- a viewer that cannot
keep up simply sees a less smooth picture, which is the correct thing to
sacrifice.

Rendering is injected rather than imported. Drawing landmarks needs OpenCV,
and this module has to stay importable -- and testable -- on a machine with no
vision stack, like every CI runner this project has.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

# Overlays, cheapest first. The full tesselation is roughly 2600 line
# segments and is the one that costs real time; contours and irises are a
# small fraction of it and are what anyone actually looks at.
DRAW_MODES = ("none", "irises", "contours", "mesh")

# What the landmarks are drawn on. "black" is the default and means the
# camera image is never copied out of the capture loop at all, so the
# endpoint serves geometry and nothing else. "camera" puts the mesh over the
# live picture, which is what you want when the question is why tracking is
# poor -- backlight, framing, a lens somebody has leaned a board against.
BACKGROUNDS = ("black", "camera")

# How long a one-shot request (a snapshot, or the JSON) keeps the pipeline
# warm. Without this the loop stops offering frames the moment the last
# stream disconnects, and /frame.jpg would answer with whatever was last
# left in the slot, or nothing at all.
ONESHOT_WARM_SECONDS = 2.0

BOUNDARY = "gazeframe"

PAGE = """<!doctype html>
<title>Gaze preview</title>
<style>
  body { margin:0; background:#111; color:#ccc;
         font:14px system-ui,sans-serif; text-align:center }
  img  { max-width:100%; height:auto; display:block; margin:0 auto }
  p    { margin:.6rem }
</style>
<img src="/stream.mjpg" alt="live view">
<p>live landmark preview &middot; <a style="color:#6af" href="/frame.jpg">snapshot</a>
&middot; <a style="color:#6af" href="/landmarks.json">landmarks</a></p>
"""


class _Slot:
    """One frame, overwritten in place, with a sequence number to wait on.

    Deliberately not a queue. A queue would let a slow viewer accumulate a
    backlog of stale frames and, worse, give it a way to apply backpressure
    to the thread that produces them.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item = None
        self._seq = 0
        self._closed = False

    def put(self, item) -> None:
        with self._cond:
            self._item = item
            self._seq += 1
            self._cond.notify_all()

    def wait(self, since: int, timeout: float):
        """Block until the sequence moves past ``since``. Returns (seq, item).

        ``(since, None)`` means the timeout expired or the slot closed, which
        the caller treats the same way: stop waiting and give up the thread.
        """
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._closed or self._seq != since, timeout
            ):
                return since, None
            if self._closed:
                return since, None
            return self._seq, self._item

    def latest(self):
        with self._cond:
            return self._seq, self._item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.0: each response closes the connection, which is what we want
    # for a stream that only ends when the viewer goes away.
    protocol_version = "HTTP/1.0"

    @property
    def preview(self) -> Preview:
        return self.server.preview  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # noqa: A003 - BaseHTTPRequestHandler API
        # The default writes every request to stderr, which on a demo machine
        # buries the state transitions anyone is actually watching for.
        log.debug("preview: %s", fmt % args)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/stream.mjpg":
                self._stream()
            elif path == "/frame.jpg":
                self._snapshot()
            elif path == "/landmarks.json":
                self._landmarks()
            else:
                self._send(b"not found\n", "text/plain", status=404)
        except (BrokenPipeError, ConnectionResetError):
            # The viewer closed the tab. Entirely normal, and not worth a
            # traceback on a machine somebody is demonstrating.
            pass

    # -- routes ------------------------------------------------------------

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        preview = self.preview
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        preview.viewer_joined(self.client_address[0])
        seq = 0
        try:
            while not preview.stopping:
                seq, jpeg = preview.next_jpeg(seq, timeout=1.0)
                if jpeg is None:
                    continue
                self.wfile.write(
                    b"--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                    % (BOUNDARY.encode(), len(jpeg))
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                preview.pace()
        finally:
            preview.viewer_left(self.client_address[0])

    def _snapshot(self) -> None:
        jpeg = self.preview.snapshot(timeout=2.0)
        if jpeg is None:
            self._send(b"no frame yet\n", "text/plain", status=503)
            return
        self._send(jpeg, "image/jpeg")

    def _landmarks(self) -> None:
        body = self.preview.landmarks_json(timeout=2.0)
        if body is None:
            self._send(b'{"error":"no frame yet"}\n', "application/json", status=503)
            return
        self._send(body, "application/json")


class _Server(ThreadingHTTPServer):
    # Viewers that wedge must not hold up process exit; the permit output
    # shutting down promptly matters more than a tidy disconnect.
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, preview: Preview) -> None:
        self.preview = preview
        super().__init__(address, handler)


class Preview:
    """Serves the live landmark view. Inert unless ``preview_port`` is set.

    ``offer`` is called from the capture loop and is the only method that runs
    there. It is cheap on purpose, and it does nothing at all unless somebody
    is actually watching.
    """

    def __init__(self, config, renderer=None) -> None:
        self.port = int(getattr(config, "preview_port", 0) or 0)
        self.bind = getattr(config, "preview_bind", "0.0.0.0") or "0.0.0.0"
        self.quality = int(getattr(config, "preview_quality", 80))
        self.max_fps = float(getattr(config, "preview_max_fps", 10.0))
        self.draw = getattr(config, "preview_draw", "contours")
        self.background = getattr(config, "preview_background", "black")
        self.enabled = self.port > 0

        self._renderer = renderer
        self._slot = _Slot()
        self._lock = threading.Lock()
        self._viewers = 0
        self._wanted_until = 0.0
        self._server = None
        self._thread: threading.Thread | None = None
        self.stopping = False

        # One encode per frame however many viewers there are.
        self._encoded_seq = -1
        self._encoded: bytes | None = None

        self.frames_offered = 0
        self.frames_encoded = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Bring the server up. False means it stayed inert."""
        if not self.enabled:
            return False
        if self._renderer is None:
            log.warning(
                "preview_port is %d but no renderer was supplied; "
                "the preview will not start",
                self.port,
            )
            self.enabled = False
            return False
        if self.draw not in DRAW_MODES:
            log.warning("Unknown preview_draw %r; falling back to contours", self.draw)
            self.draw = "contours"
        if self.background not in BACKGROUNDS:
            log.warning(
                "Unknown preview_background %r; falling back to black",
                self.background,
            )
            self.background = "black"

        try:
            self._server = _Server((self.bind, self.port), _Handler, self)
        except OSError as exc:
            log.error(
                "Could not start the preview on %s:%d: %s", self.bind, self.port, exc
            )
            self.enabled = False
            return False

        self._thread = threading.Thread(
            target=self._server.serve_forever, name="gaze-preview", daemon=True
        )
        self._thread.start()
        where = self.bind if self.bind != "0.0.0.0" else "<this host>"
        if self.background == "camera":
            log.warning(
                "Preview serving LIVE CAMERA IMAGES on http://%s:%d/ with no "
                "authentication. A demonstration aid, not something to leave "
                "running on an installed machine.",
                where,
                self.port,
            )
        else:
            log.warning(
                "Preview serving landmarks on http://%s:%d/ with no "
                "authentication. No camera image is sent (preview_background "
                "is %r). A demonstration aid, not something to leave running "
                "on an installed machine.",
                where,
                self.port,
                self.background,
            )
        return True

    def stop(self) -> None:
        if not self.enabled or self._server is None:
            return
        self.stopping = True
        self._slot.close()
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- from the capture loop ---------------------------------------------

    def wants_frames(self, now: float) -> bool:
        """True when something is watching, or has just asked.

        Checked before the frame is copied, so an unwatched preview costs the
        capture loop one comparison per frame and nothing else.
        """
        if not self.enabled:
            return False
        with self._lock:
            return self._viewers > 0 or now < self._wanted_until

    def offer(self, frame, landmarks, now: float) -> None:
        """Hand over the newest frame. Called from the capture loop.

        On the default black background the pixels are not taken at all, only
        the frame's size. That is the cheaper path and the more defensible
        one: no image of anybody is copied, held in a buffer, or reachable
        over the network.

        When the camera image is wanted it is copied, because the capture
        backend reuses its buffer and an HTTP thread encoding a frame that is
        being overwritten underneath it produces a torn image at best.
        """
        if not self.wants_frames(now):
            return
        image = frame.copy() if self.background == "camera" else None
        self._slot.put((image, frame.shape[:2], landmarks, now))
        self.frames_offered += 1

    # -- from the HTTP threads ---------------------------------------------

    def viewer_joined(self, who: str) -> None:
        with self._lock:
            self._viewers += 1
            count = self._viewers
        log.info("Preview viewer connected from %s (%d watching)", who, count)

    def viewer_left(self, who: str) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)
            count = self._viewers
        log.info("Preview viewer from %s disconnected (%d watching)", who, count)

    @property
    def viewers(self) -> int:
        with self._lock:
            return self._viewers

    def _warm(self) -> None:
        """Keep the loop offering frames for a moment after a one-shot GET."""
        with self._lock:
            self._wanted_until = time.monotonic() + ONESHOT_WARM_SECONDS

    def pace(self) -> None:
        """Hold a stream to ``preview_max_fps``."""
        if self.max_fps > 0:
            time.sleep(1.0 / self.max_fps)

    def _encode(self, seq: int, item) -> bytes | None:
        """Render and encode, reusing the result across concurrent viewers."""
        if item is None:
            return None
        with self._lock:
            if self._encoded_seq == seq and self._encoded is not None:
                return self._encoded
        frame, size, landmarks, _ = item
        jpeg = self._renderer(frame, size, landmarks, self.draw, self.quality)
        with self._lock:
            self._encoded_seq = seq
            self._encoded = jpeg
            self.frames_encoded += 1
        return jpeg

    def next_jpeg(self, since: int, timeout: float = 1.0):
        """Wait for a frame newer than ``since`` and return (seq, jpeg)."""
        seq, item = self._slot.wait(since, timeout)
        if item is None:
            return seq, None
        return seq, self._encode(seq, item)

    def snapshot(self, timeout: float = 2.0) -> bytes | None:
        """One frame, waiting for a fresh one if the pipeline was cold."""
        self._warm()
        seq, item = self._slot.latest()
        if item is None:
            seq, item = self._slot.wait(seq, timeout)
        return self._encode(seq, item) if item is not None else None

    def landmarks_json(self, timeout: float = 2.0) -> bytes | None:
        """The points as numbers, for anything that computes rather than looks."""
        self._warm()
        seq, item = self._slot.latest()
        if item is None:
            seq, item = self._slot.wait(seq, timeout)
        if item is None:
            return None
        _, _, landmarks, captured_at = item
        points = [] if landmarks is None else _as_list(landmarks)
        return json.dumps(
            {
                "frame": seq,
                "count": len(points),
                "age_seconds": round(max(0.0, time.monotonic() - captured_at), 3),
                # Normalised to the frame: x and y in [0, 1], z relative to
                # the head centre in the same scale as x.
                "points": points,
            }
        ).encode("utf-8")


def _as_list(landmarks) -> list:
    """Accept a numpy array or anything list-like, without importing numpy."""
    tolist = getattr(landmarks, "tolist", None)
    return tolist() if callable(tolist) else [list(p) for p in landmarks]
