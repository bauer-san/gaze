"""Drawing the face mesh onto a frame, for the preview.

Split out from :mod:`gaze_monitor.preview` because this is the half that needs
OpenCV and MediaPipe, and the preview itself has to stay importable on a
machine with no vision stack. The preview holds this as an injected callable
and never imports it.

Everything here runs on an HTTP thread, never on the capture loop. Measured on
a Jetson Orin Nano at 640x480, which is where the defaults come from:

    mesh      2600 segments   4.1 ms
    contours   280 segments   0.4 ms
    irises       8 segments   0.01 ms
    jpeg encode               3.5-8 ms

Anti-aliasing costs more than the drawing does (about 12 ms on the tesselation
for a difference nobody can see on a moving image), so it is off.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .gaze import (
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_IRIS,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_IRIS,
)

log = logging.getLogger(__name__)

# BGR, chosen to read on black and on a camera image alike. The mesh is one
# colour and the landmarks that actually drive the gaze estimate are another,
# because the point of looking at this is to see whether those six are sitting
# where they should, not to admire the tesselation.
MESH_COLOUR = (0, 170, 0)
IRIS_COLOUR = (0, 255, 255)
CORNER_COLOUR = (255, 160, 0)
WARNING_COLOUR = (0, 0, 255)

_GAZE_LANDMARKS = (
    (LEFT_IRIS, IRIS_COLOUR, 4),
    (RIGHT_IRIS, IRIS_COLOUR, 4),
    (LEFT_EYE_OUTER, CORNER_COLOUR, 3),
    (LEFT_EYE_INNER, CORNER_COLOUR, 3),
    (RIGHT_EYE_INNER, CORNER_COLOUR, 3),
    (RIGHT_EYE_OUTER, CORNER_COLOUR, 3),
)

_CONNECTIONS: dict[str, np.ndarray] = {}


def _connections(mode: str) -> np.ndarray | None:
    """Index pairs for one overlay, built once and cached.

    Sorted rather than left as MediaPipe's frozenset, so the drawing order is
    the same on every run and a rendering difference means something changed.
    """
    if mode == "none":
        return None
    if mode in _CONNECTIONS:
        return _CONNECTIONS[mode]
    # Imported here, and only for the modes that need a connection set, so
    # that "none" and a frame with no face need nothing but OpenCV. That is
    # also what lets the drawing be tested without the whole vision stack.
    import mediapipe as mp

    sets = mp.solutions.face_mesh_connections
    source = {
        "mesh": sets.FACEMESH_TESSELATION,
        "contours": sets.FACEMESH_CONTOURS,
        "irises": sets.FACEMESH_IRISES,
    }.get(mode)
    index = None if source is None else np.array(sorted(source), dtype=np.int32)
    _CONNECTIONS[mode] = index
    return index


def extract(face_landmarks) -> np.ndarray:
    """MediaPipe's protobuf into a plain array of normalised (x, y, z).

    Called from the capture loop, so it is the one thing here that has to be
    cheap. Pulling the points out now is what lets the frame and the landmarks
    travel together into the preview's slot, and lets everything expensive
    happen on a thread that cannot slow the monitor down.
    """
    return np.array(
        [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark], dtype=np.float32
    )


def render(frame, size, landmarks, mode: str = "mesh", quality: int = 80) -> bytes:
    """Draw the overlay and encode to JPEG.

    ``frame`` is the camera image, or ``None`` to draw on black. None is the
    normal case: the preview only carries the pixels when it has been asked
    to show them, so by default no image of anyone ever leaves the capture
    loop and the endpoint serves geometry alone.

    When a frame is given it is drawn into in place. That is safe because the
    preview hands over a copy it made itself, and it saves a second one.
    """
    h, w = size
    if frame is None:
        frame = np.zeros((h, w, 3), np.uint8)

    if landmarks is None or len(landmarks) == 0:
        cv2.putText(
            frame,
            "no face",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WARNING_COLOUR,
            2,
        )
    else:
        points = np.clip(
            landmarks[:, :2] * np.array([w, h], dtype=np.float32),
            [0, 0],
            [w - 1, h - 1],
        ).astype(np.int32)

        index = _connections(mode)
        if index is not None and len(index) and index.max() < len(points):
            # One C-level call for the whole overlay. A Python loop over
            # cv2.line costs about the same as anti-aliasing, for nothing.
            cv2.polylines(frame, points[index], False, MESH_COLOUR, 1)

        for idx, colour, radius in _GAZE_LANDMARKS:
            if idx < len(points):
                cv2.circle(frame, tuple(points[idx]), radius, colour, -1)

    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:  # pragma: no cover - imencode failing on a valid BGR frame
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
