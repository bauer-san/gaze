"""The saw, the room it stands in, the camera mount and the attention zone.

Dimensions are from the 14" floor-standing meat band saw general-arrangement
drawing that accompanied this work (front, side, top and isometric views at
1:8). Only gross dimensions are used and nothing of the drawing itself is
reproduced:

======================================  ==========
Overall height, floor to head top       1.80 m (70.96 in)
Table surface                           0.915 m
Underside of the upper wheel housing    1.34 m
Upper housing width                     0.48 m
Fixed table plus sliding carriage       about 0.9 m wide, 0.7 m deep
======================================  ==========

As the operator sees it: the cutting side of the blade runs down the left
of the head, the column carrying the blade back up is on the right, and the
sliding carriage is to the left of the blade. The operator stands at the
front edge of the table, pushing product through the blade from the left.

World frame, metres: origin on the floor directly below the blade, ``+x`` to
the operator's right, ``+y`` up, ``+z`` towards the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import materials as M
from .mathutil import normalize
from .meshes import box, prism, quad, tube, uv_sphere

TABLE_Y = 0.915
HEAD_BOTTOM_Y = 1.34
HEAD_TOP_Y = 1.80
TABLE_FRONT_Z = 0.36
# Upper housing, from the drawing: 0.47 m wide, from 47 mm left of the blade;
# 0.15 m deep, its front face 60 mm ahead of the teeth.
HEAD_LEFT_X = -0.047
HEAD_RIGHT_X = 0.427
HEAD_FRONT_Z = 0.06
HEAD_BACK_Z = -0.087
GUIDE_BOTTOM_Y = 1.09
# Median position of the default operator's eyes while watching the cut:
# what a camera bracket is aimed at when it is set up.
OPERATOR_EYES = (-0.023, 1.572, 0.425)
# Where the camera is aimed: the bridge of the nose, a little below the eyes,
# which centres the whole face in the frame rather than the eyes. FaceMesh
# needs the chin and forehead too, and this leaves about the same margin
# above and below for the operator to move, and for the mount to tip.
OPERATOR_FACE = (-0.023, 1.545, 0.425)
POST_X = 0.012
POST_Z = -0.108
BLADE_TEETH_Z = 0.010


@dataclass(frozen=True)
class Zone:
    """A planar rectangle in the world: the region the operator must watch.

    Corners in the calibration routine's order: top-left, top-right,
    bottom-left, bottom-right, *as the operator sees them*. ``u`` runs left to
    right and ``v`` top to bottom, both 0..1 inside, which is the same
    convention as :meth:`gaze_monitor.attention.AttentionZone.project`.
    """

    top_left: tuple
    top_right: tuple
    bottom_left: tuple
    bottom_right: tuple

    @property
    def corners(self) -> np.ndarray:
        return np.array(
            [self.top_left, self.top_right, self.bottom_left, self.bottom_right]
        )

    def point(self, u: float, v: float) -> np.ndarray:
        tl, tr, bl, br = self.corners
        top = tl + (tr - tl) * u
        bottom = bl + (br - bl) * u
        return top + (bottom - top) * v

    def project_ray(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> tuple[float, float]:
        """(u, v) where a ray meets the zone's plane; nan if it never does."""
        tl, tr, bl, _ = self.corners
        eu = tr - tl
        ev = bl - tl
        n = np.cross(eu, ev)
        denom = float(np.dot(n, direction))
        if abs(denom) < 1e-9:
            return float("nan"), float("nan")
        t = float(np.dot(n, tl - origin)) / denom
        if t <= 0:
            return float("nan"), float("nan")
        p = origin + t * np.asarray(direction) - tl
        # Solve p = u * eu + v * ev in the plane (least squares is exact here).
        A = np.stack([eu, ev], axis=1)
        (u, v), *_ = np.linalg.lstsq(A, p, rcond=None)
        return float(u), float(v)


# "The blade and the cut immediately around it": from the bottom of the upper
# blade guide down the blade to the table, and the strip of table in front of
# the teeth where the product is being cut. Tilted, because the operator
# looks down at it at about 45 degrees.
BLADE_ZONE = Zone(
    top_left=(-0.12, 1.07, 0.0),
    top_right=(0.12, 1.07, 0.0),
    bottom_left=(-0.12, TABLE_Y, 0.15),
    bottom_right=(0.12, TABLE_Y, 0.15),
)


@dataclass
class CameraMount:
    """Where the camera is and what it points at.

    The default is the mount the placement study recommends (see
    :mod:`.placement` and the README): on a short bracket off the front-left
    corner of the upper wheel housing, as the operator sees it. That puts it

    * at the housing's left edge, 50 mm to the operator's left of the blade,
      nearly in line with the operator's face, so it sees the face within
      about 14 degrees of head-on while they watch the cut, within a couple
      of degrees of the best position side to side;
    * with its centre 25 mm above the housing's underside (53.7 in from the
      floor), so the camera body stays clear of the product, which passes
      beneath the housing at up to the saw's maximum cut height;
    * 20 mm proud of the housing's front face. The operator leans over the
      table with their face only about 0.4 m from here, and at that range
      the face nearly fills the D435's 42-degree vertical field: further
      forward is more head-on but leaves too little margin for the chin and
      forehead to stay in frame as the operator moves;
    * tilted up about 27 degrees, aimed at the bridge of the operator's nose
      so the whole face is centred in the frame.

    Every option above the housing's underside looks *down* onto the upper
    eyelid, because the operator's line of sight to the cut runs well below
    it. That is the main cost of any mount the product clearance allows.
    """

    position: tuple = (-0.05, HEAD_BOTTOM_Y + 0.025, 0.08)
    target: tuple = OPERATOR_FACE
    width: int = 1280
    height: int = 720
    # D435i colour sensor at 1280x720.
    hfov_deg: float = 70.4
    vfov_deg: float = 43.3
    # Rotation of the camera about its own optical axis, degrees.
    roll_deg: float = 0.0


@dataclass
class Scene:
    """Static geometry plus named gaze targets."""

    zone: Zone = BLADE_ZONE
    mount: CameraMount = field(default_factory=CameraMount)
    product_x: float = -0.06

    def targets(self) -> dict[str, np.ndarray]:
        """Named world points the operator looks at."""
        return {
            "cut": np.array([0.0, 0.965, 0.03]),
            "blade_guide": np.array([0.0, 1.05, 0.0]),
            "product": np.array([-0.17, 0.98, 0.10]),
            "carriage": np.array([-0.40, 0.95, 0.28]),
            "tray": np.array([0.62, 0.93, 0.30]),
            "controls": np.array([0.37, 1.12, 0.046]),
            "camera": np.asarray(self.mount.position, float),
            "coworker_left": np.array([-2.0, 1.62, 0.45]),
            "coworker_right": np.array([2.0, 1.60, 0.2]),
            "floor": np.array([0.10, 0.0, 0.55]),
            "clock": np.array([-1.2, 2.35, -1.8]),
            "head_housing": np.array([0.17, 1.62, 0.07]),
        }

    def static_meshes(self) -> list:
        """The saw, the furniture and the room. Built once per renderer."""
        meshes = []
        add = meshes.append
        # Room. The back wall is behind the operator: it is what the camera
        # sees around the operator's head.
        add(
            quad(
                [(-4, 0, -2.5), (4, 0, -2.5), (4, 0, 4.0), (-4, 0, 4.0)][::-1],
                M.FLOOR,
                24,
            )
        )
        add(
            quad([(-4, 0, 4.0), (4, 0, 4.0), (4, 3.2, 4.0), (-4, 3.2, 4.0)], M.WALL, 12)
        )
        add(
            quad(
                [(-4, 0, -2.5), (-4, 0, 4.0), (-4, 3.2, 4.0), (-4, 3.2, -2.5)],
                M.WALL,
                12,
            )
        )
        add(
            quad([(4, 0, 4.0), (4, 0, -2.5), (4, 3.2, -2.5), (4, 3.2, 4.0)], M.WALL, 12)
        )
        add(
            quad(
                [(-4, 3.2, 4.0), (4, 3.2, 4.0), (4, 3.2, -2.5), (-4, 3.2, -2.5)],
                M.WALL,
                12,
            )
        )
        for lx in (-1.5, 1.5):
            for lz in (0.5, 2.5):
                add(
                    box((lx - 0.6, 3.17, lz - 0.15), (lx + 0.6, 3.2, lz + 0.15), M.LAMP)
                )
        # Doorway and a stainless rack on the back wall, so the background is
        # not a featureless plane.
        add(box((-2.2, 0.0, 3.95), (-1.2, 2.1, 4.0), M.DARK))
        add(box((0.8, 0.0, 3.4), (2.2, 1.8, 3.9), M.METAL, 2))
        for sy in (0.4, 0.9, 1.4):
            add(box((0.8, sy, 3.35), (2.2, sy + 0.03, 3.9), M.METAL))

        # The saw. Gross dimensions from the drawing; detail after the
        # Hi-Yield 14's published features (peaked upper housing, swing-open
        # stainless doors). No markings are reproduced.
        meshes.extend(saw_meshes())
        # Product tray to the operator's right.
        add(box((0.50, 0.85, 0.05), (1.10, 0.88, 0.65), M.METAL, 2))
        add(box((0.60, 0.88, 0.15), (1.0, 0.95, 0.55), M.APRON, 2))
        for lx in (0.52, 1.08):
            for lz in (0.07, 0.63):
                add(
                    tube(
                        (lx, 0.0, lz),
                        (lx, 0.85, lz),
                        0.015,
                        material=M.METAL,
                        segments=8,
                    )
                )
        return meshes


def saw_meshes() -> list:
    """The band saw itself, in the world frame."""
    meshes = []
    add = meshes.append
    T = TABLE_Y

    # Upper wheel housing, with the peaked top that sheds washdown water.
    peak_z = (HEAD_FRONT_Z + HEAD_BACK_Z) / 2
    add(
        prism(
            [
                (HEAD_BACK_Z, HEAD_BOTTOM_Y),
                (HEAD_FRONT_Z, HEAD_BOTTOM_Y),
                (HEAD_FRONT_Z, HEAD_TOP_Y - 0.06),
                (peak_z, HEAD_TOP_Y),
                (HEAD_BACK_Z, HEAD_TOP_Y - 0.06),
            ],
            HEAD_LEFT_X,
            HEAD_RIGHT_X,
            M.METAL,
        )
    )
    # Door seam, hinges and latch on the front face.
    f = HEAD_FRONT_Z
    for y0, y1, x0, x1 in (
        (HEAD_BOTTOM_Y + 0.012, HEAD_BOTTOM_Y + 0.016, HEAD_LEFT_X + 0.01, 0.32),
        (
            HEAD_TOP_Y - 0.074,
            HEAD_TOP_Y - 0.07,
            HEAD_LEFT_X + 0.01,
            HEAD_RIGHT_X - 0.01,
        ),
    ):
        add(box((x0, y0, f), (x1, y1, f + 0.002), M.DARK))
    for hy in (1.45, 1.66):
        add(
            box(
                (HEAD_RIGHT_X - 0.004, hy - 0.03, f - 0.03),
                (HEAD_RIGHT_X + 0.012, hy + 0.03, f),
                M.METAL,
            )
        )
    add(
        tube(
            (HEAD_LEFT_X, 1.63, 0.01),
            (HEAD_LEFT_X - 0.05, 1.63, 0.01),
            0.008,
            material=M.METAL,
            segments=8,
        )
    )
    add(
        box((HEAD_LEFT_X - 0.06, 1.60, 0.0), (HEAD_LEFT_X - 0.045, 1.66, 0.02), M.METAL)
    )
    # Column carrying the blade's return run, on the operator's right.
    add(box((0.32, T, HEAD_BACK_Z), (HEAD_RIGHT_X, HEAD_BOTTOM_Y, 0.04), M.METAL, 2))
    add(box((0.318, T + 0.02, 0.0), (0.322, HEAD_BOTTOM_Y - 0.02, 0.04), M.DARK))

    # Lower cabinet with its door, legs and levelling feet.
    add(box((-0.045, 0.16, -0.30), (0.40, T - 0.035, 0.24), M.METAL, 3))
    add(box((0.0, 0.20, 0.24), (0.36, T - 0.08, 0.243), M.METAL))
    add(box((0.30, 0.48, 0.243), (0.33, 0.58, 0.255), M.DARK))
    for lx in (-0.025, 0.38):
        for lz in (-0.28, 0.22):
            add(
                box(
                    (lx - 0.016, 0.03, lz - 0.016),
                    (lx + 0.016, 0.17, lz + 0.016),
                    M.METAL,
                )
            )
            add(
                tube(
                    (lx, 0.0, lz),
                    (lx, 0.03, lz),
                    0.012,
                    material=M.METAL,
                    segments=8,
                    caps=False,
                )
            )
            add(uv_sphere((lx, 0.008, lz), (0.045, 0.045, 0.012), 4, 14, M.DARK))

    # Fixed table to the right of the blade, and its apron.
    add(box((-0.005, T - 0.02, -0.35), (0.43, T, TABLE_FRONT_Z), M.METAL, 3))
    add(
        box(
            (-0.045, T - 0.08, TABLE_FRONT_Z - 0.03),
            (0.43, T - 0.02, TABLE_FRONT_Z),
            M.METAL,
            2,
        )
    )

    # Sliding carriage to the left: a cutting board on a stainless frame,
    # riding on two round rails, with a hand rail along the front.
    add(box((-0.56, T, -0.32), (-0.008, T + 0.02, TABLE_FRONT_Z), M.BOARD, 3))
    add(box((-0.57, T - 0.03, -0.33), (-0.006, T, TABLE_FRONT_Z + 0.01), M.METAL, 2))
    for rx in (-0.50, -0.07):
        add(
            tube(
                (rx, T - 0.06, -0.42),
                (rx, T - 0.06, TABLE_FRONT_Z + 0.06),
                0.014,
                material=M.METAL,
                segments=10,
            )
        )
    add(box((-0.53, T - 0.10, -0.40), (-0.04, T - 0.07, -0.37), M.METAL))
    add(
        box(
            (-0.53, T - 0.10, TABLE_FRONT_Z + 0.03),
            (-0.04, T - 0.07, TABLE_FRONT_Z + 0.06),
            M.METAL,
        )
    )
    for hx in (-0.53, -0.08):
        add(
            tube(
                (hx, T + 0.02, TABLE_FRONT_Z + 0.01),
                (hx, T + 0.07, TABLE_FRONT_Z + 0.05),
                0.009,
                material=M.METAL,
                segments=8,
            )
        )
    add(
        tube(
            (-0.53, T + 0.07, TABLE_FRONT_Z + 0.05),
            (-0.08, T + 0.07, TABLE_FRONT_Z + 0.05),
            0.012,
            material=M.METAL,
            segments=10,
        )
    )
    # Pusher plate (end gauge) behind the product, with its lever handle.
    add(box((-0.30, T + 0.02, 0.16), (-0.03, T + 0.16, 0.175), M.METAL))
    add(
        tube(
            (-0.16, T + 0.12, 0.175),
            (-0.16, T + 0.15, 0.27),
            0.009,
            material=M.METAL,
            segments=8,
        )
    )
    add(
        tube(
            (-0.22, T + 0.15, 0.27),
            (-0.08, T + 0.15, 0.27),
            0.013,
            material=M.DARK,
            segments=10,
        )
    )

    # Blade, upper guide and guide post, fitted to the drawing. The guide
    # is about 46 mm wide and stands about 20 mm proud of the teeth; the
    # post runs up behind the housing, about 110 mm behind the teeth, and
    # carries the guide on an arm with a star knob (here at 47.8 in, but
    # it moves with the guide).
    add(
        box(
            (-0.0006, T + 0.02, -0.006), (0.0006, HEAD_BOTTOM_Y, BLADE_TEETH_Z), M.BLADE
        )
    )
    add(box((-0.022, GUIDE_BOTTOM_Y, -0.03), (0.024, HEAD_BOTTOM_Y, 0.03), M.METAL))
    add(box((-0.012, 1.20, POST_Z - 0.012), (0.03, 1.228, -0.03), M.METAL))
    add(uv_sphere((-0.03, 1.214, POST_Z), (0.02, 0.012, 0.02), 6, 10, M.DARK))
    add(
        tube(
            (POST_X, 1.15, POST_Z),
            (POST_X, HEAD_TOP_Y + 0.15, POST_Z),
            0.011,
            material=M.METAL,
            segments=8,
        )
    )
    for by in (1.44, 1.67):  # the two post brackets
        add(
            box(
                (POST_X - 0.025, by - 0.02, POST_Z - 0.02),
                (POST_X + 0.02, by + 0.02, HEAD_BACK_Z),
                M.METAL,
            )
        )

    # Product: a primal being cut across the blade line, and steaks already
    # cut lying on the fixed table.
    add(uv_sphere((-0.10, T + 0.065, 0.02), (0.11, 0.10, 0.06), 10, 18, M.MEAT))
    for sx, sz in ((0.10, 0.05), (0.16, 0.12), (0.22, 0.03)):
        # Lying flat: thin in y.
        add(uv_sphere((sx, T + 0.012, sz), (0.06, 0.012, 0.05), 6, 14, M.MEAT))

    # Controls on the column face.
    add(box((0.33, 1.06, 0.04), (0.405, 1.18, 0.045), M.PANEL))
    add(uv_sphere((0.367, 1.14, 0.046), (0.013, 0.013, 0.006), 6, 10, M.WALL))
    add(uv_sphere((0.367, 1.09, 0.046), (0.013, 0.013, 0.006), 6, 10, M.MEAT))
    return meshes


def camera_rotation(mount: CameraMount) -> np.ndarray:
    """World-to-camera rotation (OpenCV axes) for a mount."""
    from .mathutil import look_at, rot_z

    R = look_at(np.asarray(mount.position, float), np.asarray(mount.target, float))
    if mount.roll_deg:
        R = rot_z(np.radians(mount.roll_deg)) @ R
    return R


def angular_size_deg(zone: Zone, eye: np.ndarray) -> tuple[float, float]:
    """The zone's width and height in degrees, seen from ``eye``."""
    c = zone.corners
    left = normalize((c[0] + c[2]) / 2 - eye)
    right = normalize((c[1] + c[3]) / 2 - eye)
    top = normalize((c[0] + c[1]) / 2 - eye)
    bottom = normalize((c[2] + c[3]) / 2 - eye)
    return (
        float(np.degrees(np.arccos(np.clip(left @ right, -1, 1)))),
        float(np.degrees(np.arccos(np.clip(top @ bottom, -1, 1)))),
    )
