"""A simulated meat-saw operator, for tuning and evaluating the gaze monitor.

A rendered person stands at a 14" floor-standing band saw and works: watches
the cut, glances at the product, the tray, the controls and coworkers,
blinks, sways and steps. The camera is the D435i the monitor uses, mounted on
the saw, and every frame comes with the ground truth no real camera can give:
where the operator was actually looking, how each eye was rotated in the
head, the head's pose, the distance and the true landmark positions.

See tools/operator_sim/README.md. The modules:

=================  ==========================================================
``behaviour``      Gaze, eye-head coordination, VOR, blinks, posture; scenarios
``face``           The face (MediaPipe's canonical mesh), eyeballs and eyelids
``operator``       Body, arms, and assembling the posed operator
``scene``          The saw, the room, the camera mount, the attention zone
``render``         Rendering and the RealSense sensor model
``sim``            Scenario in, frames and ground truth out
``evaluate``       Generate runs, replay the real pipeline, score, sweep
``source``         A camera the real monitor can open
``placement``      Where to mount the camera, and how far it may drift
``benchmark``      A fixed suite of cases and a scorecard, for comparing changes
``promo``          The promotional video
=================  ==========================================================

Only numpy is needed to render. FaceMesh (``generate``) needs mediapipe and
OpenCV, as the monitor itself does.
"""

from .behaviour import SCENARIOS, Segment, Traits
from .face import PRESETS, Appearance
from .scene import BLADE_ZONE, CameraMount, Scene, Zone
from .sim import SimFrame, Simulation
from .source import SimulatedCamera

__all__ = [
    "SCENARIOS",
    "PRESETS",
    "BLADE_ZONE",
    "Appearance",
    "CameraMount",
    "Scene",
    "Segment",
    "SimFrame",
    "SimulatedCamera",
    "Simulation",
    "Traits",
    "Zone",
]
