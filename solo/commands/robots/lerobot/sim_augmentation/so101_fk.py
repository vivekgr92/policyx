"""
Local, lightweight forward kinematics for the SO-101 follower arm.

This exists as a local alternative to computing EEF pose via Isaac Lab's
`Articulation.data.body_state_w` (proven working on the Runpod/Isaac Lab pod
in a prior task), for tools that need gripper pose on THIS machine without
the full Isaac Sim/Isaac Lab GPU stack -- e.g. the hand-eye (camera-to-base)
extrinsics calibration in `hand_eye_calibration.py`, which drives the real
arm and needs EEF pose locally, in the loop, with no GPU pod involved.

Uses `ikpy` (pure Python, URDF-driven) against the real, public SO-101 URDF
from TheRobotStudio/SO-ARM100 (`Simulation/SO101/so101_new_calib.urdf`,
vendored at `assets/so101_new_calib.urdf`) -- the same "new_calib" file whose
nominal joint limits are already cross-referenced in `so101_joint_mapping.py`
(confirmed byte-identical limit values between that module and this URDF).

Key real finding: unlike the Isaac Lab/USD asset (which renamed joints to
`Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll/Jaw`), this public URDF's joint
names are IDENTICAL to LeRobot's own recorded action names (`shoulder_pan`,
`shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) -- no
name remapping needed here, only the degrees->radians conversion.

The URDF defines a `gripper_frame_joint` (fixed) ending at `gripper_frame_link`
as a dedicated, non-actuated tool-center-point frame -- separate from the
moving jaw (`moving_jaw_so101_v1_link`, behind the revolute `gripper` joint).
`ikpy`'s URDF chain builder follows the `gripper_frame_joint` branch and stops
there, so the chain built here naturally ends at this fixed TCP frame, NOT the
moving jaw -- which is exactly the frame a tag mounted on the (non-moving)
wrist/gripper housing represents, and the frame whose pose is independent of
gripper open/close state.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_URDF_PATH = Path(__file__).parent / "assets" / "so101_new_calib.urdf"

# Order matches the chain's active (non-fixed) links, in URDF traversal order.
_ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        import ikpy.chain  # local import: heavy-ish (scipy), only pay the cost if FK is actually used

        if not _URDF_PATH.is_file():
            raise FileNotFoundError(
                f"SO-101 URDF not found at {_URDF_PATH}. It should be vendored in this repo "
                "(solo/commands/robots/lerobot/sim_augmentation/assets/so101_new_calib.urdf)."
            )
        _chain = ikpy.chain.Chain.from_urdf_file(str(_URDF_PATH))
    return _chain


def compute_gripper_pose(recorded_deg: dict) -> np.ndarray:
    """
    Forward kinematics: recorded joint values (LeRobot convention, degrees for
    the 5 arm joints -- see `so101_joint_mapping.py` for the full provenance
    of this convention) -> gripper_frame_link pose as a 4x4 homogeneous
    transform in the robot base frame.

    `recorded_deg` must have keys "shoulder_pan", "shoulder_lift",
    "elbow_flex", "wrist_flex", "wrist_roll" (degrees). A "gripper" key, if
    present, is ignored -- gripper_frame_link's pose does not depend on jaw
    state (see module docstring).
    """
    chain = _get_chain()
    vec = [0.0] * len(chain.links)
    for i, link in enumerate(chain.links):
        if link.name in _ARM_JOINT_NAMES:
            vec[i] = recorded_deg[link.name] * math.pi / 180.0
    return chain.forward_kinematics(vec)


def joint_limits_deg() -> dict[str, tuple[float, float]]:
    """Real joint limits read directly from the URDF (radians -> degrees), for
    the 5 arm joints. Matches `so101_joint_mapping.py`'s `nominal_limit_rad`
    values (verified identical at the time this module was written -- both
    trace back to the same public URDF)."""
    chain = _get_chain()
    out = {}
    for link in chain.links:
        if link.name in _ARM_JOINT_NAMES and link.bounds is not None:
            lo, hi = link.bounds
            out[link.name] = (math.degrees(lo), math.degrees(hi))
    return out
