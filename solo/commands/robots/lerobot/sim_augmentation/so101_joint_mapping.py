"""
SO-101 joint-space conversion: LeRobot recorded values -> Isaac Lab/USD radians.

This is a property of the SO-101 robot + this project's recording convention
(LeRobot's `so_follower` module, DEGREES norm mode for the 5 arm joints,
RANGE_0_100 for the gripper), not of any particular dataset -- reused by any
SO-101 LeRobot dataset this pipeline is pointed at.

Provenance (real source read, not assumed):
- lerobot.robots.so_follower.config_so_follower: use_degrees=True by default
  -> MotorNormMode.DEGREES for the 5 arm joints, MotorNormMode.RANGE_0_100
  hardcoded for gripper regardless of use_degrees.
- DEGREES norm: value is physical degrees about each robot's own calibrated
  mid-point (homing-offset already subtracted in hardware), UNCLAMPED -- can
  exceed a nominal +-100 span if the arm is driven past where it was swept
  during calibration. So: sim_radians = recorded_pos_deg * pi / 180, no
  per-dataset ticks/calibration math needed.
- gripper is a 0-100 percent-of-travel value, not degrees.

Joint limits below are the public TheRobotStudio SO-ARM101 "new_calib" URDF
nominal values (https://github.com/TheRobotStudio/SO-ARM100), used only as a
sanity-check fallback. The real clip bounds used at runtime should come from
the actual loaded Isaac Lab Articulation (`articulation.data.joint_limits`)
on whichever pod/USD is in use -- see `get_runtime_joint_limits()` in
fk_batch.py. Different physical SO-101 units calibrate different spans, and
different USD builds may differ slightly from the nominal URDF.

Gripper open/close direction (does 0% map to the URDF's lower or upper bound)
is UNVERIFIED -- no source evidence found for which end is open vs closed.
Confirm empirically (command 0% and 100%, inspect the simulated Jaw joint)
before trusting GRIPPER_PCT_MAPS_TO_LOWER_AT_ZERO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Does gripper.pos == 0 correspond to the URDF's lower joint limit (True) or
# upper limit (False)? UNVERIFIED -- see module docstring. Flip this constant
# once confirmed empirically against the real sim.
GRIPPER_PCT_MAPS_TO_LOWER_AT_ZERO = True


@dataclass(frozen=True)
class JointMapping:
    recorded_name: str  # e.g. "shoulder_pan" (LeRobot action/state feature, minus ".pos")
    sim_joint_name: str  # e.g. "Rotation" (Isaac Lab/USD articulation joint name)
    is_gripper: bool
    nominal_limit_rad: tuple[float, float]  # public URDF fallback, (lower, upper)
    sign_confidence: str  # "high" | "medium" | "low" -- see investigation notes
    notes: str

    def to_sim_radians(self, recorded_value: float) -> float:
        if self.is_gripper:
            lo, hi = self.nominal_limit_rad
            pct = recorded_value / 100.0
            if not GRIPPER_PCT_MAPS_TO_LOWER_AT_ZERO:
                pct = 1.0 - pct
            return lo + pct * (hi - lo)
        return recorded_value * math.pi / 180.0


# Isaac joint-name correspondence confirmed medium-confidence by motor ID
# (1:1 with LeRobot's motor IDs 1-6); flagged for on-pod confirmation against
# the real loaded Articulation's joint name list.
SO101_JOINT_MAPPING: dict[str, JointMapping] = {
    "shoulder_pan": JointMapping(
        recorded_name="shoulder_pan",
        sim_joint_name="Rotation",
        is_gripper=False,
        nominal_limit_rad=(-1.91986, 1.91986),
        sign_confidence="medium",
        notes=(
            "DEGREES norm mode, unclamped. Dataset action values come from the "
            "StarArm102 leader bridge using the same deg formula against the leader's "
            "own calibration, live signs=1.0/gains=1.0/offsets=0.0 (~/.solo/starai_map.json) "
            "-- no inversion was ever needed by the integrator, soft evidence sign already "
            "matches the follower/URDF convention."
        ),
    ),
    "shoulder_lift": JointMapping(
        recorded_name="shoulder_lift",
        sim_joint_name="Pitch",
        is_gripper=False,
        nominal_limit_rad=(-1.74533, 1.74533),
        sign_confidence="medium",
        notes="Same provenance as shoulder_pan. Real dataset values can exceed this nominal span slightly -- clip to the runtime Articulation's real limits, not this nominal value.",
    ),
    "elbow_flex": JointMapping(
        recorded_name="elbow_flex",
        sim_joint_name="Elbow",
        is_gripper=False,
        nominal_limit_rad=(-1.69, 1.69),
        sign_confidence="medium",
        notes="Same provenance. Real dataset values can exceed this nominal span slightly.",
    ),
    "wrist_flex": JointMapping(
        recorded_name="wrist_flex",
        sim_joint_name="Wrist_Pitch",
        is_gripper=False,
        nominal_limit_rad=(-1.65806, 1.65806),
        sign_confidence="medium",
        notes="Same provenance. Real dataset values observed up to ~10 deg past this nominal span on some units.",
    ),
    "wrist_roll": JointMapping(
        recorded_name="wrist_roll",
        sim_joint_name="Wrist_Roll",
        is_gripper=False,
        nominal_limit_rad=(-2.74385, 2.84121),
        sign_confidence="medium",
        notes=(
            "recorded_pos = leader Motor_5 degrees + leader Motor_3 (forearm-roll) degrees "
            "blended in (DEFAULT_CONTINUOUS_JOINTS, stararm102_so101.py), hence the widest/"
            "most asymmetric real range. Still plain degrees -> same formula applies."
        ),
    ),
    "gripper": JointMapping(
        recorded_name="gripper",
        sim_joint_name="Jaw",
        is_gripper=True,
        nominal_limit_rad=(-0.174533, 1.74533),
        sign_confidence="low",
        notes=(
            "RANGE_0_100 percent-of-travel, hardcoded regardless of use_degrees. "
            "Open/close direction at 0% is UNVERIFIED -- see GRIPPER_PCT_MAPS_TO_LOWER_AT_ZERO."
        ),
    ),
}


def convert_frame(recorded: dict[str, float]) -> dict[str, float]:
    """
    Convert one frame's recorded joint values (keyed by LeRobot joint name,
    e.g. {"shoulder_pan": -14.15, ...}) to simulated joint values (keyed by
    Isaac Lab/USD joint name, in radians), using the nominal/fallback limits.
    For clipping against a real loaded Articulation's actual limits, use
    convert_frame_with_limits() instead.
    """
    return {
        mapping.sim_joint_name: mapping.to_sim_radians(recorded[name])
        for name, mapping in SO101_JOINT_MAPPING.items()
    }


def convert_frame_with_limits(
    recorded: dict[str, float], runtime_limits_rad: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """
    Same as convert_frame(), but clips each converted value to real runtime
    joint limits read from the loaded Isaac Lab Articulation
    (`runtime_limits_rad`, keyed by sim_joint_name) instead of the nominal
    public-URDF fallback baked into SO101_JOINT_MAPPING.
    """
    out = {}
    for name, mapping in SO101_JOINT_MAPPING.items():
        value = mapping.to_sim_radians(recorded[name])
        lo, hi = runtime_limits_rad.get(mapping.sim_joint_name, mapping.nominal_limit_rad)
        out[mapping.sim_joint_name] = max(lo, min(hi, value))
    return out
