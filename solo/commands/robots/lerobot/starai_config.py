"""
Star Arm 102 (StarAI / Fashionstar) configuration utilities for Solo CLI.

The Star Arm 102 leader drives an SO101 follower through the adapter in
`solo.commands.robots.lerobot.teleoperators.stararm102_so101`. Because the two
arms are different hardware, the transfer needs a small per-joint correction
(which way each axis runs, how far it swings, where its zero sits). That
correction lives in `~/.solo/starai_map.json` so every mode - teleop, record,
inference - picks up the same tuning, and `solo robo --star-tune` can edit it.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from solo.commands.robots.lerobot.teleoperators.stararm102_so101 import (
    DEFAULT_CONTINUOUS_JOINTS,
    DEFAULT_JOINT_MAP,
    SO101_JOINTS,
)

STARAI_ROBOT_TYPES = ("stararm102",)

STARAI_MAP_PATH = Path.home() / ".solo" / "starai_map.json"

DEFAULT_STARAI_MAP: Dict[str, Any] = {
    "joint_map": dict(DEFAULT_JOINT_MAP),
    "signs": {joint: 1.0 for joint in SO101_JOINTS},
    "gains": {joint: 1.0 for joint in SO101_JOINTS},
    "offsets": {joint: 0.0 for joint in SO101_JOINTS},
    # {follower_joint: {leader_servo: weight}} — extra leader servos folded into a
    # follower joint. Empty by default so the stock mapping stays a plain 1:1.
    "blend": {},
    "continuous_joints": list(DEFAULT_CONTINUOUS_JOINTS),
    "smoothing": 0.0,
    # Caps how far the follower may be asked to move in one control step (degrees).
    # Cross-brand mapping errors show up as a large jump, so this is on by default.
    "max_relative_target": 12.0,
    "clamp_to_follower_limits": True,
}


def is_starai_robot(robot_type: Optional[str]) -> bool:
    """Whether this robot type uses a Star Arm 102 leader."""
    return robot_type in STARAI_ROBOT_TYPES


def load_starai_map() -> Dict[str, Any]:
    """Load the saved leader->follower mapping, filled in with defaults."""
    mapping = json.loads(json.dumps(DEFAULT_STARAI_MAP))  # deep copy

    if not STARAI_MAP_PATH.exists():
        return mapping

    try:
        with open(STARAI_MAP_PATH) as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        typer.echo(f"⚠️  Could not read {STARAI_MAP_PATH} ({e}); using defaults.")
        return mapping

    for key, value in saved.items():
        if key in ("joint_map", "signs", "gains", "offsets", "blend") and isinstance(value, dict):
            mapping[key].update(value)
        else:
            mapping[key] = value

    return mapping


def save_starai_map(mapping: Dict[str, Any]) -> None:
    """Persist the leader->follower mapping."""
    os.makedirs(STARAI_MAP_PATH.parent, exist_ok=True)
    with open(STARAI_MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=4)
    typer.echo(f"💾 Star Arm 102 mapping saved to {STARAI_MAP_PATH}")


def get_follower_joint_limits(follower_id: Optional[str]) -> Dict[str, List[float]]:
    """
    Read the SO101 follower's calibration and express its travel in degrees.

    LeRobot normalises a follower body joint as `(count - midpoint) * 360 / resolution`,
    so the calibrated range becomes a symmetric degree window around zero. Handing
    those bounds to the leader adapter means a wrongly signed axis is clamped at the
    edge of the follower's own travel instead of being driven into a hard stop.

    Returns an empty dict when the calibration file cannot be found or read; the
    adapter then simply applies no clamp.
    """
    if not follower_id:
        return {}

    try:
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
    except ImportError:
        return {}

    fpath = HF_LEROBOT_CALIBRATION / ROBOTS / SO101Follower.name / f"{follower_id}.json"
    if not fpath.is_file():
        return {}

    try:
        with open(fpath) as f:
            calibration = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    resolution = FeetechMotorsBus.model_resolution_table.get("sts3215", 4096) - 1

    limits: Dict[str, List[float]] = {}
    for joint, entry in calibration.items():
        try:
            range_min = float(entry["range_min"])
            range_max = float(entry["range_max"])
        except (KeyError, TypeError, ValueError):
            continue

        if joint == "gripper":
            limits[joint] = [0.0, 100.0]
            continue

        half_span = (range_max - range_min) / 2.0 * 360.0 / resolution
        limits[joint] = [-half_span, half_span]

    return limits


def resolve_starai_leader_port(current_port: Optional[str], verbose: bool = True) -> Optional[str]:
    """
    Return a port that actually has a Star Arm 102 on it.

    A saved `leader_port` belongs to whichever leader was configured last, which
    may well be an SO101 on a different connector. Handing that port to the
    FashionStar driver fails deep in the bus with "motor not found id:0", so
    check it first and rescan when it is not the right arm.
    """
    from solo.commands.robots.lerobot.scan import scan_starai_port

    if current_port and scan_starai_port(current_port):
        return current_port

    if current_port and verbose:
        typer.echo(f"ℹ️  No Star Arm 102 answered on {current_port} — rescanning...")

    try:
        from solo.commands.robots.lerobot.scan import get_serial_ports
        for port in get_serial_ports():
            if port != current_port and scan_starai_port(port):
                if verbose:
                    typer.echo(f"✅ Found the Star Arm 102 on {port}")
                return port
    except Exception as e:
        if verbose:
            typer.echo(f"⚠️  Port scan failed: {e}")

    if verbose:
        typer.echo(
            "❌ No Star Arm 102 found on any port. Check that the UC-01 board has 12V "
            "and its USB cable is connected, then run 'solo robo --scan'."
        )
    return None


def ensure_starai_leader_port(
    config: Dict[str, Any], mode: str, robot_type: Optional[str], leader_port: Optional[str]
) -> Optional[str]:
    """
    Check a saved leader port before a mode runs with it, and correct it in place.

    Saved settings are reused verbatim, so a `leader_port` recorded for an SO101
    survives a switch to a Star Arm 102 and takes the run down inside the
    FashionStar driver. Returns the port to use, or None if no Star Arm 102 is
    connected at all. Non-StarAI robot types pass straight through.
    """
    if not is_starai_robot(robot_type):
        return leader_port

    resolved = resolve_starai_leader_port(leader_port)
    if resolved and resolved != leader_port:
        from solo.commands.robots.lerobot.mode_config import update_mode_config_port
        update_mode_config_port(config, mode, 'leader_port', resolved)
    return resolved


def get_leader_joint_spans(leader_id: Optional[str]) -> Dict[str, float]:
    """
    How many degrees each Star Arm 102 servo covers, per its calibration.

    This is the travel that was actually recorded during calibration, not the
    servo's mechanical limit, so a joint that was swept lazily reads small here.
    """
    if not leader_id:
        return {}

    try:
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
        from solo.commands.robots.lerobot.teleoperators.stararm102_so101 import (
            STARAI_COUNTS_PER_TURN,
            StarArm102SO101Leader,
        )
    except ImportError:
        return {}

    fpath = HF_LEROBOT_CALIBRATION / TELEOPERATORS / StarArm102SO101Leader.name / f"{leader_id}.json"
    if not fpath.is_file():
        return {}

    try:
        with open(fpath) as f:
            calibration = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    spans: Dict[str, float] = {}
    for motor, entry in calibration.items():
        try:
            span = float(entry["range_max"]) - float(entry["range_min"])
        except (KeyError, TypeError, ValueError):
            continue
        spans[motor] = span * 360.0 / STARAI_COUNTS_PER_TURN

    return spans


def suggest_gains(
    leader_id: Optional[str], follower_id: Optional[str], mapping: Dict[str, Any]
) -> Dict[str, float]:
    """
    Gains that map each leader joint's recorded sweep onto the follower's travel.

    The two arms are different sizes, so a 1:1 degree transfer leaves the follower
    pinned at a limit wherever the leader swings further, and short of its reach
    wherever the leader swings less. Scaling by the ratio of the calibrated spans
    makes the leader's full sweep cover the follower's full travel.

    The gripper is left alone - both sides are already normalised 0-100.
    """
    leader_spans = get_leader_joint_spans(leader_id)
    follower_limits = get_follower_joint_limits(follower_id)
    if not leader_spans or not follower_limits:
        return {}

    blend = mapping.get("blend", {})
    gains: Dict[str, float] = {}

    for joint, motor in mapping.get("joint_map", {}).items():
        if joint == "gripper":
            continue

        limits = follower_limits.get(joint)
        if not limits:
            continue

        # A blended joint is driven by several servos, so its usable leader sweep
        # is the weighted total of theirs.
        leader_span = leader_spans.get(motor, 0.0)
        for source, weight in blend.get(joint, {}).items():
            leader_span += abs(weight) * leader_spans.get(source, 0.0)

        follower_span = limits[1] - limits[0]
        if leader_span <= 1.0 or follower_span <= 0:
            continue

        gains[joint] = round(follower_span / leader_span, 2)

    return gains


def create_starai_leader_config(
    port: str,
    leader_id: Optional[str] = None,
    follower_id: Optional[str] = None,
):
    """Build the Star Arm 102 -> SO101 leader config from the saved mapping."""
    from solo.commands.robots.lerobot.teleoperators.stararm102_so101 import (
        StarArm102SO101LeaderConfig,
    )

    mapping = load_starai_map()

    joint_limits: Dict[str, List[float]] = {}
    if mapping.get("clamp_to_follower_limits", True):
        joint_limits = get_follower_joint_limits(follower_id)

    return StarArm102SO101LeaderConfig(
        port=port,
        id=leader_id or "stararm102_leader",
        joint_map=dict(mapping["joint_map"]),
        signs=dict(mapping["signs"]),
        gains=dict(mapping["gains"]),
        offsets=dict(mapping["offsets"]),
        blend={j: dict(w) for j, w in mapping.get("blend", {}).items()},
        joint_limits=joint_limits,
        continuous_joints=list(mapping.get("continuous_joints", [])),
        smoothing=float(mapping.get("smoothing", 0.0)),
    )


def starai_max_relative_target() -> Optional[float]:
    """Per-step motion cap the SO101 follower should run with under a Star Arm 102 leader."""
    value = load_starai_map().get("max_relative_target")
    if value in (None, 0, 0.0):
        return None
    return float(value)


def describe_map(mapping: Dict[str, Any], saved: bool = True) -> None:
    """Print a mapping, saved or still being edited."""
    suffix = "" if saved else " (unsaved changes)"
    typer.echo(f"\n🔗 Star Arm 102 → SO101 joint mapping{suffix}:")
    for joint in SO101_JOINTS:
        motor = mapping["joint_map"].get(joint)
        if not motor:
            typer.echo(f"   • {joint:<14} (unmapped)")
            continue
        sign = mapping["signs"].get(joint, 1.0)
        gain = mapping["gains"].get(joint, 1.0)
        offset = mapping["offsets"].get(joint, 0.0)
        blended = mapping.get("blend", {}).get(joint, {})
        extra = "".join(f" {w:+.2f}×{src}" for src, w in sorted(blended.items()))
        typer.echo(
            f"   • {joint:<14} ← {motor}{extra:<14} "
            f"sign {sign:+.0f}  gain {gain:.2f}  offset {offset:+.1f}°"
        )

    driving = set(mapping["joint_map"].values())
    for sources in mapping.get("blend", {}).values():
        driving.update(sources)
    unmapped = sorted({f"Motor_{i}" for i in range(6)} - driving)
    if unmapped:
        typer.echo(f"   (unused leader joints: {', '.join(unmapped)})")

    cap = mapping.get("max_relative_target")
    typer.echo(f"   Per-step motion cap: {cap}°" if cap else "   Per-step motion cap: off")


def describe_starai_map() -> None:
    """Print the mapping currently saved on disk."""
    describe_map(load_starai_map())
