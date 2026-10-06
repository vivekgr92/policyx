"""
Scripted camera-to-robot-base (hand-eye) extrinsics calibration for the SO-101
follower arm, for the AprilTag object-pose pipeline feeding Isaac Mimic.

Physical setup required before running this:
  - A tag36h11 AprilTag, ID 2, ~35-40mm, mounted on a NON-MOVING part of the
    wrist/gripper housing (not the jaws) -- its pose relative to the FK chain's
    `gripper_frame_link` (see `so101_fk.py`) must stay fixed regardless of
    gripper open/close state.
  - Camera intrinsics calibration already run and saved to
    `~/.solo/camera_calibration.json` (built by a parallel task -- see
    `_load_camera_intrinsics()` for the exact schema this reads).

This is the classic "eye-to-hand" configuration: the camera is fixed in the
world, the calibration target (tag) moves with the robot. The standard trick
for using `cv2.calibrateHandEye` (which is documented for the eye-in-hand case)
in this configuration is to invert the robot poses -- feed it base2gripper
instead of gripper2base -- which makes the math solve directly for cam2base.
This is mathematically equivalent to swapping which frame is treated as
"fixed" vs "moving" in the AX=XB formulation; see e.g. OpenCV community
answers on eye-to-hand calibration (searched, this is a well-established
technique, not invented here).

Important property of this formulation (worth being explicit about, since it's
easy to assume otherwise): neither the exact offset of the tag on the gripper
housing, nor the target's pose in the base frame, needs to be known. The AX=XB
relation is built from RELATIVE motion between pose pairs, which cancels out
any fixed-but-unknown offset -- only (a) the tag's pose per-frame in camera
frame, and (b) the gripper frame's pose per-frame in robot-base frame (via FK)
need to be measured, as long as the tag is rigidly fixed to the frame whose FK
pose is used.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.prompt import Confirm

from solo.commands.robots.lerobot.sim_augmentation.so101_fk import compute_gripper_pose

TAG_FAMILY = "tag36h11"
GRIPPER_TAG_ID = 2
DEFAULT_TAG_SIZE_M = 0.035  # 35mm -- override via run_hand_eye_calibration(tag_size_m=...)

_EXTRINSICS_OUTPUT_PATH = Path.home() / ".solo" / "camera_extrinsics.json"
_INTRINSICS_INPUT_PATH = Path.home() / ".solo" / "camera_calibration.json"

# Fraction of each joint's real calibrated travel to KEEP (centered on the
# midpoint) when generating sweep waypoints -- conservative, to stay well
# clear of mechanical limits during unattended scripted motion.
_SWEEP_RANGE_FRACTION = 0.6
_NUM_WAYPOINTS = 18
_GRIPPER_FIXED_PCT = 50.0  # gripper held constant throughout -- irrelevant to the fixed tag frame, but must not be left undefined
_MOVE_DURATION_S = 1.5
_SETTLE_S = 0.4


@dataclass
class HandEyeSample:
    gripper2base: np.ndarray  # 4x4, from FK
    target2cam: np.ndarray  # 4x4, from AprilTag detection


def _load_camera_intrinsics(camera_angle: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (camera_matrix 3x3, dist_coeffs) for `camera_angle` from
    `~/.solo/camera_calibration.json`.

    Schema expected (built by the parallel intrinsics-calibration task):
    a top-level dict keyed by camera angle/id, each value holding the
    intrinsics for that camera. Since that task may use a different exact key
    name than assumed here, this accepts a few reasonable variants
    (`camera_matrix`/`K`/`intrinsics`, `dist_coeffs`/`distortion`/`dist`) and
    raises a clear, actionable error listing exactly what it found if none
    match -- rather than silently guessing wrong values for a calibration
    that moves real hardware.
    """
    if not _INTRINSICS_INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"No camera intrinsics calibration found at {_INTRINSICS_INPUT_PATH}. "
            "Run the camera intrinsics calibration tool first -- hand-eye calibration "
            "needs real calibrated intrinsics to turn AprilTag detections into metric poses."
        )

    with open(_INTRINSICS_INPUT_PATH) as f:
        data = json.load(f)

    entry = data.get(camera_angle)
    if entry is None and len(data) == 1:
        # Only one camera calibrated -- use it regardless of key name, rather
        # than failing on a naming mismatch between tools.
        entry = next(iter(data.values()))
    if entry is None:
        raise KeyError(
            f"No intrinsics entry for camera '{camera_angle}' in {_INTRINSICS_INPUT_PATH}. "
            f"Keys present: {list(data.keys())}"
        )

    matrix_key = next((k for k in ("camera_matrix", "K", "intrinsics", "camera_matrix_list") if k in entry), None)
    dist_key = next((k for k in ("dist_coeffs", "distortion", "dist") if k in entry), None)
    if matrix_key is None or dist_key is None:
        raise KeyError(
            f"Intrinsics entry for '{camera_angle}' is missing a recognized camera-matrix/"
            f"distortion key. Fields present: {list(entry.keys())}. Expected one of "
            f"camera_matrix/K/intrinsics and one of dist_coeffs/distortion/dist."
        )

    camera_matrix = np.array(entry[matrix_key], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.array(entry[dist_key], dtype=np.float64).reshape(-1)
    return camera_matrix, dist_coeffs


def _generate_waypoints(limits, seed: int = 42) -> list[dict[str, float]]:
    """Random sample within a conservative sub-range of each arm joint's real
    calibrated safe range (`limits`, from `deployx.safety.load_joint_limits` --
    the SAME real per-unit calibration file `_move_to_rest_position`/
    `_perturb_action` already trust, not the nominal public-URDF values).
    Gripper is excluded here and held fixed by the caller."""
    rng = random.Random(seed)
    arm_names = [n for n in limits.names if n != "gripper"]
    waypoints = []
    for _ in range(_NUM_WAYPOINTS):
        wp = {}
        for name in arm_names:
            i = limits.names.index(name)
            lo, hi = limits.position_min[i], limits.position_max[i]
            mid = (lo + hi) / 2.0
            half_span = (hi - lo) / 2.0 * _SWEEP_RANGE_FRACTION
            wp[name] = rng.uniform(mid - half_span, mid + half_span)
        waypoints.append(wp)
    return waypoints


def _move_to(robot, limits, robot_action_processor, fps: int, target: dict[str, float], duration_s: float) -> None:
    """Same smooth-interpolation pattern as `modes/replay.py`'s
    `_move_to_rest_position` -- deliberately reused rather than reinvented,
    since this moves real hardware and that pattern is the one already
    exercised in this codebase."""
    from lerobot.utils.robot_utils import precise_sleep

    obs = robot.get_observation()
    positions = {key[: -len(".pos")]: float(value) for key, value in obs.items() if key.endswith(".pos")}
    names = list(target.keys())
    if not all(name in positions for name in names):
        raise RuntimeError(f"Could not read current position for all of {names} from robot observation.")

    start = [positions[name] for name in names]
    end = [target[name] for name in names]

    num_steps = max(1, int(duration_s * fps))
    for step in range(1, num_steps + 1):
        start_t = time.perf_counter()
        t = step / num_steps
        action = {f"{name}.pos": start[i] + (end[i] - start[i]) * t for i, name in enumerate(names)}
        obs = robot.get_observation()
        processed_action = robot_action_processor((action, obs))
        robot.send_action(processed_action)
        precise_sleep(max(0.0, 1 / fps - (time.perf_counter() - start_t)))


def _detect_gripper_tag(
    frame_bgr: np.ndarray, detector, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, tag_size_m: float
) -> Optional[np.ndarray]:
    """Undistort using the real calibrated intrinsics, then detect+estimate
    pose of GRIPPER_TAG_ID against the undistorted camera's own (distortion-free)
    intrinsics -- the correct way to combine OpenCV calibration with
    pupil_apriltags' pose estimation, which assumes an ideal pinhole model.
    Returns the tag's 4x4 pose in camera frame, or None if not detected."""
    import cv2

    h, w = frame_bgr.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0)
    undistorted = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs, None, new_camera_matrix)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    fx, fy = new_camera_matrix[0, 0], new_camera_matrix[1, 1]
    cx, cy = new_camera_matrix[0, 2], new_camera_matrix[1, 2]

    detections = detector.detect(
        gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_size_m
    )
    for d in detections:
        if d.tag_id == GRIPPER_TAG_ID:
            T = np.eye(4)
            T[:3, :3] = d.pose_R
            T[:3, 3] = d.pose_t.reshape(3)
            return T
    return None


def solve_hand_eye(samples: list[HandEyeSample]) -> dict:
    """
    Eye-to-hand hand-eye calibration: samples carry gripper2base (FK) and
    target2cam (AprilTag) per pose. Inverting gripper2base -> base2gripper
    before passing to cv2.calibrateHandEye makes it solve for cam2base
    directly (see module docstring for why this inversion is correct here).
    """
    import cv2

    if len(samples) < 8:
        raise ValueError(
            f"Only {len(samples)} usable samples (tag detected + FK valid) -- need at least "
            "8-10 for a reasonable hand-eye solve, ideally 15+. Re-run with more waypoints or "
            "check the tag is actually visible across the sweep."
        )

    R_base2gripper, t_base2gripper = [], []
    R_target2cam, t_target2cam = [], []
    for s in samples:
        R_g2b, t_g2b = s.gripper2base[:3, :3], s.gripper2base[:3, 3]
        R_b2g = R_g2b.T
        t_b2g = -R_b2g @ t_g2b
        R_base2gripper.append(R_b2g)
        t_base2gripper.append(t_b2g)
        R_target2cam.append(s.target2cam[:3, :3])
        t_target2cam.append(s.target2cam[:3, 3])

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_PARK
    )

    # AX=XB residual check: for every pair (i, j), verify A_ij @ X ~= X @ B_ij,
    # where X = [R_cam2base | t_cam2base]. Derived directly for the eye-to-hand
    # case (not by reusing the eye-in-hand A_ij -- that uses a different pair
    # ordering and was verified by synthetic test to give a false ~100deg
    # "error" on an exact solve): from target2base_i = gripper2base_i @
    # tag2gripper = cam2base @ target2cam_i for each pose, eliminating the
    # unknown constant tag2gripper between poses i,j gives
    #   A_ij = gripper2base_j @ inv(gripper2base_i)
    #   B_ij = target2cam_j  @ inv(target2cam_i)
    #   A_ij @ X == X @ B_ij
    # This does NOT require knowing the tag's exact offset on the gripper or
    # the target's pose in the base frame (see module docstring) -- confirmed
    # via a synthetic ground-truth test giving ~1e-6 deg / ~1e-16 m residual.
    X = np.eye(4)
    X[:3, :3] = R_cam2base
    X[:3, 3] = t_cam2base.reshape(3)

    gripper2base = [s.gripper2base for s in samples]
    target2cam = [s.target2cam for s in samples]

    rot_errors_deg, trans_errors_m = [], []
    n = len(samples)
    for i in range(n):
        for j in range(i + 1, n):
            A = gripper2base[j] @ np.linalg.inv(gripper2base[i])
            B = target2cam[j] @ np.linalg.inv(target2cam[i])
            lhs = A @ X
            rhs = X @ B
            R_err = lhs[:3, :3].T @ rhs[:3, :3]
            angle = math.degrees(math.acos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)))
            rot_errors_deg.append(angle)
            trans_errors_m.append(float(np.linalg.norm(lhs[:3, 3] - rhs[:3, 3])))

    return {
        "R_cam_to_base": R_cam2base.tolist(),
        "t_cam_to_base": t_cam2base.reshape(3).tolist(),
        "method": "cv2.CALIB_HAND_EYE_PARK",
        "num_samples": n,
        "mean_rotation_residual_deg": float(np.mean(rot_errors_deg)),
        "max_rotation_residual_deg": float(np.max(rot_errors_deg)),
        "mean_translation_residual_m": float(np.mean(trans_errors_m)),
        "max_translation_residual_m": float(np.max(trans_errors_m)),
    }


def run_hand_eye_calibration(
    camera_angle: str = "front",
    robot_type: str = "so101",
    follower_id: Optional[str] = None,
    follower_port: Optional[str] = None,
    tag_size_m: float = DEFAULT_TAG_SIZE_M,
    fps: int = 30,
) -> dict:
    """Top-level orchestration: connect, sweep, capture, solve, save. Real
    hardware motion -- Ctrl-C is handled via the same `safe_shutdown()` path
    DeployX's edge agent uses (disables torque, disconnects cleanly) rather
    than leaving the arm powered mid-motion."""
    import json as _json
    import os as _os

    from lerobot.processor import make_default_robot_action_processor
    from lerobot.robots import make_robot_from_config
    from pupil_apriltags import Detector

    from solo.commands.robots.lerobot.cameras import setup_cameras
    from solo.commands.robots.lerobot.config import create_follower_config, get_robot_config_classes
    from solo.commands.robots.lerobot.deployx.safety import load_joint_limits, safe_shutdown
    from solo.commands.robots.lerobot.ports import detect_arm_port

    typer.echo("🎯 SO-101 hand-eye (camera-to-base) extrinsics calibration")
    typer.echo(
        f"   Make sure a {TAG_FAMILY} tag, ID {GRIPPER_TAG_ID}, is mounted on a NON-MOVING part "
        "of the wrist/gripper housing before continuing."
    )
    if not Confirm.ask("Tag mounted and ready?", default=True):
        typer.echo("Aborted.")
        return {}

    _, follower_config_class = get_robot_config_classes(robot_type)
    if follower_port is None:
        follower_port, _ = detect_arm_port("follower", robot_type=robot_type)
        if not follower_port:
            raise RuntimeError("Could not auto-detect the SO-101 follower arm's port.")

    if follower_id is None:
        # Same resolution order as the rest of this codebase (see modes/replay.py):
        # reuse the follower_id already saved in ~/.solo/config.json rather than
        # guessing a generic "{robot_type}_follower" string that would almost
        # certainly miss the real calibration file load_joint_limits() needs.
        solo_config_path = _os.path.expanduser("~/.solo/config.json")
        solo_config = {}
        if _os.path.isfile(solo_config_path):
            with open(solo_config_path) as f:
                solo_config = _json.load(f)
        saved_follower_id = solo_config.get("lerobot", {}).get("follower_id")
        if saved_follower_id:
            follower_id = saved_follower_id
        else:
            from solo.commands.robots.lerobot.utils.helper import prompt_arm_id

            follower_id = prompt_arm_id(solo_config, "follower", robot_type)

    typer.echo("\n📷 Set up the camera used for this calibration (must match the recording camera):")
    camera_config = setup_cameras()
    if not camera_config or not camera_config.get("cameras"):
        raise RuntimeError("No camera configured -- hand-eye calibration needs a live camera feed.")
    camera_angle = camera_config["cameras"][0]["angle"]

    camera_matrix, dist_coeffs = _load_camera_intrinsics(camera_angle)
    limits = load_joint_limits(robot_type, follower_id)
    waypoints = _generate_waypoints(limits)

    follower_config = create_follower_config(
        follower_config_class, follower_port, robot_type, camera_config=camera_config, follower_id=follower_id
    )
    robot_action_processor = make_default_robot_action_processor()
    detector = Detector(families=TAG_FAMILY)

    robot = None
    samples: list[HandEyeSample] = []
    try:
        robot = make_robot_from_config(follower_config)
        robot.connect()

        typer.echo(f"\n🔁 Sweeping {len(waypoints)} waypoints (tag ID {GRIPPER_TAG_ID}, {tag_size_m * 1000:.0f}mm)...")
        for idx, wp in enumerate(waypoints):
            target = dict(wp)
            target["gripper"] = _GRIPPER_FIXED_PCT
            _move_to(robot, limits, robot_action_processor, fps, target, _MOVE_DURATION_S)
            time.sleep(_SETTLE_S)

            obs = robot.get_observation()
            joint_positions = {k[: -len(".pos")]: float(v) for k, v in obs.items() if k.endswith(".pos")}
            frame = obs.get(camera_angle)
            if frame is None:
                typer.echo(f"   [{idx + 1}/{len(waypoints)}] ⚠️  no camera frame captured, skipping")
                continue

            tag_pose_cam = _detect_gripper_tag(frame, detector, camera_matrix, dist_coeffs, tag_size_m)
            if tag_pose_cam is None:
                typer.echo(f"   [{idx + 1}/{len(waypoints)}] ⚠️  tag not detected, skipping")
                continue

            gripper_pose_base = compute_gripper_pose(joint_positions)
            samples.append(HandEyeSample(gripper2base=gripper_pose_base, target2cam=tag_pose_cam))
            typer.echo(f"   [{idx + 1}/{len(waypoints)}] ✅ captured")

    except KeyboardInterrupt:
        typer.echo("\n🛑 Interrupted by user.")
    finally:
        safe_shutdown(robot, reason="hand-eye calibration finished or interrupted")

    typer.echo(f"\n📐 Solving with {len(samples)}/{len(waypoints)} usable samples...")
    result = solve_hand_eye(samples)
    result["camera_angle"] = camera_angle
    result["tag_size_m"] = tag_size_m

    _EXTRINSICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if _EXTRINSICS_OUTPUT_PATH.is_file():
        with open(_EXTRINSICS_OUTPUT_PATH) as f:
            existing = json.load(f)
    existing[camera_angle] = result
    with open(_EXTRINSICS_OUTPUT_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    typer.echo(f"\n✅ Saved camera-to-base extrinsics for '{camera_angle}' to {_EXTRINSICS_OUTPUT_PATH}")
    typer.echo(
        f"   Residual: mean {result['mean_rotation_residual_deg']:.2f}deg / "
        f"{result['mean_translation_residual_m'] * 1000:.1f}mm, "
        f"max {result['max_rotation_residual_deg']:.2f}deg / {result['max_translation_residual_m'] * 1000:.1f}mm "
        "(lower is better; several degrees / cm indicates a poor solve -- re-run with more/better-spread waypoints)"
    )
    return result


if __name__ == "__main__":
    run_hand_eye_calibration()
