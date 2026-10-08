"""
Scripted camera-to-robot-base (hand-eye) extrinsics calibration for the SO-101
follower arm, for the AprilTag object-pose pipeline feeding Isaac Mimic.

Physical setup required before running this:
  - A tag36h11 AprilTag, ID 3, ~40mm (NOT smaller -- see DEFAULT_TAG_SIZE_M),
    mounted on a NON-MOVING part of the
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
GRIPPER_TAG_ID = 3
DEFAULT_TAG_SIZE_M = 0.040  # 40mm -- a real run at 20mm produced a catastrophic
# 27-32deg hand-eye residual, traced (after ruling out FK and every plausible
# matrix-convention bug) to AprilTag rotation-estimation noise at a tag too
# small/distant for reliable pose accuracy; reverted back to 40mm. Override
# via run_hand_eye_calibration(tag_size_m=...) if you've verified a smaller
# tag is still accurate enough for your real setup.

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

# Manual-mode stationarity check (see where it's used, in
# run_hand_eye_calibration_manual): arm joints are in degrees, gripper in
# 0-100 percent-of-travel (so101_joint_mapping.py) -- a single small
# threshold works across both since the gripper is held fixed throughout.
_SETTLE_CHECK_S = 0.3
_SETTLE_CHECK_MAX_DRIFT = 0.5


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


def _resolve_hand_eye_setup(camera_angle, robot_type, follower_id, follower_port):
    """Shared by both the autonomous and manual flows: resolve the port/id,
    set up the camera, and load intrinsics. No hardware motion happens here."""
    import json as _json
    import os as _os

    from solo.commands.robots.lerobot.cameras import setup_cameras
    from solo.commands.robots.lerobot.config import get_robot_config_classes
    from solo.commands.robots.lerobot.ports import detect_arm_port

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
    return follower_config_class, follower_port, follower_id, camera_config, camera_angle, camera_matrix, dist_coeffs


def _resolve_leader_setup(robot_type, leader_id, leader_port):
    """Resolve the leader arm's port/id the same way `teleoperation.py` does
    -- reusing the port/id already saved in ~/.solo/config.json from normal
    `solo robo` teleop setup, rather than asking the user to re-enter it."""
    import json as _json
    import os as _os

    from solo.commands.robots.lerobot.config import get_robot_config_classes
    from solo.commands.robots.lerobot.ports import detect_arm_port

    leader_config_class, _ = get_robot_config_classes(robot_type)
    if leader_port is None:
        solo_config_path = _os.path.expanduser("~/.solo/config.json")
        solo_config = {}
        if _os.path.isfile(solo_config_path):
            with open(solo_config_path) as f:
                solo_config = _json.load(f)
        leader_port = solo_config.get("lerobot", {}).get("leader_port")
        if not leader_port:
            leader_port, _ = detect_arm_port("leader", robot_type=robot_type)
            if not leader_port:
                raise RuntimeError("Could not auto-detect the SO-101 leader arm's port.")

    if leader_id is None:
        solo_config_path = _os.path.expanduser("~/.solo/config.json")
        solo_config = {}
        if _os.path.isfile(solo_config_path):
            with open(solo_config_path) as f:
                solo_config = _json.load(f)
        saved_leader_id = solo_config.get("lerobot", {}).get("leader_id")
        if saved_leader_id:
            leader_id = saved_leader_id
        else:
            from solo.commands.robots.lerobot.utils.helper import prompt_arm_id

            leader_id = prompt_arm_id(solo_config, "leader", robot_type)

    return leader_config_class, leader_port, leader_id


_SAMPLES_DUMP_PATH = Path.home() / ".solo" / "camera_extrinsics_debug_samples.json"


_SAMPLE_FRAMES_DIR = Path.home() / ".solo" / "camera_extrinsics_debug_frames"


def _dump_sample_frame(frame, index: int) -> None:
    """Saves the real camera frame for a captured sample, so a bad/suspect
    sample (flagged by analyze_sample_consistency) can be visually inspected
    afterward -- same technique that found the real root cause of the
    checkerboard capture issues earlier (numerical reasoning alone wasn't
    enough; looking at the actual image was)."""
    import cv2

    _SAMPLE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(_SAMPLE_FRAMES_DIR / f"sample_{index:02d}.png"), frame)


def _dump_raw_samples(samples: list[HandEyeSample], camera_angle: str) -> None:
    """Saves the raw gripper2base/target2cam 4x4 matrices from a real run so
    they can be inspected/re-analyzed afterward without needing to re-run
    hardware -- added after a real run produced a catastrophic residual
    (27-32deg / 400-470mm) with FK independently verified exactly correct
    against lerobot's own authoritative implementation, so the next real
    run's raw data is what's needed to pin down whether it's AprilTag
    measurement noise/accuracy or insufficient pose diversity."""
    data = {
        "camera_angle": camera_angle,
        "samples": [
            {"gripper2base": s.gripper2base.tolist(), "target2cam": s.target2cam.tolist()}
            for s in samples
        ],
    }
    _SAMPLES_DUMP_PATH.write_text(json.dumps(data, indent=2))
    typer.echo(f"💾 Raw samples dumped to {_SAMPLES_DUMP_PATH} (for diagnostics, see analyze_sample_consistency())")


def analyze_sample_consistency(samples: list[HandEyeSample]) -> None:
    """
    Convention-invariant sanity check, independent of solve_hand_eye(): for
    each consecutive sample pair, compute the ROTATION ANGLE MAGNITUDE of the
    relative motion two ways -- (a) via FK (gripper2base) and (b) via the
    AprilTag (target2cam) -- and compare them directly.

    This works without knowing the true cam2base transform or any axis
    convention, because rotation ANGLE magnitude (not axis) is invariant
    under a fixed similarity transform: if gripper2base_j @ inv(gripper2base_i)
    has rotation angle theta, then cam2base @ (that same relative motion,
    expressed in camera frame) also has rotation angle theta, for ANY fixed
    cam2base -- so target2cam_j @ inv(target2cam_i) should have close to the
    SAME rotation angle, regardless of what cam2base actually is.

    If the two angle sequences track closely -> the two measurement sources
    (FK, AprilTag) are mutually consistent, and a bad solve is more likely a
    pose-diversity/conditioning issue. If they disagree substantially and
    inconsistently -> hard evidence of a real problem in one of the two
    measurement sources (most likely AprilTag pose noise/accuracy, since FK
    was independently verified exact).
    """
    import math as _math

    n = len(samples)
    if n < 2:
        typer.echo("Need at least 2 samples to analyze consistency.")
        return

    typer.echo(f"\n🔬 Relative-rotation consistency check ({n} samples, {n - 1} consecutive pairs):")
    typer.echo(f"{'pair':>8}  {'FK angle':>10}  {'tag angle':>10}  {'diff':>8}")
    diffs = []
    for i in range(n - 1):
        g_i, g_j = samples[i].gripper2base, samples[i + 1].gripper2base
        t_i, t_j = samples[i].target2cam, samples[i + 1].target2cam

        R_fk_rel = (g_j[:3, :3] @ np.linalg.inv(g_i[:3, :3]))
        angle_fk = _math.degrees(_math.acos(np.clip((np.trace(R_fk_rel) - 1) / 2, -1.0, 1.0)))

        R_tag_rel = (t_j[:3, :3] @ np.linalg.inv(t_i[:3, :3]))
        angle_tag = _math.degrees(_math.acos(np.clip((np.trace(R_tag_rel) - 1) / 2, -1.0, 1.0)))

        diff = abs(angle_fk - angle_tag)
        diffs.append(diff)
        flag = "  <-- LARGE DISAGREEMENT" if diff > 10.0 else ""
        typer.echo(f"{i:>5}->{i + 1:<2}  {angle_fk:9.2f}°  {angle_tag:9.2f}°  {diff:7.2f}°{flag}")

    typer.echo(
        f"\nMean angle diff: {float(np.mean(diffs)):.2f}°  Max: {float(np.max(diffs)):.2f}°"
    )
    if float(np.mean(diffs)) > 10.0:
        typer.echo(
            "⚠️  Large, consistent disagreement between FK-measured and AprilTag-measured "
            "rotation -- since FK is independently verified exact, this points to the AprilTag "
            "pose estimate itself (check: real tag size vs --tag-size-m, tag flatness/rigidity "
            "on the mount, viewing angle/distance during capture, motion blur)."
        )
    else:
        typer.echo(
            "✅ FK and AprilTag measurements are mutually consistent -- a bad solve is more "
            "likely a pose-diversity/conditioning issue (poses too similar/co-planar) than a "
            "measurement-source bug."
        )
    also_small = sum(1 for a in diffs if a < 10.0)
    fk_angles = []
    for i in range(n - 1):
        g_i, g_j = samples[i].gripper2base, samples[i + 1].gripper2base
        R_fk_rel = (g_j[:3, :3] @ np.linalg.inv(g_i[:3, :3]))
        fk_angles.append(_math.degrees(_math.acos(np.clip((np.trace(R_fk_rel) - 1) / 2, -1.0, 1.0))))
    if max(fk_angles) < 15.0:
        typer.echo(
            f"⚠️  Also: max FK-measured rotation between ANY consecutive pair is only "
            f"{max(fk_angles):.1f}° -- the captured poses may simply be too similar/clustered "
            "for a well-conditioned hand-eye solve, independent of measurement accuracy."
        )


def _all_pairs_sample_scores(samples: list[HandEyeSample]) -> list[float]:
    """Per-sample badness score: mean FK-vs-tag rotation-angle disagreement
    across EVERY other sample (not just its consecutive neighbor). Consecutive-
    only comparison (analyze_sample_consistency) can miss a bad sample whose
    neighbors happen to be fine but whose relationship to FAR samples is
    inconsistent -- a real run showed exactly this: one sample appeared in 6
    of the 10 worst (i,j) pairs across the full all-pairs comparison despite
    looking unremarkable against just its immediate neighbors."""
    import math as _math

    n = len(samples)
    per_sample = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g_i, g_j = samples[i].gripper2base, samples[j].gripper2base
            t_i, t_j = samples[i].target2cam, samples[j].target2cam
            R_fk = g_j[:3, :3] @ np.linalg.inv(g_i[:3, :3])
            angle_fk = _math.degrees(_math.acos(np.clip((np.trace(R_fk) - 1) / 2, -1.0, 1.0)))
            R_tag = t_j[:3, :3] @ np.linalg.inv(t_i[:3, :3])
            angle_tag = _math.degrees(_math.acos(np.clip((np.trace(R_tag) - 1) / 2, -1.0, 1.0)))
            diff = abs(angle_fk - angle_tag)
            per_sample[i].append(diff)
            per_sample[j].append(diff)
    return [float(np.mean(d)) for d in per_sample]


def filter_outlier_samples(samples: list[HandEyeSample], min_samples: int = 8) -> list[HandEyeSample]:
    """Greedily drops the single worst-scoring sample (by all-pairs disagreement,
    see _all_pairs_sample_scores), re-solves, and keeps the removal only if the
    mean rotation residual actually improved -- repeats until it stops helping
    or min_samples is reached. Reports what it removed and why, so this is a
    visible, auditable step, not a silent data-massaging trick."""
    current = list(samples)
    best_result = solve_hand_eye(current)
    best_rot = best_result["mean_rotation_residual_deg"]
    typer.echo(f"\n🧹 Outlier filtering: starting residual {best_rot:.2f}° with {len(current)} samples")

    while len(current) > min_samples:
        scores = _all_pairs_sample_scores(current)
        worst_idx = int(np.argmax(scores))
        candidate = [s for i, s in enumerate(current) if i != worst_idx]
        try:
            candidate_result = solve_hand_eye(candidate)
        except ValueError:
            break
        candidate_rot = candidate_result["mean_rotation_residual_deg"]
        if candidate_rot < best_rot:
            typer.echo(
                f"   dropping sample (score {scores[worst_idx]:.1f}°) -> "
                f"residual {best_rot:.2f}° to {candidate_rot:.2f}° ({len(candidate)} left)"
            )
            current = candidate
            best_rot = candidate_rot
        else:
            typer.echo(
                f"   worst remaining sample (score {scores[worst_idx]:.1f}°) wouldn't improve the "
                f"solve ({candidate_rot:.2f}° vs current {best_rot:.2f}°) -- stopping here"
            )
            break

    if len(current) < len(samples):
        typer.echo(f"🧹 Kept {len(current)}/{len(samples)} samples, final residual {best_rot:.2f}°")
    else:
        typer.echo("🧹 No sample removal improved the solve -- keeping all samples")
    return current


def _save_hand_eye_result(samples: list[HandEyeSample], camera_angle: str, tag_size_m: float) -> dict:
    _dump_raw_samples(samples, camera_angle)
    analyze_sample_consistency(samples)
    samples = filter_outlier_samples(samples)
    typer.echo(f"\n📐 Solving with {len(samples)} usable samples...")
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
        "(lower is better; several degrees / cm indicates a poor solve -- re-run with more/better-spread poses)"
    )
    return result


def run_hand_eye_calibration_manual(
    camera_angle: str = "front",
    robot_type: str = "so101",
    follower_id: Optional[str] = None,
    follower_port: Optional[str] = None,
    tag_size_m: float = DEFAULT_TAG_SIZE_M,
    num_samples: int = _NUM_WAYPOINTS,
    min_samples: int = 8,
    use_leader: bool = False,
    leader_id: Optional[str] = None,
    leader_port: Optional[str] = None,
) -> dict:
    """Manual: the script never scripts a move on its own. Two ways to pose
    the arm between captures:
      - use_leader=False (default): torque is disabled right after connecting
        (same mechanism `safe_shutdown()` uses as the hardware's safe-stop,
        just applied at the START instead of the end) so the follower is
        freely back-drivable BY HAND.
      - use_leader=True: torque stays ON, and this runs a real (but minimal)
        teleop mirror loop -- same per-step pattern as lerobot's own
        `teleop_loop()` (read leader action, process, send to follower) --
        so you drive the follower by moving the LEADER arm, reusing the
        leader port/id already saved from normal `solo robo` teleop setup.
    Either way: pose the arm, watch the preview window for a green 'tag
    detected' status, press Enter in THIS TERMINAL to capture (or type
    'done' once you have >= min_samples to solve early)."""
    import json as _json
    import queue
    import threading

    import cv2
    from lerobot.robots import make_robot_from_config
    from pupil_apriltags import Detector

    from solo.commands.robots.lerobot.config import create_follower_config, create_leader_config
    from solo.commands.robots.lerobot.deployx.safety import safe_shutdown

    typer.echo("🎯 SO-101 hand-eye (camera-to-base) extrinsics calibration -- MANUAL mode")
    typer.echo(
        f"   Make sure a {TAG_FAMILY} tag, ID {GRIPPER_TAG_ID}, is mounted on a NON-MOVING part "
        "of the wrist/gripper housing before continuing. The arm will NOT move on a script -- "
        + ("you drive the follower by moving the LEADER arm." if use_leader else "you pose it by hand.")
    )
    if not Confirm.ask("Tag mounted and ready?", default=True):
        typer.echo("Aborted.")
        return {}

    (follower_config_class, follower_port, follower_id, camera_config,
     camera_angle, camera_matrix, dist_coeffs) = _resolve_hand_eye_setup(
        camera_angle, robot_type, follower_id, follower_port
    )

    follower_config = create_follower_config(
        follower_config_class, follower_port, robot_type, camera_config=camera_config, follower_id=follower_id
    )
    detector = Detector(families=TAG_FAMILY)

    teleop = None
    teleop_action_processor = robot_action_processor_pipeline = None
    if use_leader:
        from lerobot.processor import make_default_processors
        from lerobot.teleoperators import make_teleoperator_from_config

        leader_config_class, leader_port, leader_id = _resolve_leader_setup(robot_type, leader_id, leader_port)
        leader_config = create_leader_config(
            leader_config_class, leader_port, robot_type, leader_id=leader_id, follower_id=follower_id
        )
        teleop = make_teleoperator_from_config(leader_config)
        teleop_action_processor, robot_action_processor_pipeline, _ = make_default_processors()

    robot = None
    samples: list[HandEyeSample] = []
    try:
        robot = make_robot_from_config(follower_config)
        robot.connect()

        if use_leader:
            teleop.connect()
            typer.echo(f"🎮 Leader connected ({leader_port}) -- move it to drive the follower.")
        else:
            bus = getattr(robot, "bus", None)
            if bus is not None and hasattr(bus, "disable_torque"):
                bus.disable_torque()
                typer.echo("🔓 Torque disabled -- the arm is free to move by hand.")
            else:
                typer.echo(
                    "⚠️  Could not find a way to disable torque on this robot object -- the arm may "
                    "still be powered/holding position. Move it carefully, or use --use-leader."
                )

        typer.echo(
            f"\nTarget: {num_samples} samples (will solve early with 'done' once you have "
            f">= {min_samples}). For each: pose the arm, watch the preview window for a green "
            "'tag detected' status, then press ENTER in THIS TERMINAL to capture (or type 'done').\n"
        )

        # input() blocks, but macOS requires cv2's GUI window to be pumped
        # from the main thread -- so terminal input runs in a background
        # thread (just blocking stdin I/O, no GUI/hardware calls) and feeds
        # a queue the main thread polls non-blockingly, while the main
        # thread owns the live camera loop AND all robot.get_observation()
        # calls (no concurrent access to the robot object from two threads).
        input_queue: "queue.Queue[str]" = queue.Queue()

        def _input_worker():
            while True:
                try:
                    line = input()
                except EOFError:
                    input_queue.put("done")
                    return
                input_queue.put(line)

        threading.Thread(target=_input_worker, daemon=True).start()

        latest_joint_positions: Optional[dict] = None
        latest_frame = None
        window = "Hand-eye calibration (manual) - pose arm, then ENTER in terminal"

        while len(samples) < num_samples:
            obs = robot.get_observation()

            if use_leader:
                # Same per-step pattern as lerobot's own teleop_loop(): read
                # the leader's action, process it, send it to the follower --
                # continuous mirroring, exactly like normal teleop.
                raw_action = teleop.get_action()
                teleop_action = teleop_action_processor((raw_action, obs))
                robot_action_to_send = robot_action_processor_pipeline((teleop_action, obs))
                robot.send_action(robot_action_to_send)
                obs = robot.get_observation()  # re-read post-move for an up-to-date sample

            latest_joint_positions = {k[: -len(".pos")]: float(v) for k, v in obs.items() if k.endswith(".pos")}
            latest_frame = obs.get(camera_angle)

            if latest_frame is not None:
                tag_found = _detect_gripper_tag(latest_frame, detector, camera_matrix, dist_coeffs, tag_size_m) is not None
                display = latest_frame.copy()
                status = f"Samples: {len(samples)}/{num_samples}  tag_found={tag_found}  (ENTER in terminal to capture)"
                color = (0, 255, 0) if tag_found else (0, 0, 255)
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow(window, display)
            cv2.waitKey(1)  # pumps the GUI event loop so the window actually renders/updates

            try:
                raw = input_queue.get_nowait()
            except queue.Empty:
                continue

            if raw.strip().lower() == "done":
                if len(samples) < min_samples:
                    typer.echo(f"   Need at least {min_samples} samples, only have {len(samples)} -- keep going.")
                    continue
                break

            if latest_frame is None:
                typer.echo("   ⚠️  no camera frame captured, try again")
                continue

            # Settle check: a sample is only valid if the joint encoders agree
            # across two reads taken _SETTLE_CHECK_S apart. Without this, a
            # sample taken while the arm is still catching up to a commanded
            # pose (servo lag, or you pressed Enter mid-motion) pairs a joint
            # reading with a camera frame of a DIFFERENT real pose -- exactly
            # the kind of mismatch that produces a garbage hand-eye solve
            # (confirmed: a real run without this check gave a ~27deg/40cm
            # mean residual, nonsense given the solver math itself was
            # already verified correct on synthetic data).
            time.sleep(_SETTLE_CHECK_S)
            obs2 = robot.get_observation()
            recheck_positions = {k[: -len(".pos")]: float(v) for k, v in obs2.items() if k.endswith(".pos")}
            max_drift = max(
                abs(recheck_positions.get(name, v) - v) for name, v in latest_joint_positions.items()
            )
            if max_drift > _SETTLE_CHECK_MAX_DRIFT:
                typer.echo(f"   ⚠️  still moving (drift {max_drift:.2f}) -- hold still and press ENTER again")
                continue
            latest_joint_positions = recheck_positions
            latest_frame = obs2.get(camera_angle, latest_frame)

            tag_pose_cam = _detect_gripper_tag(latest_frame, detector, camera_matrix, dist_coeffs, tag_size_m)
            if tag_pose_cam is None:
                typer.echo("   ⚠️  tag not detected in this pose -- reposition and try again")
                continue

            gripper_pose_base = compute_gripper_pose(latest_joint_positions)
            samples.append(HandEyeSample(gripper2base=gripper_pose_base, target2cam=tag_pose_cam))
            _dump_sample_frame(latest_frame, len(samples) - 1)
            typer.echo(f"   ✅ captured ({len(samples)}/{num_samples})")

    except KeyboardInterrupt:
        typer.echo("\n🛑 Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        if teleop is not None:
            try:
                if getattr(teleop, "is_connected", False):
                    teleop.disconnect()
            except Exception as e:
                typer.echo(f"⚠️  Error disconnecting leader: {e}")
        safe_shutdown(robot, reason="manual hand-eye calibration finished or interrupted")

    if len(samples) < min_samples:
        typer.echo(f"❌ Only {len(samples)} samples -- need at least {min_samples}. Re-run to collect more.")
        return {}

    return _save_hand_eye_result(samples, camera_angle, tag_size_m)


def run_hand_eye_calibration(
    camera_angle: str = "front",
    robot_type: str = "so101",
    follower_id: Optional[str] = None,
    follower_port: Optional[str] = None,
    tag_size_m: float = DEFAULT_TAG_SIZE_M,
    fps: int = 30,
) -> dict:
    """Autonomous version: scripts the arm through `_NUM_WAYPOINTS` real
    moves on its own. NOT the default entrypoint anymore -- a real run
    produced unexpected/alarming motion, so `run_hand_eye_calibration_manual`
    (no scripted motion at all, you pose the arm by hand) is the default when
    this module is run directly. Kept here for later, once the autonomous
    path's behavior is understood and fixed. Ctrl-C is handled via the same
    `safe_shutdown()` path DeployX's edge agent uses (disables torque,
    disconnects cleanly) rather than leaving the arm powered mid-motion."""
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

    (follower_config_class, follower_port, follower_id, camera_config,
     camera_angle, camera_matrix, dist_coeffs) = _resolve_hand_eye_setup(
        camera_angle, robot_type, follower_id, follower_port
    )

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

    return _save_hand_eye_result(samples, camera_angle, tag_size_m)


def _cli(
    use_leader: bool = typer.Option(False, "--use-leader", help="Drive the follower via the leader arm (teleop) instead of hand-posing it."),
    num_samples: int = typer.Option(_NUM_WAYPOINTS, help="Target number of samples."),
    min_samples: int = typer.Option(8, help="Minimum samples needed to solve."),
    tag_size_m: float = typer.Option(DEFAULT_TAG_SIZE_M, help="Gripper tag physical size, in meters."),
):
    run_hand_eye_calibration_manual(
        use_leader=use_leader, num_samples=num_samples, min_samples=min_samples, tag_size_m=tag_size_m
    )


if __name__ == "__main__":
    typer.run(_cli)
