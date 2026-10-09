"""
SO-101 pose<->action bridge for Isaac Lab Mimic's `ManagerBasedRLMimicEnv`
interface (`target_eef_pose_to_action` / `action_to_target_eef_pose`).

REAL FINDING (read directly from isaac-sim/IsaacLab's actual source at tag
v2.3.1 -- isaaclab_mimic/envs/franka_stack_ik_{abs,rel}_mimic_env.py -- not
guessed): these methods do NOT produce or consume raw joint-space actions.
Both Franka reference variants operate on a CARTESIAN pose action:
  - ABS: [pos(3), quat_wxyz(4), gripper(1)] = 8 dims, action IS the target
    pose directly (no current-pose math needed at all).
  - REL: [delta_pos(3), delta_rot_axis_angle(3), gripper(1)] = 7 dims,
    action is a DELTA from the CURRENT eef pose.
In both cases, the actual joint-level IK solve happens INSIDE Isaac Lab's
own `DifferentialInverseKinematicsAction` term when the action is applied
to the robot during simulation -- not in this bridge, and not via ikpy.

CONSEQUENCE FOR THE SO-101 SCENE/ENV CONFIG (not something this pure-Python
module can do -- flagging for whoever builds that scene): the SO-101 Isaac
Lab task currently uses `JointPositionActionCfg` (raw joint-position
actions). Making Mimic work requires reconfiguring that env's action space
to use a Differential IK action term (abs or rel) instead -- a real
scene/action-manager change, not optional.

This module provides both the ABS and REL bridge functions using the exact
same math as Franka's real reference implementation, with SO-101's
`compute_gripper_pose` (so101_fk.py) standing in for the live
`get_robot_eef_pose` the real env would read from its own observation
buffer -- since no such obs term exists for SO-101 yet, callers supply
current joint positions (degrees, LeRobot convention) and this module
computes current EEF pose via the same proven FK chain.

Also provides a separate, optional ikpy-based raw-joint-space IK solver
(`solve_ik_for_pose`) as a real, independently useful utility in case a
joint-space action path turns out to be preferred over Differential IK for
the SO-101 scene -- not used by the ABS/REL bridge functions above.

Pure numpy, no torch/Isaac Lab dependency -- importable standalone on this
Mac, or from the Isaac Lab Python env on a GPU pod (converting numpy<->torch
at the call site there).
"""

from __future__ import annotations

import math

import numpy as np

from solo.commands.robots.lerobot.sim_augmentation.so101_fk import _get_chain, compute_gripper_pose

__all__ = [
    "get_current_eef_pose",
    "target_eef_pose_to_action_abs",
    "action_to_target_eef_pose_abs",
    "target_eef_pose_to_action_rel",
    "action_to_target_eef_pose_rel",
    "solve_ik_for_pose",
]


# ---------------------------------------------------------------------------
# Shared pose<->quaternion helpers (wxyz order, matching Isaac Lab's
# `PoseUtils.quat_from_matrix` / `matrix_from_quat` convention confirmed in
# the real Franka reference: "Quaternion format is w,x,y,z").
# ---------------------------------------------------------------------------


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Standard Shepperd's-method rotation-matrix -> quaternion (w,x,y,z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis*angle (3-vector), matching Isaac Lab's
    `axis_angle_from_quat` composed with `quat_from_matrix` in the real
    Franka REL reference."""
    angle = math.acos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    ) / (2 * math.sin(angle))
    return axis * angle


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Inverse of _matrix_to_axis_angle (Rodrigues' formula). Identity if
    the angle is ~0, matching the real Franka REL reference's explicit
    close-to-zero handling."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64
    )
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# Current EEF pose (stand-in for the live env's obs_buf eef_pos/eef_quat)
# ---------------------------------------------------------------------------


def get_current_eef_pose(recorded_deg: dict) -> np.ndarray:
    """Current gripper_frame_link pose (4x4), via the same proven FK chain
    so101_fk.py already builds and validates -- not a second chain."""
    return compute_gripper_pose(recorded_deg)


# ---------------------------------------------------------------------------
# ABS variant -- pure encode/decode, no current-pose math needed at all
# (matches franka_stack_ik_abs_mimic_env.py exactly).
# ---------------------------------------------------------------------------


def target_eef_pose_to_action_abs(target_eef_pose: np.ndarray, gripper_action: float) -> np.ndarray:
    """[pos(3), quat_wxyz(4), gripper(1)] = 8-dim action, directly encoding
    the target pose -- no current pose involved (ABS mode)."""
    pos = target_eef_pose[:3, 3]
    quat = _matrix_to_quat_wxyz(target_eef_pose[:3, :3])
    return np.concatenate([pos, quat, [gripper_action]]).astype(np.float32)


def action_to_target_eef_pose_abs(action: np.ndarray) -> np.ndarray:
    """Inverse of target_eef_pose_to_action_abs: 8-dim action -> 4x4 pose."""
    pos = action[:3]
    quat = action[3:7]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quat_wxyz_to_matrix(quat)
    pose[:3, 3] = pos
    return pose


# ---------------------------------------------------------------------------
# REL variant -- delta from current pose (matches
# franka_stack_ik_rel_mimic_env.py exactly, including its explicit
# near-zero-rotation handling).
# ---------------------------------------------------------------------------


def target_eef_pose_to_action_rel(
    target_eef_pose: np.ndarray, current_eef_pose: np.ndarray, gripper_action: float, clamp: bool = False
) -> np.ndarray:
    """[delta_pos(3), delta_rot_axis_angle(3), gripper(1)] = 7-dim action."""
    delta_position = target_eef_pose[:3, 3] - current_eef_pose[:3, 3]
    delta_rot_mat = target_eef_pose[:3, :3] @ current_eef_pose[:3, :3].T
    delta_rotation = _matrix_to_axis_angle(delta_rot_mat)
    pose_action = np.concatenate([delta_position, delta_rotation])
    if clamp:
        pose_action = np.clip(pose_action, -1.0, 1.0)
    return np.concatenate([pose_action, [gripper_action]]).astype(np.float32)


def action_to_target_eef_pose_rel(action: np.ndarray, current_eef_pose: np.ndarray) -> np.ndarray:
    """Inverse of target_eef_pose_to_action_rel."""
    delta_position = action[:3]
    delta_rotation = action[3:6]
    target_pos = current_eef_pose[:3, 3] + delta_position
    delta_rot_mat = _axis_angle_to_matrix(delta_rotation)
    target_rot = delta_rot_mat @ current_eef_pose[:3, :3]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = target_rot
    pose[:3, 3] = target_pos
    return pose


# ---------------------------------------------------------------------------
# Optional: raw joint-space IK (NOT used by the ABS/REL bridge above -- a
# separate utility in case a joint-space Mimic action path is preferred over
# Differential IK for the SO-101 scene).
# ---------------------------------------------------------------------------


def solve_ik_for_pose(target_pose: np.ndarray, initial_guess_deg: dict | None = None) -> dict:
    """
    Inverse kinematics via ikpy's real solver, using the SAME chain object
    so101_fk.py builds (not a second one). Returns a dict of real joint
    names -> degrees for the 5 arm joints.

    Raises ValueError if the solve doesn't converge to within a real,
    checked tolerance (does not silently return a garbage/unreachable
    solution) -- callers should catch this for out-of-workspace targets.
    """
    chain = _get_chain()
    from solo.commands.robots.lerobot.sim_augmentation.so101_fk import _ARM_JOINT_NAMES

    initial = [0.0] * len(chain.links)
    if initial_guess_deg:
        for i, link in enumerate(chain.links):
            if link.name in _ARM_JOINT_NAMES:
                initial[i] = initial_guess_deg.get(link.name, 0.0) * math.pi / 180.0

    solution = chain.inverse_kinematics_frame(target_pose, initial_position=initial)
    achieved = chain.forward_kinematics(solution)

    pos_err = float(np.linalg.norm(achieved[:3, 3] - target_pose[:3, 3]))
    rot_err_deg = math.degrees(
        math.acos(np.clip((np.trace(achieved[:3, :3].T @ target_pose[:3, :3]) - 1) / 2, -1.0, 1.0))
    )
    if pos_err > 0.01 or rot_err_deg > 5.0:  # 1cm / 5deg -- target likely unreachable
        raise ValueError(
            f"IK did not converge to the target pose (pos_err={pos_err * 1000:.1f}mm, "
            f"rot_err={rot_err_deg:.1f}deg) -- target is likely outside the real reachable "
            "workspace, or too close to a singularity."
        )

    out = {}
    for i, link in enumerate(chain.links):
        if link.name in _ARM_JOINT_NAMES:
            out[link.name] = math.degrees(solution[i])
    return out
