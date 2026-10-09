# Real, new (not from NVIDIA) -- per-cup grasp detection for the SO-101 Isaac
# Mimic pick-place task. Uses the same gripper-closed + object-proximity
# heuristic already validated on real recorded data in this project's
# mimic_hdf5_export.py (_compute_grasped_signal), rather than the vials
# task's contact-sensor approach, since that needs force/height tuning
# this task doesn't have time to validate in sim.
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cup_grasped(
    env: ManagerBasedRLEnv,
    cup_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_joint_name: str = "Jaw",
    gripper_closed_threshold: float = 0.0,
    proximity_threshold: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when the gripper is closed past threshold AND within
    proximity_threshold meters of the named cup's current position. Matches
    the real approach already validated on recorded hardware data (gripper
    value + nearest-object distance), not a new/untested heuristic."""
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    cup = env.scene[cup_name]

    jaw_idx = robot.data.joint_names.index(gripper_joint_name)
    jaw_pos = robot.data.joint_pos[:, jaw_idx]
    is_closed = jaw_pos < gripper_closed_threshold

    # .torch: cup.data.root_pos_w returns a Warp ProxyArray (confirmed via
    # live isaaclab 3.0.0-beta2 testing), while ee_frame.data.target_pos_w
    # and robot.data.joint_pos are plain Tensors -- same class of bug fixed
    # in obs.py earlier, missed here. Didn't corrupt the numbers in practice
    # (torch.linalg.norm tolerated the mixed types), but fixing for
    # correctness/forward-compat.
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    cup_pos_w = cup.data.root_pos_w.torch
    dist = torch.linalg.norm(ee_pos_w - cup_pos_w, dim=-1)
    is_near = dist < proximity_threshold

    return (is_closed & is_near).unsqueeze(-1)
