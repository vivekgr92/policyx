# SPDX-License-Identifier: Apache-2.0
# New (not from NVIDIA): SO-101 Isaac Mimic env subclass. Real reference,
# fetched from public isaac-sim/IsaacLab source at tag v2.3.1 (matching
# installed isaaclab_mimic==1.0.16): franka_stack_ik_abs_mimic_env.py +
# pick_place_mimic_env.py.
#
# CORRECTED (per the parallel IK-bridge task's real finding): Isaac Lab's
# DifferentialInverseKinematicsAction action term (now configured in
# cup_pickplace_mimic_env_cfg.py's arm_action) does the joint-level IK
# itself during simulation -- target_eef_pose_to_action/
# action_to_target_eef_pose just pack/unpack a Cartesian pose action
# ([pos(3), quat_wxyz(4), gripper(1)] = 8 dims, ABS mode), no FK/IK math
# needed in this file at all.
from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class SO101CupPickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for the real SO-101 3-cup
    pick-and-place task (Differential IK, absolute pose action)."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Real EE pose from the scene's own ee_frame FrameTransformer
        (LerobotSo101BaseSceneCfg, targets {ENV_REGEX_NS}/Robot/gripper) --
        this env's ObservationsCfg doesn't expose eef_pos/eef_quat under
        those obs_buf keys the way Franka's env does, so this reads the
        real scene sensor directly instead (same data flow as
        mdp/obs.py's ee_frame_state)."""
        if env_ids is None:
            env_ids = slice(None)
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]
        root_pos, root_quat = robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids]
        ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
        ee_quat_w = ee_frame.data.target_quat_w[env_ids, 0, :]
        ee_pos_b, ee_quat_b = PoseUtils.subtract_frame_transforms(root_pos, root_quat, ee_pos_w, ee_quat_w)
        return PoseUtils.make_pose(ee_pos_b, PoseUtils.matrix_from_quat(ee_quat_b))

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Verbatim real pattern from isaaclab_mimic's generic
        PickPlaceAbsMimicEnv.get_object_poses -- fully generic, no
        SO-101-specific change needed."""
        if env_ids is None:
            env_ids = slice(None)
        scene_state = self.scene.get_state(is_relative=True)
        rigid_object_states = scene_state["rigid_object"]
        articulation_states = scene_state["articulation"]
        robot_root_pose = articulation_states["robot"]["root_pose"]
        root_pos = robot_root_pose[env_ids, :3]
        root_quat = robot_root_pose[env_ids, 3:7]
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, obj_state["root_pose"][env_ids, :3], obj_state["root_pose"][env_ids, 3:7]
            )
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos_obj_base, PoseUtils.matrix_from_quat(quat_obj_base))
        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "grasp_1": subtask_terms["grasp_1"][env_ids],
            "grasp_2": subtask_terms["grasp_2"][env_ids],
            "grasp_3": subtask_terms["grasp_3"][env_ids],
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # last dim is the separate gripper_action term (see env cfg)
        return {"robot": actions[:, -1:]}

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Real ABS pattern (franka_stack_ik_abs_mimic_env.py): the action
        IS the target pose directly, no current-pose/delta math at all --
        Isaac Lab's own DifferentialInverseKinematicsAction term resolves
        this to joint commands internally when the action is applied."""
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0)
        if action_noise_dict is not None:
            eef_name = list(self.cfg.subtask_configs.keys())[0]
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action = pose_action + noise

        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inverse of target_eef_pose_to_action -- just unpacks the action's
        pos+quat, no FK needed since it's an absolute pose action."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        target_pos = action[:, :3]
        target_quat = action[:, 3:7]
        target_rot = PoseUtils.matrix_from_quat(target_quat)
        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}
