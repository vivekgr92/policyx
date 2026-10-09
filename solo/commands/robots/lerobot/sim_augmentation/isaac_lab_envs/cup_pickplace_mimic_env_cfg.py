# SPDX-License-Identifier: Apache-2.0
# New (not from NVIDIA): SO-101 3-cup pick-and-place Isaac Mimic env config.
# Adapts real patterns in this repo (vials_to_rack_env_cfg.py's 3-object
# scene structure) + real public isaac-sim/IsaacLab source (tag v2.3.1,
# matching installed isaaclab_mimic==1.0.16) for the SubTaskConfig structure
# and the Differential IK action space (franka/stack_ik_abs_env_cfg.py) --
# CORRECTED from an earlier JointPositionActionCfg attempt: Mimic's
# target_eef_pose_to_action/action_to_target_eef_pose operate on CARTESIAN
# pose actions that Isaac Lab's own DifferentialInverseKinematicsAction term
# resolves to joint commands internally -- not raw joint-space actions.
import os

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaacsim.core.experimental.utils.transform import euler_angles_to_quaternion

# isaacsim.core.utils (used by this file in the original isaaclab 2.3.1 /
# isaacsim 4.x era) was removed from the default extension set in Isaac Sim
# 6.0.1 - confirmed via direct import testing on a real isaaclab
# 3.0.0-beta2-post1 install, even with a fully launched AppLauncher app, not
# just a cold import. isaacsim.core.experimental.utils.transform's
# euler_angles_to_quaternion is the current replacement; same (w, x, y, z)
# convention, but returns a Warp array instead of a bare numpy array.


def euler_angles_to_quat(euler_angles, degrees=False):
    return euler_angles_to_quaternion(euler_angles, degrees=degrees).numpy()

from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import reset_joints_by_offset, JointPositionActionCfg, time_out
from sim_to_real_so101.mdp.cup_mdp import cup_grasped
from .task_env_cfg import SO101TaskSceneCfg, SO101TaskEnvCfg, TaskEventCfg, TaskObservationsCfg

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

_cup_base = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cup",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{assets_path}/usd/Vial_opaque.usda",
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=50.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
)

CUP_SPAWN_Z = 0.05


@configclass
class SO101CupPickPlaceSceneCfg(SO101TaskSceneCfg):
    cup_a = _cup_base.replace()
    cup_a.prim_path = "{ENV_REGEX_NS}/Cup_A"
    cup_a.init_state.pos = (0.22, -0.10, CUP_SPAWN_Z)
    cup_a.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    cup_b = _cup_base.replace()
    cup_b.prim_path = "{ENV_REGEX_NS}/Cup_B"
    cup_b.init_state.pos = (0.22, 0.0, CUP_SPAWN_Z)
    cup_b.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    cup_c = _cup_base.replace()
    cup_c.prim_path = "{ENV_REGEX_NS}/Cup_C"
    cup_c.init_state.pos = (0.22, 0.10, CUP_SPAWN_Z)
    cup_c.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)


@configclass
class CupPickPlaceActionsCfg:
    """Real Differential IK action space (CORRECTED from JointPositionActionCfg
    -- see module docstring). arm_action covers the 5 real arm joints only;
    gripper stays a separate continuous joint-position term since our real
    recorded gripper data is 0-100% continuous, not binary open/close."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        body_name="gripper",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    )
    gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Jaw"],
        scale=1,
        use_default_offset=False,
    )


@configclass
class CupPickPlaceEventCfg(TaskEventCfg):
    reset_cup_a = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0, 0), "velocity_range": (0, 0)},
    )


@configclass
class CupPickPlaceObservationsCfg(TaskObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        grasp_1 = ObsTerm(func=cup_grasped, params={"cup_name": "cup_a"})
        grasp_2 = ObsTerm(func=cup_grasped, params={"cup_name": "cup_b"})
        grasp_3 = ObsTerm(func=cup_grasped, params={"cup_name": "cup_c"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class CupPickPlaceTerminationsCfg:
    """Termination terms for the cup pick-place task.

    isaaclab_mimic's setup_env_config() requires env_cfg.terminations.success
    to exist (raises NotImplementedError otherwise, confirmed via live
    testing) -- it reads and then clears it before running DataGenerator,
    since success there is evaluated out-of-band per generated trial, not
    as a live env termination. Reuses cup_grasped (same validated heuristic
    as the subtask-progress observations above) on cup_b -- matching the
    single-subtask simplification above, this must stay in sync with
    subtask_configs' object_ref -- as a simple proxy for "task done", not a
    rigorous placement check. (Confirmed via a real run: this was pointed at
    cup_c while the subtask targeted cup_b, so 0% of trials ever registered
    as successful even though they ran without crashing.)
    """

    time_out = DoneTerm(func=time_out, time_out=True)
    success = DoneTerm(func=cup_grasped, time_out=False, params={"cup_name": "cup_b"})


@configclass
class SO101CupPickPlaceEnvCfg(SO101TaskEnvCfg):
    scene: SO101CupPickPlaceSceneCfg = SO101CupPickPlaceSceneCfg()
    observations: CupPickPlaceObservationsCfg = CupPickPlaceObservationsCfg()
    terminations: CupPickPlaceTerminationsCfg = CupPickPlaceTerminationsCfg()
    actions: CupPickPlaceActionsCfg = CupPickPlaceActionsCfg()
    events: CupPickPlaceEventCfg = CupPickPlaceEventCfg()


@configclass
class SO101CupPickPlaceMimicEnvCfg(SO101CupPickPlaceEnvCfg, MimicEnvCfg):
    """Mimic env config for the real SO-101 3-cup pick-and-place task.

    Real task structure (confirmed from actual recorded data, 3 episodes,
    vivekgr92/tags dataset, direct user confirmation): each episode grasps
    and places multiple different cups in sequence -- modeled here as one
    grasp subtask per cup.
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_so101_cup_pickplace_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = False  # ABS pose action, see env.py
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # TEMPORARY SIMPLIFICATION, confirmed necessary via a live
        # KeyError('grasp_1') crash in isaaclab_mimic's datagen_info_pool:
        # mimic_hdf5_export.py's _compute_grasped_signal() currently
        # collapses a real multi-cup episode into ONE combined
        # subtask_term_signals["grasped"] boundary + one majority-voted
        # active_object (per episode 0's real data: "cup_b") -- it does not
        # yet emit the per-cup-transition grasp_1/grasp_2 signals a genuine
        # 3-subtask sequence (cup_a -> cup_b -> cup_c) would need. Modeling
        # a single subtask here matches what the exporter actually produces
        # today. Real follow-up: extend _compute_grasped_signal to expose
        # per-segment boundaries with grasp_N-style keys, then restore the
        # 3-subtask loop this replaced (git log has the original version).
        self.subtask_configs["robot"] = [
            SubTaskConfig(
                object_ref="cup_b",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp and place cup_b",
                next_subtask_description=None,
            )
        ]

        # Real crash fix: this Kit build does not recognize the
        # /rtx/translucency/reflectAtAllBounce carb setting that
        # rendering_mode="quality" (inherited from so101_env_cfg.py,
        # affects ALL SO-101 envs, not something this Mimic env introduced)
        # tries to set -- SimulationContext._apply_render_settings_from_cfg
        # raises ValueError on it. Likely an Isaac Sim/Kit version drift
        # since this pod was last used. Downgrading to the default render
        # mode here to unblock scene loading.
        self.sim.render.rendering_mode = "performance"
        self.sim.render.enable_translucency = True
        # REAL root cause (found by tracing _apply_render_settings_from_cfg's
        # source, not guessed): task_env_cfg.py's SO101TaskEnvCfg.__post_init__
        # sets self.sim.render.carb_settings to a dict containing
        # rtx.translucency.reflectAtAllBounce and other RTX path-tracing keys
        # that do not exist on this pod's installed Kit build (isaacsim
        # 5.1.0.0) -- a real version-drift issue, inherited unchanged from
        # the base task config used by every SO-101 env. Clearing it here is
        # the actual fix (my two earlier attempts touched different fields).
        self.sim.render.carb_settings = None


# Registered here (not in the upstream sim_to_real_so101/tasks/__init__.py)
# because, as of isaaclab 3.0.0-beta2-post1, isaaclab_tasks.utils.import_packages
# only auto-imports __init__.py packages, not bare .py modules -- so a
# gym.register() living in a plain env_cfg.py file like this one is never
# auto-discovered. Confirmed by direct testing: import_packages walks right
# past this file. An external caller must explicitly import this module
# (see register_cup_task.py-style callback) to trigger this registration.
import gymnasium as gym  # noqa: E402

gym.register(
    id="Lerobot-So101-Cup-PickPlace-Mimic",
    entry_point="sim_to_real_so101.tasks.cup_pickplace_mimic_env:SO101CupPickPlaceMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:SO101CupPickPlaceMimicEnvCfg",
    },
)

