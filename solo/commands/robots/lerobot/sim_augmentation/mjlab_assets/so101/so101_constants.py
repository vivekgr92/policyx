"""SO-101 constants for mjlab.

Vendored model: DeepMind mujoco_menagerie lineage (TheRobotStudio/SO-ARM100),
Apache 2.0. See ../so101/SOURCE.md. The MJCF already defines reasonable
position actuators for every joint (class "sts3215", kp/kv tuned from the
real STS3215 servo), so this first-step config relies on those native
actuators directly (articulation=None) rather than redefining PD gains from
motor torque/inertia specs, unlike i2rt_yam's yam_constants.py pattern.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

##
# MJCF and assets.
##
SO101_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "so101" / "so101.xml"
assert SO101_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(SO101_XML))


##
# Keyframe config.
##
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)


def get_so101_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        spec_fn=get_spec,
        articulation=None,
    )


if __name__ == "__main__":
    from mjlab.entity.entity import Entity

    robot = Entity(get_so101_robot_cfg())
    model = robot.spec.compile()
    print("[OK] SO-101 entity built and spec compiled.")
    print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu)
    print("joint names:", [model.joint(i).name for i in range(model.njnt)])
    print("actuator names:", [model.actuator(i).name for i in range(model.nu)])

    import mujoco as mj

    data = mj.MjData(model)
    for i in range(50):
        mj.mj_step(model, data)
    print("[OK] stepped 50 times without error.")
    print("final qpos:", data.qpos)
