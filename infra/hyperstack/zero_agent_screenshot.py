# Loads the registered task and captures one real viewport screenshot to
# /tmp/so101_screenshot.png -- used to verify what's actually rendering
# without trusting a human's description of the WebRTC stream. Real finding
# from this: the scene genuinely renders (a cup placeholder box is visible,
# correctly lit), but the SO-101 robot's visual mesh never appears despite
# its physics being fully loaded -- a material/visibility bug, not a camera
# framing issue (see CLAUDE.md sim_augmentation section).
#
# Usage (from /workspace/isaaclab on the VM, after deploy-cup-mimic-env.sh):
#   export PYTHONPATH=/workspace/Sim-to-Real-SO-101-Workshop/source:$PYTHONPATH
#   ./isaaclab.sh -p zero_agent_screenshot.py --task Lerobot-So101-Cup-PickPlace-Mimic \
#     --num_envs 1 --livestream 2 --kit_args "--enable isaacsim.core.experimental.prims"
# Then: docker cp vscode:/tmp/so101_screenshot.png . and pull it off the VM.
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks
import sim_to_real_so101.tasks.cup_pickplace_mimic_env_cfg
import omni.kit.viewport.utility as vp_util

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg)
print("[INFO] env created, resetting...")
env.reset()
print("[INFO] env reset OK")
for _ in range(60):
    simulation_app.update()

vp = vp_util.get_active_viewport()
vp_util.capture_viewport_to_file(vp, "/tmp/so101_screenshot.png")
for _ in range(30):
    simulation_app.update()
print("[INFO] screenshot saved")
