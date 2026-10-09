# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Solo CLI is a Python CLI tool for Physical AI - enabling model inference on edge hardware, robotics operations (motor control, teleoperation, training), and model deployment via Ollama, vLLM, or llama.cpp servers. It also integrates with Solo Hub for authentication and model downloads.

## Development Setup

```bash
# Prerequisites: Git LFS, uv package manager
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Common Commands

```bash
# Run the CLI
solo

# Interactive setup (hardware detection, server config, saves to ~/.solo/config.json)
solo setup

# Linting/formatting (no config files - uses defaults)
black solo/
isort solo/

# Tests
pytest
```

## Architecture

**CLI Framework:** Typer with Rich for terminal UI. Entry point is `solo/cli.py`.

**Lazy-loaded commands:** All CLI commands in `solo/commands/` are imported inside function bodies in `cli.py` for startup performance. Follow this pattern when adding new commands.

**Key modules:**

- `solo/cli.py` - Typer command definitions (lazy imports from commands/)
- `solo/main.py` - Setup wizard logic (hardware detection, config generation)
- `solo/commands/` - Individual command implementations (serve, status, login, download, etc.)
- `solo/hub/` - Solo Hub integration (auth via device code flow, API client, model caching)
- `solo/utils/server_utils.py` - Server lifecycle management for Ollama/vLLM/llama.cpp (large file, ~1200 lines)
- `solo/utils/hardware.py` - Hardware detection (CPU, GPU - NVIDIA/AMD/Apple Silicon, memory)
- `solo/config/` - YAML defaults (`config.yaml`) + user JSON config (`~/.solo/config.json`)
- `solo/commands/robots/lerobot/` - LeRobot robotics integration (calibration, teleop, recording, training, inference)
- `solo/commands/robots/lerobot/teleoperators/` - Solo-provided LeRobot teleoperators, e.g. the Star Arm 102 (StarAI/Fashionstar) leader remapped onto SO101 joints; paired with `starai_config.py` (mapping saved in `~/.solo/starai_map.json`) and `starai_tune.py` (`solo robo --star-tune`)
- `solo/commands/robots/lerobot/vlm_judge.py` - VLM-based data-quality judge for `solo robo --replay --loop --perturb --vlm-judge`: judges each replayed episode VALID/INVALID against `vlm_judge_rules.md`'s editable rubric before it's kept, fail-closed (discard on any verdict other than an explicit pass). Backends: `gemini` (default, `gemini-robotics-er-2-preview`) or `ollama`, chosen interactively at replay time; `runpod` is reachable only via `VLM_JUDGE_BACKEND=runpod`. `vlm_judge_playground.py` is a standalone Gradio tool for manually testing the judge against a live camera (run as `python3 -m solo.commands.robots.lerobot.vlm_judge_playground`, not as a direct script path - this directory's own `lerobot.py` shadows the real `lerobot` package otherwise). `vlm_judge_test_fixtures/` holds known adversarial test episodes.
- `solo/commands/robots/lerobot/camera_calibration.py` - Camera intrinsics calibration (checkerboard-based), step 1 of the AprilTag object-pose pipeline below. Run as a module: `python3 -m solo.commands.robots.lerobot.camera_calibration <generate-pattern|capture|solve|run>` (same shadowing issue as above). Saves to `~/.solo/camera_calibration.json`.
- `solo/commands/robots/lerobot/sim_augmentation/` - Isaac Mimic / Cosmos sim-augmentation pipeline: scales a small set of real validated SO-101 demos into synthetic training data (Isaac Lab Mimic for trajectory variation, Cosmos for photorealistic re-rendering). Generic by design - any SO-101 LeRobot dataset (HF Hub repo id or local path) can be passed through it, not hardcoded to one dataset.
  - `dataset_loader.py` - generic LeRobot v3 dataset loader, zero Isaac/torch dependency.
  - `so101_joint_mapping.py` / `so101_fk.py` - SO-101 joint-name/unit mapping (LeRobot recorded degrees -> Isaac Lab/USD radians) and a lightweight local FK (via `ikpy` against a vendored real SO-101 URDF in `assets/`), independent of the Isaac Lab/GPU stack.
  - `hand_eye_calibration.py` - scripted camera-to-robot-base extrinsics calibration (AprilTag on the gripper + arm sweep + `cv2.calibrateHandEye`, eye-to-hand configuration). Run via `python -m solo.commands.robots.lerobot.sim_augmentation.hand_eye_calibration`. Saves to `~/.solo/camera_extrinsics.json`. Requires camera intrinsics calibrated first.
  - `apriltag_pose.py` - generates printable `tag36h11` AprilTags and extracts per-frame object poses from recorded video (camera frame -> robot-base frame via the hand-eye extrinsics), with SLERP/linear gap interpolation for occlusion.
  - `mimic_hdf5_export.py` - converts a recorded LeRobot dataset into the exact HDF5 schema `isaaclab_mimic` reads (`obs/datagen_info/{eef_pose,target_eef_pose,object_pose,subtask_term_signals}`, confirmed against its real source, not assumed). Handles multi-episode-per-video-file packing (reads each episode's real `videos/<camera>/{chunk_index,file_index,from_timestamp,to_timestamp}` from `meta/episodes/`, never guesses from episode index) and multi-cycle grasp detection (multiple pick/place cycles touching different objects within one episode, not just the first).
  - `so101_ik_bridge.py` - SO-101 Franka-style ABS/REL action encoding (`target_eef_pose_to_action`/`action_to_target_eef_pose`) for the Isaac Lab Mimic bridge, plus a secondary `ikpy`-based `solve_ik_for_pose` utility. Confirmed real limitation: SO-101 is 5-DOF, so arbitrary full 6-DOF (position+orientation) pose targets aren't always simultaneously achievable - position nails exactly, orientation can be 22-134° off with a neutral initial IK guess. Not yet committed to git.
  - **Camera intrinsics + hand-eye extrinsics calibration: solved.** `camera_calibration.py` uses a record-video-then-extract workflow (`record-video`/`extract-from-video`/`solve-video` commands) since live SPACE/Q keyboard capture is unreliable on macOS cv2 windows. `hand_eye_calibration.py`'s default entrypoint (`run_hand_eye_calibration_manual`, supports `--use-leader` for leader-arm teleop control) is now fully manual - no autonomous scripted arm motion - and every run automatically greedy-filters outlier samples (`filter_outlier_samples`, all-pairs FK-vs-AprilTag disagreement scoring) before solving, which took a real run from 18.34° to 2.85° mean rotation residual.
  - `isaac_lab_envs/` - custom SO-101 Isaac Lab Mimic env (`cup_pickplace_mimic_env_cfg.py`/`cup_pickplace_mimic_env.py`/`cup_mdp.py`), originally built on a Runpod network volume and now committed here so that volume is no longer a dependency. Built against `isaaclab==2.3.1`/`isaaclab_mimic==1.0.16`.
  - **Isaac Lab Mimic `DataGenerator.generate()` - runs end-to-end, not yet succeeding.** Confirmed live on `../../infra/hyperstack/`'s Hyperstack GPU host (L40, Isaac Sim 6.0.1 / Isaac Lab 3.0.0-beta2-post1) against `isaac_lab_envs/`'s custom SO-101 env, deployed on top of a fresh clone of the public upstream `isaac-sim/Sim-to-Real-SO-101-Workshop` repo. 5 real, confirmed-by-evidence bugs found and fixed across a long debugging session, in order:
    1. `isaacsim.core.utils`/`isaacsim.core.prims` removed from Isaac Sim 6.0.1's default extension set (confirmed: `--enable isaacsim.core.prims` fails with "no versions satisfy" from Kit's own dependency solver - not just disabled, gone). Fixed in our file (swapped to `isaacsim.core.experimental.utils.transform.euler_angles_to_quaternion`) and in 4 upstream files via a patch (`infra/hyperstack/sim-to-real-so101-workshop-isaac-sim-6-compat.patch`, applied by `infra/hyperstack/deploy-cup-mimic-env.sh`). Also: `isaacsim.core.experimental.prims` exists but isn't enabled by default - pass `--enable isaacsim.core.experimental.prims` on the CLI.
    2. The upstream repo's USD robot/scene assets are Git LFS pointer files (~130 bytes) on a plain clone, not the real binaries - Isaac Sim spawns the pointer file as the "robot" without erroring, then fails much later with a confusing `RuntimeError: Expected exactly one ArticulationRootAPI prim ... found 0`. Fix: `git lfs install && git lfs pull` after cloning (now in `deploy-cup-mimic-env.sh`).
    3. `isaaclab_mimic`'s `setup_env_config()` requires `env_cfg.terminations.success` to exist (`NotImplementedError` otherwise) - added `CupPickPlaceTerminationsCfg.success` using `cup_grasped`.
    4. `isaaclab 3.0.0-beta2`'s `isaacsim.core.experimental.prims.XformPrim`/`Articulation.data.*_w` properties return Warp `ProxyArray` objects, not plain torch Tensors (`.torch` materializes them) - fixed in `mdp/obs.py` (upstream, patched) and `cup_mdp.py` (ours).
    5. **Episode/success-schema mismatches in our own exporter and env cfg**, found via 3 parallel debugging subagents against a live trial: `mimic_hdf5_export.py`'s `eef_pose`/`target_eef_pose` needed per-eef-name HDF5 grouping (not a flat array) to match what `DatagenInfo` expects; the exported episode needed trimming to just the active cup's own pick-place cycle (`_compute_grasped_signal` now returns a `trim_window`) instead of the full multi-cup recording, since exporting the whole thing as "one subtask" fed Mimic a motion arc that included releasing the cup to grasp the next one; `cup_pickplace_mimic_env_cfg.py`'s success term checked `cup_c` while the (necessarily simplified, see below) subtask targeted `cup_b`; `cup_grasped`'s `gripper_closed_threshold=0.0` was mathematically unreachable for this dataset's real recorded gripper values (24-32%) given `so101_joint_mapping.py`'s real `to_sim_radians` conversion (`nominal_limit_rad=(-0.174533, 1.74533)`, `GRIPPER_PCT_MAPS_TO_LOWER_AT_ZERO=True` maps that range to **positive** radians, never negative) - fixed to `0.40`.
    - **Current state after all 5 fixes: still 0% success, but very close** - live instrumentation showed minimum distance-to-cup of `0.0717m`, just 1.2cm short of the `0.06m` proximity threshold, with the gripper fully open exactly at closest approach (closes only after the arm has already retreated - a temporal mismatch between the retargeted arm motion and the replayed gripper action). Tried setting `SubTaskConfig.num_fixed_steps` from `0` to `30` (theory: with 0 fixed steps there's no phase after reaching the interpolated target where the real recorded gripper-closing action actually replays) - **did not resolve it**; a live debug run after that fix showed `jaw_pos` frozen at `1.7453` (max-open) with repeating identical distance values across many consecutive calls, suggesting `num_fixed_steps` isn't doing what was assumed, or there's a further bug in how the gripper action gets applied during that phase. **Next step**: read `isaaclab_mimic/datagen/waypoint.py` and `data_generator.py`'s actual handling of `num_fixed_steps` (on the VM: `/workspace/isaaclab/source/isaaclab_mimic/isaaclab_mimic/datagen/`) to find the real mechanism, rather than guessing at more parameter values.
    - Separately unresolved, and now root-caused (not just a camera-framing guess): the WebRTC viewer shows the Isaac Sim window but not the robot. Verified directly via `omni.kit.viewport.utility.capture_viewport_to_file()` (not by trusting user reports of the stream) - a real screenshot taken with `--livestream 2` shows the scene genuinely rendering (one small gray cup placeholder box is clearly visible, correctly lit), but **the SO-101 robot's visual mesh never appears**, despite its physics being 100% confirmed working (joint reads, grasp detection all correct). So this is not a camera-angle problem - the simple cup asset renders fine at the same framing where the robot should be. Points to a material/texture loading failure or a `visibility`/prim-path mismatch specific to the robot's real USD asset (23MB, real PBR materials) vs. the simple placeholder cup geometry, under this Isaac Sim 6.0.1 build. Next step: inspect the robot's USD for a `visibility=invisible` attribute or broken material binding, and/or check Kit's log for silent material/shader load failures scoped to the robot's prim path specifically. A `--/isaaclab/has_gui=true` CLI override was tried first (wrong theory, camera-framing) and made no difference, consistent with this being a rendering-not-framing issue.
    - Reproducible test harness: `infra/hyperstack/deploy-cup-mimic-env.sh` (clones the workshop repo, applies the compat patch, drops in `isaac_lab_envs/`'s files, writes the external-callback registration module), then from `/workspace/isaaclab`: `./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Lerobot-So101-Cup-PickPlace-Mimic --input_file <hdf5> --output_file <out.hdf5> --generation_num_trials 5 --num_envs 1 --headless --enable isaacsim.core.experimental.prims --external_callback register_cup_task.register`. A real source HDF5 is reproducible via `python3 -m solo.commands.robots.lerobot.sim_augmentation.mimic_hdf5_export --dataset vivekgr92/tags --output <path>.hdf5 --tag-ids '{"cup_a": 0, "cup_b": 1, "cup_c": 2}' --episodes "0"` (needs `h5py`, `ikpy`, `pupil-apriltags`, `opencv-python-headless`, `huggingface_hub`, `pyarrow` in a plain venv - zero Isaac/torch dependency).
    - The gripper open/close direction in `so101_joint_mapping.py` is no longer purely unverified - the `GRIPPER_PCT_MAPS_TO_LOWER_AT_ZERO=True` convention is now confirmed consistent with live sim behavior (closed gripper = higher percentage maps toward the negative end of the real joint range, per the threshold fix above). Cosmos (photorealistic re-rendering) hasn't been touched at all yet.
- `infra/hyperstack/` - Scripts to reproduce the working Isaac Sim GPU setup on a fresh Hyperstack VM without redoing the manual debugging (driver version floor, Docker GPU runtime registration, WebRTC viewer public-reachability fix). See its README for the full writeup, including why Runpod and Lambda Cloud were both ruled out as structural dead ends (broken Vulkan - a shared-host container isolation bug on Runpod, a broken driver on Lambda's stock image) and why AWS was blocked (account-wide EC2 block, not Marketplace-specific). Uses NVIDIA's own `isaac-sim/isaac-launchable` repo (containerized Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2-post1 + WebRTC streaming viewer) rather than a bare-metal AMI.
- `solo/mcp/` - Domain-specific Model Context Protocol implementations (agriculture, education, healthcare, etc.)

**Multi-platform support:** Hardware detection and server selection adapt to CUDA (NVIDIA), HIP (AMD), Metal (Apple Silicon), and CPU-only environments.

**Configuration:** Two-tier system - YAML defaults in `solo/config/config.yaml`, user settings in `~/.solo/config.json`.
