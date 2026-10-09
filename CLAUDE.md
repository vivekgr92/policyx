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
  - **Known open gaps, not yet resolved:** Isaac Lab Mimic's actual trajectory-scaling step (`DataGenerator.generate()`) has not yet been run end-to-end against real exported HDF5 - a custom SO-101 Mimic env (`cup_pickplace_mimic_env_cfg.py`/`cup_pickplace_mimic_env.py`/`cup_mdp.py`, built against `isaaclab==2.3.1`/`isaaclab_mimic==1.0.16` on a Runpod volume) exists but was never verified to actually generate trajectories. `../../infra/hyperstack/` now provides a working GPU host (see below) to finally test this - but that stack ships newer `isaaclab==3.0.0-beta2-post1`/`isaaclab_mimic==1.3.3`, so check for API drift before assuming the env code runs unmodified. The gripper open/close direction in `so101_joint_mapping.py` is flagged `UNVERIFIED` against real data. Cosmos (photorealistic re-rendering) hasn't been touched at all yet.
- `infra/hyperstack/` - Scripts to reproduce the working Isaac Sim GPU setup on a fresh Hyperstack VM without redoing the manual debugging (driver version floor, Docker GPU runtime registration, WebRTC viewer public-reachability fix). See its README for the full writeup, including why Runpod and Lambda Cloud were both ruled out as structural dead ends (broken Vulkan - a shared-host container isolation bug on Runpod, a broken driver on Lambda's stock image) and why AWS was blocked (account-wide EC2 block, not Marketplace-specific). Uses NVIDIA's own `isaac-sim/isaac-launchable` repo (containerized Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2-post1 + WebRTC streaming viewer) rather than a bare-metal AMI.
- `solo/mcp/` - Domain-specific Model Context Protocol implementations (agriculture, education, healthcare, etc.)

**Multi-platform support:** Hardware detection and server selection adapt to CUDA (NVIDIA), HIP (AMD), Metal (Apple Silicon), and CPU-only environments.

**Configuration:** Two-tier system - YAML defaults in `solo/config/config.yaml`, user settings in `~/.solo/config.json`.
