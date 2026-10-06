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
  - `mimic_hdf5_export.py` - converts a recorded LeRobot dataset into the exact HDF5 schema `isaaclab_mimic` 1.0.16 reads (`obs/datagen_info/{eef_pose,target_eef_pose,object_pose,subtask_term_signals}`, confirmed against its real source, not assumed).
  - **Known open gaps, not yet resolved:** Isaac Lab Mimic's actual trajectory-scaling step (`DataGenerator.generate()`) still needs a custom SO-101 `MimicEnvCfg`/env authored in Isaac Lab - `mimic_hdf5_export.py` only produces Mimic-ready *source* annotations, it doesn't run Mimic itself. The gripper open/close direction in `so101_joint_mapping.py` is flagged `UNVERIFIED` against real data. The camera calibration scripts' live-hardware capture/motion paths are untested against real camera/robot hardware (validated via synthetic data only). PhysX GPU-accelerated physics does not work on the Runpod container used for Isaac Lab work (structural host GPU-isolation limitation) - CPU PhysX works, just slower at scale.
- `solo/mcp/` - Domain-specific Model Context Protocol implementations (agriculture, education, healthcare, etc.)

**Multi-platform support:** Hardware detection and server selection adapt to CUDA (NVIDIA), HIP (AMD), Metal (Apple Silicon), and CPU-only environments.

**Configuration:** Two-tier system - YAML defaults in `solo/config/config.yaml`, user settings in `~/.solo/config.json`.
