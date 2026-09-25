## Interactive CLI for Robots:



Use `solo robo` command to run a complete robotics workflow with LeRobot: motor setup, calibration, teleoperation, data recording, training, inference, and replay.

**To install solo-cli, follow our [installation guide](https://github.com/GetSoloTech/solo-cli)**

### Quick start

```bash
# 1) Calibrate (both arms)
solo robo --calibrate all  # leader, follower, all

# 2) Teleoperate
solo robo --teleop

# 3) Record dataset
solo robo --record

# 4) Replay a recorded episode
solo robo --replay

# 5) Train a VLA policy
solo robo --train

# 6) Run inference on VLA policy
solo robo --inference

# 7) Setup motor IDs (both arms), Only needed if you encounter "missing motor IDs" errors
solo robo --motors all   # leader, follower, all

# 8) Scan for connected motors (troubleshooting)
solo robo --scan

# 9) Run connection diagnostics (troubleshooting)
solo robo --diagnose
```

### Auto-use saved settings

Use `--yes` or `-y` to automatically use previously saved settings without prompts:

```bash
solo robo --teleop -y
solo robo --record --yes
solo robo --train -y
solo robo --inference --yes
solo robo --replay -y
```

### Supported Robot Types

Solo CLI supports multiple robot configurations:

- **SO100/SO101**: Single-arm setups using Feetech STS3215 motors
- **Koch**: Single-arm setups using Dynamixel motors
- **Bimanual**: Dual-arm configurations (left + right)
- **RealMan**: Network-connected industrial robots (SO101 leader + RealMan follower)
- **Star Arm 102**: StarAI / Fashionstar leader driving an SO101 follower (see below)

The robot type is auto-detected when you connect your arms, or you can manually select it during setup.

### Star Arm 102 (StarAI / Fashionstar) leader

The Star Arm 102 `LD` leader talks over the FashionStar UART bus rather than the
Feetech or Dynamixel protocols, and has 6 joints plus a gripper where the SO101
has 5 plus a gripper. Solo pairs it with a standard SO101 follower and remaps the
leader's joints onto SO101 joint names on the way through.

**One-time setup**

```bash
# The vendor LeRobot plugin for the arm (not installed by default)
pip install 'solo-cli[starai]'

# Confirm the arm is seen - 7 FashionStar servos, ids 0-6
solo robo --scan

# Calibrate the leader, then the SO101 follower
solo robo --calibrate leader
solo robo --calibrate follower
```

Calibration matters more here than in a like-for-like pairing: each joint angle
is measured from the midpoint of the range you record, so sweep every joint
through its full travel on both arms.

There is no `--motors` step for the Star Arm 102 - its servos ship with fixed bus
ids.

**Joint mapping**

| SO101 joint     | Star Arm 102 servo |
|-----------------|--------------------|
| `shoulder_pan`  | `Motor_0`          |
| `shoulder_lift` | `Motor_1`          |
| `elbow_flex`    | `Motor_2`          |
| `wrist_flex`    | `Motor_4`          |
| `wrist_roll`    | `Motor_5`          |
| `gripper`       | `gripper`          |

`Motor_3` is the leader's forearm roll. The SO101 has no forearm roll - only a
wrist roll - so by default that joint drives nothing. Since the leader has 6 body
joints and the follower 5, one axis has to go somewhere; see **Blending** below
for the alternative.

**Tuning**

Expect at least one axis to run backwards on first use - the two arms are
different hardware. Fix it without editing anything by hand:

```bash
solo robo --star-tune
```

| Option | What it does |
|--------|--------------|
| 1 | **Identify leader joints** - move one joint at a time; the row that lights up names that servo and says what it drives (or that it is the spare) |
| 2 | **Live mapping view** - each leader reading beside the follower command it produces, and whether it is hitting a limit |
| 3-5 | Flip a joint's direction, set its gain, set its offset |
| 6 | Reassign which leader servo drives a joint |
| 7 | Blend an extra leader servo into a joint |
| 8-9 | Per-step motion cap, smoothing |
| a | Auto-fit gains to the follower's travel |

Everything is saved to `~/.solo/starai_map.json` and picked up by teleop, record
and inference alike.

**Blending the spare joint**

Option 7 folds an extra leader servo into a follower joint, so the Star Arm's two
twist axes can both drive the SO101's single wrist roll:

```
wrist_roll ← Motor_5 +1.00×Motor_3
```

Weights are leader degrees per follower degree. Use `-1.0` when the two axes turn
opposite ways - with `+1.0` on opposed axes they cancel and the joint goes dead.
Set a weight of `0` to remove a blend. The gripper cannot be blended; it is a
single 0-100 axis.

**Matching the travel**

The arms are not the same size, so a 1:1 degree transfer leaves the follower
pinned at a limit wherever the leader swings further, and short of its reach
wherever it swings less. Option `a` compares the two calibrations and scales each
gain so the leader's recorded sweep covers the follower's travel.

If it suggests a gain above ~2, that joint was probably not swept fully during
leader calibration - recalibrate rather than accept the gain, because a large
gain magnifies hand tremor as well as motion.

**Safety**

The follower runs with `max_relative_target` set (12° per control step by
default), so a wrong sign shows up as a slow drift you can stop rather than a
lunge. Commands are also clamped to the follower's own calibrated travel, which
needs the follower id to have been calibrated under the name you select. Raise,
lower or disable the cap from `solo robo --star-tune` once the mapping is right.

---

## 1) Calibrate

Calibrate the leader and/or follower arm(s) after motor IDs are set.

```bash
# Calibrate both
solo robo --calibrate all

# Calibrate only leader
solo robo --calibrate leader

# Calibrate only follower
solo robo --calibrate follower
```

### Interactive flow
- Optionally reuse saved `robot_type` and ports or re-detect them.
- Choose or confirm `leader_id` and `follower_id` (stored for later reuse).
- For each arm selected, LeRobot's calibration is launched and you follow on-screen instructions.

### Tips/Notes
- Calibrate in a safe, clear workspace; follow the on-screen instructions carefully.
- If either arm fails to calibrate, you can rerun calibration for that arm only.
- Once both arms are calibrated, you're ready for Teleop and Recording.

---

## 2) Teleop

Teleoperate the follower arm using the leader arm. Requires both arms calibrated.

```bash
solo robo --teleop
```

### Interactive flow
- If saved Teleop settings exist, choose to reuse or enter new ones.
- Confirm or detect `leader_port` and `follower_port`, confirm `robot_type`.
- Optionally set up cameras (OpenCV/RealSense), assign viewing angles, and select which cameras to display.
- Optionally choose/confirm `leader_id` and `follower_id`.
- Teleoperation starts; move the leader arm to control the follower.

### Tips/Notes
- If a connection error occurs, the tool can auto-detect new ports and retry once.
- Camera setup is optional; if no cameras are found or selected, Teleop will still work.
- Press Ctrl+C to stop Teleop at any time.
- For bimanual setups, move BOTH leader arms to control BOTH follower arms.
- For RealMan setups, ensure the RealMan robot is powered and network-connected.

---

## 3) Record

Record datasets for training while teleoperating the follower. Requires both arms calibrated.

```bash
solo robo --record
```

### Interactive flow
- Reuse saved Recording settings or enter new ones.
- Confirm `robot_type`, `leader_port`, `follower_port`, and IDs, then set up cameras.
- Choose whether to push to HuggingFace Hub; if yes, you will be prompted to authenticate.
- Enter dataset repository name; if it already exists, choose whether to resume or rename.
- Enter task description, episode duration, and number of episodes.
- Recording starts and follows LeRobot's standard UI/controls.

### Tips/Notes
- Use `local/<name>` or a bare `<name>` to keep datasets local; pushing to Hub is optional.
- Controls during recording:
  - Right Arrow (→): early stop/reset and move to next
  - Left Arrow (←): cancel current episode and re-record
  - Escape (ESC): stop session, encode videos, and upload (if enabled)
- If ports change mid-session, the tool can detect new ports and retry once.
- Incomplete datasets can be detected and you'll be prompted to delete or rename them.

---

## 4) Replay

Replay actions from a previously recorded dataset episode on the follower arm. Useful for verifying recordings or demonstrating learned behaviors.

```bash
# Interactive mode
solo robo --replay

# Non-interactive mode with CLI arguments
solo robo --replay --dataset organize_fennel_seed --episode 0
solo robo --replay --dataset local/my_dataset --episode 2 --follower-id follower_right --fps 30
```

### CLI Options for Replay
- `--dataset`: Dataset repository ID (e.g., `organize_fennel_seed` or `local/my_dataset`)
- `--episode`: Episode number to replay (default: 0)
- `--follower-id`: Follower arm ID (e.g., `follower_right`)
- `--fps`: Frames per second for replay (default: 30)

### Interactive flow
- Reuse saved Replay settings or enter new ones.
- Confirm `robot_type` and `follower_port`, and select `follower_id`.
- Enter `dataset_repo_id` (defaults to the last recorded dataset if available).
- Enter the episode number to replay.
- Replay starts and the follower arm executes the recorded actions.

### Tips/Notes
- Only the follower arm is required (no leader arm needed).
- The dataset can be local (`local/<name>`) or from HuggingFace Hub (`username/dataset`).
- If the port changes, the tool can detect the new port and retry once.
- Press Ctrl+C to stop replay at any time.

---

## 5) Train

Train a policy on recorded data.

```bash
solo robo --train
```

### Interactive flow
- Reuse saved Training settings or enter new ones.
- Provide `dataset_repo_id` (local or Hub dataset ID).
- Select policy type:
  1. **SmolVLA** - Vision-Language-Action model (default pretrained: `lerobot/smolvla_base`)
  2. **ACT** - Action Chunking with Transformers
  3. **PI0** - Policy Iteration Zero
  4. **TDMPC** - Temporal Difference MPC
  5. **Diffusion Policy** - Good for most tasks
- Set training steps and batch size.
- Choose output directory; if it exists, pick: resume, overwrite, or new directory.
- Choose whether to push the trained model to HuggingFace Hub (prompts for auth and repo name).
- Optionally enable Weights & Biases logging (prompts to login and set project).
- Training starts using LeRobot's training pipeline.

### Tips/Notes
- Some policies can start from a pretrained checkpoint; SmolVLA defaults to `lerobot/smolvla_base` if not provided.
- Video backend auto-falls back to PyAV if TorchCodec is unavailable.
- Checkpoints are saved every 1000 steps; you can resume later if needed.
- Camera names are auto-mapped if they don't match policy defaults.
- Press Ctrl+C to stop early; partial checkpoints may be saved.

---

## 6) Inference

Run a trained policy on the follower arm. Requires the follower to be calibrated. Teleoperation override is optional if the leader is calibrated and available.

```bash
solo robo --inference
```

### Interactive flow
- Reuse saved Inference settings or enter new ones.
- Validates calibrated follower; optionally enables Teleop override if leader is calibrated.
- Prompts to authenticate with HuggingFace (required for remote models, skipped for local models).
- Enter `policy_path`:
  - HuggingFace model ID (e.g., `lerobot/act_so100_test`)
  - Local path (e.g., `./outputs/train/checkpoint` or `/path/to/model`)
  - Auto-detects latest local trained model if available
- Enter task description and inference duration.
- Set up cameras (optional) and start inference.

### Tips/Notes
- If Teleop override is enabled, moving the leader arm temporarily overrides the policy.
- Same keyboard controls apply as in recording (→/←/ESC) for session control.
- If ports change, the tool can detect new ports and retry once.
- Local model paths are validated before inference starts.

---

## 7) Motors

Most arms provided by Solo Tech already have motor IDs set up, so you can usually skip this step and proceed directly to calibration. Only run this command if you encounter a "missing motor IDs" error or if you need to reassign motor IDs for the leader and/or follower arm.

```bash
# Setup both arms 
solo robo --motors all

# Setup only leader
solo robo --motors leader

# Setup only follower
solo robo --motors follower
```

### Interactive flow
- Select or reuse the **robot type** (`SO100`, `SO101`, `Koch`, or bimanual variants). Star Arm 102 leaders are skipped here - their servo ids are fixed at the factory.
- Auto-detect the **leader** and/or **follower** port(s); unplug/replug guidance is provided if needed.
- For each selected arm, the tool launches LeRobot's device and runs `setup_motors()`.

### Tips/Notes
- You will be prompted to connect the arm when the tool searches for its port, so it's best to connect and power on the arm when instructed during the process.
- If auto-detection finds multiple ports, you can select the correct one from a list.
- After successful setup, you'll see a success message and a hint to run calibration next.
- For RealMan robots, the follower uses network connection instead of USB.

---

## 8) Scan

Scan all serial ports for connected motors. Useful for troubleshooting connection issues.

```bash
solo robo --scan
```

### What it does
- Lists all available serial ports on your system.
- Scans each port for Dynamixel motors (Koch arms) and Feetech motors (SO100/SO101).
- Reports motor IDs, model numbers, and model names found on each port.
- For SO100/SO101, reads motor voltage to determine if it's a leader (5V) or follower (12V) arm.
- Auto-detects robot type based on motors found.

### Tips/Notes
- Make sure your robot arm is powered before scanning.
- SDK status is reported (dynamixel_sdk and scservo_sdk availability).
- If no motors are found, check power supply, USB connections, and cable daisy chain.

---

## 9) Diagnose

Run detailed connection diagnostics on all serial ports. Provides more in-depth troubleshooting than scan.

```bash
solo robo --diagnose
```

### What it does
- Opens each serial port and sets baud rate.
- Reports a Star Arm 102 leader (FashionStar bus) and stops there, since the checks below do not apply to it.
- Pings motors 1-6 and reports which respond.
- Reads Min_Position_Limit register to verify read operations work.
- Reads Present_Position register for each motor.
- Reports any errors encountered during diagnostics.

### Tips/Notes
- Useful when teleoperation or calibration fails with "sync read" errors.
- Shows step-by-step diagnostics for each port.
- Helps identify loose cables, power issues, or port conflicts.

---

## Cameras and Ports

- **Ports**: Auto-detection works on Windows, macOS, and Linux. If `pyserial` is missing, it is installed automatically. If the arm was already connected, you'll be guided to unplug/replug to identify the correct port.
- **Arm Type Detection**: For SO100/SO101 arms, leader vs follower is detected by motor voltage (5V = leader, 12V = follower). For Koch arms, it's detected by motor model numbers. A Star Arm 102 is always the leader, and its presence selects the `stararm102` robot type even when an SO101 leader is plugged in alongside it.
- **Cameras**: Both OpenCV and RealSense cameras are supported. RealSense requires `pyrealsense2`. You can map each camera to a viewing angle and pick which ones to display during Teleop/Record/Inference.

## Troubleshooting

- **Connection failed / wrong port**: The tool will attempt to re-detect ports and retry once. If it still fails, unplug/replug each arm and re-run the mode.
- **Motor communication / sync read errors**: Run `solo robo --scan` or `solo robo --diagnose` to verify motor connections. Check power supply (12V for follower, 5V for leader on SO arms).
- **Dataset already exists**: You will be prompted to resume or rename. Resuming continues appending episodes to the existing dataset.
- **Incomplete dataset directory**: If a previous recording failed, you'll be prompted to delete the incomplete directory or choose a new name.
- **HuggingFace authentication required**: Needed for downloading trained policies (Inference) and optionally for pushing datasets/models (Record/Train). Local models skip authentication.
- **Windows symlink warnings**: Symlink warnings are suppressed for HF Hub downloads by the tool when needed.
- **No cameras detected**: You can proceed without cameras; functionality remains available.
- **RealMan connection issues**: Ensure the robot is powered, network-connected, and the IP/port settings in `realman_config.yaml` are correct.
- **Star Arm 102 not found**: It needs 12V into the UC-01 board, and its board enumerates as a CH340 (`/dev/tty.usbserial*` on macOS, `/dev/ttyUSB*` on Linux) rather than the `usbmodem`/`ttyACM` device an SO arm presents. `solo robo --scan` reports it separately from Feetech and Dynamixel buses.
- **Star Arm 102 joint moves the wrong way or over-travels**: Run `solo robo --star-tune` and flip that joint's direction or lower its gain. If it was never calibrated, teleop refuses to start - run `solo robo --calibrate leader` first.
