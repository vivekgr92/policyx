"""
Replay mode for LeRobot
Handles replaying recorded dataset episodes on the robot
"""

import random
import time
import typer
from rich.prompt import Prompt, Confirm

from solo.commands.robots.lerobot.config import (
    validate_lerobot_config,
    get_robot_config_classes,
    save_lerobot_config,
    is_bimanual_robot,
    is_realman_robot,
    create_follower_config,
    create_bimanual_follower_config,
)
from solo.commands.robots.lerobot.mode_config import use_preconfigured_args, load_mode_config
from solo.commands.robots.lerobot.ports import detect_arm_port, detect_bimanual_arm_ports
from solo.commands.robots.lerobot.utils.text_cleaning import clean_ansi_codes


def _parse_episode_selection(value, total_episodes: int) -> list:
    """Parse an episode selection into a sorted list of unique episode indices.

    Accepts an int (single episode, for backward compatibility with old saved
    configs), or a string: a single number ("3"), a comma-separated list
    ("0,2,5"), an inclusive range ("0-10"), a combination ("0-2,5,7-9"), or
    "all"/"*" for every episode in the dataset.
    """
    if isinstance(value, list):
        episodes = [int(v) for v in value]
    elif isinstance(value, int):
        episodes = [value]
    else:
        raw = str(value).strip().lower()
        if raw in ("all", "*"):
            episodes = list(range(total_episodes))
        else:
            episodes = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start_s, end_s = part.split("-", 1)
                    episodes.extend(range(int(start_s), int(end_s) + 1))
                else:
                    episodes.append(int(part))

    episodes = sorted(set(episodes))
    invalid = [e for e in episodes if e < 0 or e >= total_episodes]
    if invalid:
        raise ValueError(
            f"Episode(s) {invalid} out of range. Dataset has {total_episodes} episode(s) "
            f"(valid range: 0 to {total_episodes - 1})."
        )
    if not episodes:
        raise ValueError("No episodes selected.")
    return episodes


def _dataset_already_exists(repo_id: str) -> bool:
    """Check whether repo_id already has a (locally cached) dataset on disk."""
    from solo.commands.robots.lerobot.dataset import check_dataset_exists
    return check_dataset_exists(repo_id)


def _perturb_action(action: dict, obs: dict, limits, perturb: float, dt: float) -> dict:
    """Add random per-step noise to a replayed action, then safety-validate it.

    Noise is drawn fresh per call as `perturb` fraction of each joint's
    calibrated safe range (not a raw tick count, since joints have very
    different ranges), then clamped via the same `validate_action()` DeployX's
    edge agent uses before ever sending it to real hardware - unlike an
    unperturbed replayed action (already safely executed once during real
    teleop), a perturbed action is new and never-executed, so it gets the same
    scrutiny a live policy's output would.
    """
    from solo.commands.robots.lerobot.deployx.safety import validate_action

    ordered = [action[f"{name}.pos"] for name in limits.names]
    noisy = [
        v + random.uniform(-1.0, 1.0) * perturb * (limits.position_max[i] - limits.position_min[i])
        for i, v in enumerate(ordered)
    ]

    positions = {key[: -len(".pos")]: float(value) for key, value in obs.items() if key.endswith(".pos")}
    current_state = (
        [positions[name] for name in limits.names]
        if all(name in positions for name in limits.names)
        else None
    )

    is_safe, clamped, reason = validate_action(noisy, current_state, dt, limits)
    if not is_safe:
        raise RuntimeError(f"Perturbed action rejected by safety check: {reason}")

    return {f"{name}.pos": clamped[i] for i, name in enumerate(limits.names)}


def replay_mode(config: dict, auto_use: bool = False, replay_options: dict = None):
    """Handle LeRobot replay mode - replay actions from recorded dataset episode(s)"""
    # Check if CLI arguments were provided (non-interactive mode)
    if replay_options and replay_options.get('dataset'):
        # Use CLI arguments
        _, follower_port, _, _, robot_type = validate_lerobot_config(config)
        follower_id = replay_options.get('follower_id')
        dataset_repo_id = replay_options.get('dataset')
        episode_raw = replay_options.get('episode', 0)
        fps = replay_options.get('fps', 30)
        play_sounds = True
        save_replay_as = replay_options.get('save_replay_as')
        # Non-interactive: auto-resume into an already-existing dataset rather
        # than prompting, since this path is meant to run unattended.
        save_replay_resume = bool(save_replay_as) and _dataset_already_exists(save_replay_as)
        repeat_count = max(1, replay_options.get('repeat') or 1)
        perturb = replay_options.get('perturb') or 0.0
        perturb_increment = replay_options.get('perturb_increment') or 0.0
        camera_config = replay_options.get('camera_config')
        if camera_config is None:
            # No cameras passed on the CLI (there's no flag for that) - fall back
            # to whatever was cached from a previous interactive replay run.
            cached_replay = load_mode_config(config, 'replay')
            camera_config = cached_replay.get('camera_config') if cached_replay else None

        typer.echo(f"📦 Dataset: {dataset_repo_id}")
        if follower_id:
            typer.echo(f"🤖 Follower ID: {follower_id}")
    else:
        # Check for preconfigured replay settings
        preconfigured, detected_robot_type = use_preconfigured_args(config, 'replay', 'Replay', auto_use=auto_use)

        if preconfigured and preconfigured.get('follower_port') and preconfigured.get('dataset_repo_id'):
            robot_type = preconfigured.get('robot_type')
            follower_port = preconfigured.get('follower_port')
            follower_id = preconfigured.get('follower_id')
            dataset_repo_id = clean_ansi_codes(preconfigured.get('dataset_repo_id', ''))
            episode_raw = preconfigured.get('episode', 0)
            fps = preconfigured.get('fps', 30)
            play_sounds = preconfigured.get('play_sounds', True)
            save_replay_as = preconfigured.get('save_replay_as')
            save_replay_resume = bool(save_replay_as) and _dataset_already_exists(save_replay_as)
            repeat_count = max(1, preconfigured.get('repeat') or 1)
            perturb = preconfigured.get('perturb') or 0.0
            perturb_increment = preconfigured.get('perturb_increment') or 0.0
            camera_config = preconfigured.get('camera_config')
        else:
            # Get robot config
            _, follower_port, _, _, saved_robot_type = validate_lerobot_config(config)

            # Use detected robot type if available (e.g., from mismatch detection), otherwise use saved
            robot_type = detected_robot_type if detected_robot_type else saved_robot_type

            if not robot_type:
                from solo.commands.robots.lerobot.utils.helper import auto_detect_robot
                robot_type = auto_detect_robot(default="so101")

            # Handle port detection based on robot type
            if is_realman_robot(robot_type):
                from solo.commands.robots.lerobot.utils.helper import get_realman_configs
                realman_config = get_realman_configs(config)
                config['realman_config'] = realman_config
                follower_port = None  # Network-based
                typer.echo(f"\n🔌 RealMan follower: {realman_config.get('ip')}:{realman_config.get('port')}")

            elif is_bimanual_robot(robot_type):
                # Bimanual port detection
                lerobot_config = config.get('lerobot', {})
                left_follower_port = lerobot_config.get('left_follower_port')
                right_follower_port = lerobot_config.get('right_follower_port')

                if not left_follower_port or not right_follower_port:
                    left_follower_port, right_follower_port = detect_bimanual_arm_ports("follower")
                    config['left_follower_port'] = left_follower_port
                    config['right_follower_port'] = right_follower_port
            else:
                # Single-arm port detection
                from solo.commands.robots.lerobot.utils.helper import port_detection
                follower_port = port_detection(config, "follower", robot_type, follower_port)

            # Get follower ID
            from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
            follower_id = prompt_arm_id(config, "follower", robot_type)

            # Get default dataset from recording config if available
            recording_config = load_mode_config(config, 'recording')
            default_dataset = recording_config.get('dataset_repo_id') if recording_config else None

            dataset_repo_id = clean_ansi_codes(Prompt.ask("Enter dataset repository ID", default=default_dataset or ""))
            if '/' not in dataset_repo_id:
                dataset_repo_id = f"local/{dataset_repo_id}"

            episode_raw = Prompt.ask(
                "Enter episode(s) to replay (number, '0,2,5', '0-10', or 'all')", default="0"
            )
            repeat_count = max(1, int(Prompt.ask("How many times to replay each selected episode?", default="1")))
            perturb = float(Prompt.ask(
                "Perturb replayed actions by this fraction of each joint's safe range for motion "
                "diversity? (0 = replay exactly as recorded)",
                default="0",
            ))
            perturb_increment = 0.0
            if repeat_count > 1:
                perturb_increment = float(Prompt.ask(
                    "Increase the perturbation by this much on each successive repeat? "
                    "(0 = same perturbation every repeat)",
                    default="0",
                ))
            fps = 30
            play_sounds = True

            save_replay_as = None
            save_replay_resume = False
            if Confirm.ask(
                "\nAlso save this replay as new recorded episode(s) (with live camera capture) "
                "for building up more training data?",
                default=False,
            ):
                save_replay_as = clean_ansi_codes(
                    Prompt.ask("Enter dataset repository ID to save into (new or existing)")
                )
                if '/' not in save_replay_as:
                    save_replay_as = f"local/{save_replay_as}"

                from solo.commands.robots.lerobot.dataset import handle_existing_dataset
                save_replay_as, save_replay_resume = handle_existing_dataset(save_replay_as)
                if save_replay_resume:
                    typer.echo(f"📂 Will add these episodes to the existing dataset '{save_replay_as}'")
                else:
                    typer.echo(f"📼 Will create a new dataset '{save_replay_as}'")

            from solo.commands.robots.lerobot.cameras import setup_cameras
            typer.echo("\n📷 Set up cameras for this replay:")
            camera_config = setup_cameras()

            # Save config
            from solo.commands.robots.lerobot.mode_config import save_replay_config
            save_replay_config(config, {
                'robot_type': robot_type, 'follower_port': follower_port, 'follower_id': follower_id,
                'dataset_repo_id': dataset_repo_id, 'episode': episode_raw, 'fps': fps, 'play_sounds': play_sounds,
                'save_replay_as': save_replay_as, 'camera_config': camera_config,
                'save_replay_resume': save_replay_resume, 'repeat': repeat_count, 'perturb': perturb,
                'perturb_increment': perturb_increment,
            })

    # Import lerobot components
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.processor import make_default_robot_action_processor
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import ACTION, OBS_STR, HF_LEROBOT_HOME
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

    robot = None
    new_dataset = None
    rerun_active = False
    try:
        # Resolve total episode count up front (needed to validate/expand the
        # episode selection, including "all") for both local and Hub datasets.
        import json
        local_dataset_path = HF_LEROBOT_HOME / dataset_repo_id
        is_local_dataset = dataset_repo_id.startswith("local/")

        if is_local_dataset:
            if not local_dataset_path.exists():
                raise FileNotFoundError(
                    f"Local dataset not found at: {local_dataset_path}\n"
                    f"Please check the dataset name and ensure it was recorded locally."
                )
            meta_info_path = local_dataset_path / "meta" / "info.json"
            if not meta_info_path.exists():
                raise FileNotFoundError(
                    f"Dataset metadata not found at: {meta_info_path}\n"
                    f"The dataset may be incomplete or corrupted."
                )
            with open(meta_info_path, 'r') as f:
                info = json.load(f)
            total_episodes = info.get('total_episodes', 0)
            data_path = local_dataset_path / "data"
            if not data_path.exists() or not any(data_path.rglob("*.parquet")):
                raise FileNotFoundError(
                    f"Dataset data files not found at: {data_path}\n"
                    f"The dataset may be incomplete or corrupted."
                )
            typer.echo(f"📂 Loading local dataset from: {local_dataset_path}")
        else:
            total_episodes = LeRobotDatasetMetadata(dataset_repo_id).total_episodes

        typer.echo(f"📊 Dataset has {total_episodes} episode(s)")

        try:
            episodes = _parse_episode_selection(episode_raw, total_episodes)
        except ValueError as e:
            typer.echo(f"❌ {e}")
            return

        typer.echo(f"📹 Episodes to replay: {episodes}")

        # Setup robot
        _, follower_config_class = get_robot_config_classes(robot_type)
        if not follower_config_class:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        # Cameras are active for every replay (live view / monitoring), not just
        # when also recording a new dataset from this replay. Reuses a cached
        # camera_config from a previous run if one was resolved above; only
        # prompts here as a fallback (e.g. first-ever CLI-args-driven run).
        if camera_config is None:
            from solo.commands.robots.lerobot.cameras import setup_cameras
            typer.echo("\n📷 Set up cameras for this replay:")
            camera_config = setup_cameras()
            from solo.commands.robots.lerobot.mode_config import save_replay_config
            save_replay_config(config, {
                'robot_type': robot_type, 'follower_port': follower_port, 'follower_id': follower_id,
                'dataset_repo_id': dataset_repo_id, 'episode': episode_raw, 'fps': fps, 'play_sounds': play_sounds,
                'save_replay_as': save_replay_as, 'camera_config': camera_config,
                'save_replay_resume': save_replay_resume, 'repeat': repeat_count, 'perturb': perturb,
                'perturb_increment': perturb_increment,
            })

        task_description = None
        if save_replay_as:
            task_description = clean_ansi_codes(Prompt.ask("Enter task description", default=""))

        def _build_follower_config(port):
            if is_realman_robot(robot_type):
                from solo.commands.robots.lerobot.realman_config import create_realman_follower_config
                return create_realman_follower_config(
                    config.get('realman_config'), camera_config=camera_config, follower_id=follower_id
                )
            elif is_bimanual_robot(robot_type):
                lerobot_config = config.get('lerobot', {})
                return create_bimanual_follower_config(
                    follower_config_class,
                    lerobot_config.get('left_follower_port'),
                    lerobot_config.get('right_follower_port'),
                    robot_type,
                    camera_config=camera_config,
                    follower_id=follower_id,
                )
            else:
                return create_follower_config(
                    follower_config_class, port, robot_type, camera_config=camera_config, follower_id=follower_id
                )

        follower_config = _build_follower_config(follower_port)

        # Load dataset (default behavior uses HF_LEROBOT_HOME / repo_id as root)
        try:
            dataset = LeRobotDataset(dataset_repo_id, episodes=episodes)
        except Exception as e:
            error_msg = str(e)
            if is_local_dataset and ("404 Client Error" in error_msg or "Repository Not Found" in error_msg):
                raise RuntimeError(
                    f"Failed to load local dataset '{dataset_repo_id}'.\n"
                    f"Dataset path: {local_dataset_path}\n"
                    f"This may be due to version compatibility issues or corrupted metadata.\n"
                    f"Try re-recording the dataset or check the dataset files."
                ) from e
            raise

        if save_replay_as:
            # Async image writing (threads > 0) so JPEG/video encoding doesn't
            # block the replay loop's per-step timing - same rationale as
            # lerobot's own recorder (DatasetRecordConfig defaults to 4/camera).
            num_cameras = len(camera_config) if camera_config else 0
            if save_replay_resume:
                # Append to an already-existing dataset, same as how lerobot's
                # own recorder resumes: just open it and start the image writer.
                new_dataset = LeRobotDataset(save_replay_as)
                if num_cameras > 0:
                    new_dataset.start_image_writer(num_processes=0, num_threads=4 * num_cameras)
                typer.echo(f"📂 Adding new episodes to existing dataset: {save_replay_as}")
            else:
                new_dataset = LeRobotDataset.create(
                    repo_id=save_replay_as,
                    fps=fps,
                    features=dataset.features,
                    robot_type=robot_type,
                    use_videos=True,
                    image_writer_threads=4 * num_cameras,
                )
                typer.echo(f"📼 New episodes will be saved to a new dataset: {save_replay_as}")

        robot_action_processor = make_default_robot_action_processor()

        limits = None
        if perturb > 0 or perturb_increment != 0:
            from solo.commands.robots.lerobot.deployx.safety import load_joint_limits
            limits = load_joint_limits(robot_type, follower_id)
            if perturb_increment != 0:
                last_perturb = perturb + (repeat_count - 1) * perturb_increment
                typer.echo(
                    f"⚠️  Perturbation enabled: starts at ±{perturb * 100:.1f}% of each joint's safe range, "
                    f"+{perturb_increment * 100:.1f}pp per repeat (up to ±{last_perturb * 100:.1f}% on the last repeat)"
                )
            else:
                typer.echo(f"⚠️  Perturbation enabled: ±{perturb * 100:.1f}% of each joint's safe range per step")

        init_rerun(session_name="replay")
        rerun_active = True

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                robot = make_robot_from_config(follower_config)
                robot.connect()

                for ep in episodes:
                    episode_frames = dataset.hf_dataset.filter(lambda x, ep=ep: x["episode_index"] == ep)
                    actions = episode_frames.select_columns(ACTION)

                    for rep in range(repeat_count):
                        # Ramp perturbation up (or down) per repeat, e.g. perturb=0.02,
                        # perturb_increment=0.01 -> repeat 1 uses 0.02, repeat 2 uses 0.03.
                        effective_perturb = max(0.0, perturb + rep * perturb_increment)

                        label = f"episode {ep}"
                        if repeat_count > 1:
                            label += f" (repeat {rep + 1}/{repeat_count})"
                            if perturb_increment != 0:
                                label += f", perturb={effective_perturb * 100:.1f}%"
                        typer.echo(f"\n📊 Replaying {label} ({len(episode_frames)} frames)")
                        log_say("Replaying episode", play_sounds, blocking=True)

                        for idx in range(len(episode_frames)):
                            start_t = time.perf_counter()

                            action = {name: actions[idx][ACTION][i] for i, name in enumerate(dataset.features[ACTION]["names"])}
                            obs = robot.get_observation()

                            if limits is not None and effective_perturb > 0:
                                action = _perturb_action(action, obs, limits, effective_perturb, dt=1.0 / fps)

                            processed_action = robot_action_processor((action, obs))
                            robot.send_action(processed_action)

                            log_rerun_data(observation=obs, action=processed_action)

                            if new_dataset is not None:
                                from lerobot.datasets.utils import build_dataset_frame
                                observation_frame = build_dataset_frame(new_dataset.features, obs, prefix=OBS_STR)
                                action_frame = build_dataset_frame(new_dataset.features, processed_action, prefix=ACTION)
                                new_dataset.add_frame({**observation_frame, **action_frame, "task": task_description})

                            precise_sleep(1 / fps - (time.perf_counter() - start_t))

                        if new_dataset is not None:
                            new_dataset.save_episode()
                            typer.echo(f"💾 Saved replayed {label} as a new episode in '{save_replay_as}'")

                robot.disconnect()
                typer.echo(f"\n✅ Replay completed! ({len(episodes)} episode(s) x {repeat_count} repeat(s))")
                break  # Success, exit retry loop

            except Exception as e:
                error_msg = str(e)
                # Check if it's a port connection error
                if "Could not connect on port" in error_msg or "Make sure you are using the correct port" in error_msg:
                    if attempt < max_retries:
                        typer.echo(f"❌ Connection failed: {error_msg}")
                        typer.echo("🔄 Attempting to detect new port...")

                        # Detect new follower port(s)
                        if is_bimanual_robot(robot_type):
                            left_follower_port, right_follower_port = detect_bimanual_arm_ports("follower")

                            if left_follower_port and right_follower_port:
                                typer.echo(f"✅ Found new follower ports: {left_follower_port}, {right_follower_port}")

                                # Save updated ports to main lerobot config
                                save_lerobot_config(config, {
                                    'left_follower_port': left_follower_port,
                                    'right_follower_port': right_follower_port
                                })

                                # Recreate follower config
                                follower_config = _build_follower_config(None)
                                typer.echo("🔄 Retrying replay with new ports...")
                                continue
                            else:
                                typer.echo("❌ Could not find new ports. Please check connections.")
                                return
                        else:
                            new_follower_port, _ = detect_arm_port("follower", robot_type=robot_type)

                            if new_follower_port and new_follower_port != follower_port:
                                follower_port = new_follower_port
                                typer.echo(f"✅ Found new follower port: {follower_port}")

                                # Save updated port to main lerobot config (shared across all modes)
                                save_lerobot_config(config, {'follower_port': follower_port})

                                # Save updated port to replay config
                                from solo.commands.robots.lerobot.mode_config import save_replay_config
                                save_replay_config(config, {
                                    'robot_type': robot_type, 'follower_port': follower_port, 'follower_id': follower_id,
                                    'dataset_repo_id': dataset_repo_id, 'episode': episode_raw, 'fps': fps,
                                    'play_sounds': play_sounds, 'save_replay_as': save_replay_as,
                                    'save_replay_resume': save_replay_resume, 'camera_config': camera_config,
                                    'repeat': repeat_count, 'perturb': perturb,
                                    'perturb_increment': perturb_increment,
                                })

                                follower_config = _build_follower_config(follower_port)
                                typer.echo("🔄 Retrying replay with new port...")
                                continue
                            else:
                                typer.echo("❌ Could not find new port. Please check connections.")
                                return
                    else:
                        typer.echo(f"❌ Replay failed after retry: {error_msg}")
                        return
                else:
                    raise  # Re-raise non-port errors

        if new_dataset is not None:
            if not save_replay_as.startswith("local/"):
                from solo.commands.robots.lerobot.auth import authenticate_huggingface
                typer.echo("\n☁️  Pushing newly recorded dataset to HuggingFace Hub...")
                login_success, _ = authenticate_huggingface()
                if login_success:
                    new_dataset.push_to_hub()
                    typer.echo(f"✅ Pushed to https://huggingface.co/datasets/{save_replay_as}")
                else:
                    typer.echo("⚠️  HuggingFace login failed - dataset saved locally only.")

    except KeyboardInterrupt:
        typer.echo("\n🛑 Stopped by user.")
        if new_dataset is not None:
            try:
                new_dataset.save_episode()
            except Exception:
                pass
    except Exception as e:
        typer.echo(f"❌ Replay failed: {e}")
    finally:
        if robot:
            try:
                robot.disconnect()
            except Exception:
                pass
        if rerun_active:
            try:
                shutdown_rerun()
            except Exception:
                pass
