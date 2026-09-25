"""
Recording mode for LeRobot
Handles data collection and recording of robot demonstrations
"""

import typer
import subprocess
import sys
from rich.prompt import Prompt, Confirm

from solo.commands.robots.lerobot.config import (
    validate_lerobot_config,
    is_bimanual_robot,
    is_realman_robot,
)
from solo.commands.robots.lerobot.auth import authenticate_huggingface
from solo.commands.robots.lerobot.dataset import handle_existing_dataset, normalize_repo_id
from solo.commands.robots.lerobot.cameras import setup_cameras
from solo.commands.robots.lerobot.mode_config import use_preconfigured_args
from solo.commands.robots.lerobot.ports import detect_and_retry_ports, detect_bimanual_arm_ports
from solo.commands.robots.lerobot.utils.text_cleaning import clean_ansi_codes
from solo.commands.robots.lerobot.utils.record_config import unified_record_config


def cleanup_rerun():
    """Kill any running rerun.exe processes to release camera and other resources."""
    if sys.platform == "win32":
        try:
            # Silently kill rerun processes on Windows
            subprocess.run(
                ["taskkill", "/IM", "rerun.exe", "/F"],
                capture_output=True,
                check=False
            )
        except Exception:
            pass  # Ignore errors if rerun wasn't running
    else:
        try:
            # Kill rerun processes on Unix-like systems
            subprocess.run(
                ["pkill", "-f", "rerun"],
                capture_output=True,
                check=False
            )
        except Exception:
            pass


def recording_mode(config: dict, auto_use: bool = False):
    """Handle LeRobot recording mode"""
    # Check for preconfigured recording settings
    preconfigured, detected_robot_type = use_preconfigured_args(config, 'recording', 'Recording', auto_use=auto_use)
    
    # Initialize variables
    leader_id = None
    follower_id = None
    
    if preconfigured:
        # Use preconfigured settings
        robot_type = preconfigured.get('robot_type')
        leader_port = preconfigured.get('leader_port')
        follower_port = preconfigured.get('follower_port')
        camera_config = preconfigured.get('camera_config')
        leader_id = preconfigured.get('leader_id')
        follower_id = preconfigured.get('follower_id')

        from solo.commands.robots.lerobot.starai_config import ensure_starai_leader_port
        leader_port = ensure_starai_leader_port(config, 'recording', robot_type, leader_port)
        if not leader_port:
            return

        dataset_repo_id = preconfigured.get('dataset_repo_id')
        # Clean ANSI escape codes to prevent file system errors
        if dataset_repo_id:
            dataset_repo_id = clean_ansi_codes(dataset_repo_id)
            
            # Additional validation: Ensure dataset_repo_id doesn't start with '/' or contain problematic characters
            if dataset_repo_id.startswith('/'):
                typer.echo(f"⚠️  Warning: dataset_repo_id starts with '/', removing it")
                dataset_repo_id = dataset_repo_id.lstrip('/')
            
            # Ensure dataset_repo_id has proper format (owner/name or local/name)
            if '/' not in dataset_repo_id:
                dataset_repo_id = f"local/{dataset_repo_id}"
                typer.echo(f"🔧 Fixed dataset_repo_id format: '{dataset_repo_id}'")
        task_description = preconfigured.get('task_description')
        episode_time = preconfigured.get('episode_time')
        num_episodes = preconfigured.get('num_episodes')
        fps = preconfigured.get('fps')
        push_to_hub = preconfigured.get('push_to_hub')
        # When using preconfigured settings, default to resume mode
        should_resume = True
        
        # Check if the dataset actually exists and is valid before assuming we can resume
        if dataset_repo_id:
            from solo.commands.robots.lerobot.dataset import check_dataset_exists, check_dataset_directory_exists
            
            if check_dataset_exists(dataset_repo_id):
                # Valid dataset exists, can resume
                typer.echo(f"📂 Found existing dataset '{dataset_repo_id}', will resume recording")
            else:
                # Check if there's an incomplete directory
                dir_exists, dataset_path = check_dataset_directory_exists(dataset_repo_id)
                
                if dir_exists:
                    # Directory exists but dataset is incomplete
                    typer.echo(f"\n⚠️  Incomplete dataset directory found: {dataset_path}")
                    typer.echo("   Missing required metadata (info.json) - previous recording may have failed.\n")
                    
                    import shutil
                    
                    typer.echo("Options:")
                    typer.echo("  1. Delete the incomplete directory and start fresh")
                    typer.echo("  2. Choose a different dataset name")
                    
                    choice = Prompt.ask("Select option", choices=["1", "2"], default="1")
                    
                    if choice == "1":
                        confirm_delete = Confirm.ask(
                            f"Delete '{dataset_path}'?", 
                            default=False
                        )
                        if confirm_delete:
                            try:
                                shutil.rmtree(dataset_path)
                                typer.echo(f"✅ Deleted incomplete dataset directory")
                                should_resume = False
                            except Exception as e:
                                typer.echo(f"❌ Failed to delete: {e}")
                                new_name = Prompt.ask("Enter a new dataset name")
                                dataset_repo_id = f"local/{new_name}" if '/' not in new_name else new_name
                                should_resume = False
                        else:
                            new_name = Prompt.ask("Enter a new dataset name")
                            dataset_repo_id = f"local/{new_name}" if '/' not in new_name else new_name
                            should_resume = False
                    else:
                        new_name = Prompt.ask("Enter a new dataset name")
                        dataset_repo_id = f"local/{new_name}" if '/' not in new_name else new_name
                        should_resume = False
                else:
                    # No directory exists, create new dataset
                    should_resume = False
                    typer.echo(f"📂 Dataset '{dataset_repo_id}' does not exist yet, will create new dataset")
        
        # Validate that we have the required settings
        # RealMan robots don't need follower_port (use network instead)
        if is_realman_robot(robot_type):
            realman_config = preconfigured.get('realman_config')
            if not (leader_port and robot_type and realman_config):
                typer.echo("❌ Preconfigured settings missing required RealMan configuration")
                typer.echo("Please run setup first or use new settings")
                preconfigured = None
        elif not (leader_port and follower_port and robot_type):
            typer.echo("❌ Preconfigured settings missing required robot configuration")
            typer.echo("Please run calibration first or use new settings")
            preconfigured = None
    
    if not preconfigured:
        # Validate configuration using utility function
        leader_port, follower_port, leader_calibrated, follower_calibrated, saved_robot_type = validate_lerobot_config(config)
        
        # Use detected robot type if available (e.g., from mismatch detection), otherwise use saved
        robot_type = detected_robot_type if detected_robot_type else saved_robot_type
        
        if not robot_type:
            from solo.commands.robots.lerobot.utils.helper import auto_detect_robot
            robot_type = auto_detect_robot(default="so101")
            config['robot_type'] = robot_type
        
        # Handle port/connection detection based on robot type
        if is_realman_robot(robot_type):
            from solo.commands.robots.lerobot.utils.helper import get_realman_configs, port_detection
            
            # Leader is SO101 (USB)
            leader_port = port_detection(config, "leader", "so101", leader_port)
            
            # Follower is RealMan (network)
            realman_config = get_realman_configs(config)
            config['realman_config'] = realman_config
            follower_port = None  # Network-based
            
            typer.echo(f"\n🔌 Connection Configuration:")
            typer.echo(f"   • Leader (SO101): {leader_port}")
            typer.echo(f"   • Follower (RealMan): {realman_config.get('ip')}:{realman_config.get('port')}")
        
        elif is_bimanual_robot(robot_type):
            # Bimanual port detection
            lerobot_config = config.get('lerobot', {})
            left_leader_port = lerobot_config.get('left_leader_port')
            right_leader_port = lerobot_config.get('right_leader_port')
            left_follower_port = lerobot_config.get('left_follower_port')
            right_follower_port = lerobot_config.get('right_follower_port')
            
            if not left_leader_port or not right_leader_port:
                left_leader_port, right_leader_port = detect_bimanual_arm_ports("leader")
                config['left_leader_port'] = left_leader_port
                config['right_leader_port'] = right_leader_port
            if not left_follower_port or not right_follower_port:
                left_follower_port, right_follower_port = detect_bimanual_arm_ports("follower")
                config['left_follower_port'] = left_follower_port
                config['right_follower_port'] = right_follower_port
        else:
            # Single-arm port detection
            from solo.commands.robots.lerobot.utils.helper import port_detection
            leader_port = port_detection(config, "leader", robot_type, leader_port)
            follower_port = port_detection(config, "follower", robot_type, follower_port)
        
        # Select ids
        from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
        leader_id = prompt_arm_id(config, "leader", robot_type)
        follower_id = prompt_arm_id(config, "follower", robot_type)

        # Step 1: HuggingFace authentication (required - recordings must be pushed to Hub
        # so remote training, e.g. on Runpod, can pull the dataset without an SSH transfer)
        typer.echo("\n📋 Step 1: HuggingFace Hub Configuration")
        typer.echo("Pushing to HuggingFace Hub is required so datasets can be trained on remotely (e.g. Runpod).")
        push_to_hub = True
        login_success, hf_username = authenticate_huggingface()

        if not login_success:
            typer.echo("❌ HuggingFace authentication failed. Recording requires a HuggingFace login - please try again.")
            raise typer.Exit(code=1)

        # Step 2: Get recording parameters
        typer.echo("\n⚙️ Step 2: Recording Configuration")

        # Get dataset name and handle existing datasets
        dataset_name = Prompt.ask("Enter dataset repository name", default="lerobot-dataset")
        username_for_format = hf_username
        initial_repo_id = normalize_repo_id(dataset_name, hf_username=username_for_format)
        # Check if dataset exists and handle appropriately
        dataset_repo_id, should_resume = handle_existing_dataset(initial_repo_id)
        # Ensure the returned id still has a namespace (user may have typed name-only)
        dataset_repo_id = normalize_repo_id(dataset_repo_id, hf_username=username_for_format)
        # Clean ANSI escape codes to prevent file system errors
        dataset_repo_id = clean_ansi_codes(dataset_repo_id)
        
        # Additional validation: Ensure dataset_repo_id doesn't start with '/' or contain problematic characters
        if dataset_repo_id.startswith('/'):
            typer.echo(f"⚠️  Warning: dataset_repo_id starts with '/', removing it")
            dataset_repo_id = dataset_repo_id.lstrip('/')
        
        # Ensure dataset_repo_id has proper format (owner/name or local/name)
        if '/' not in dataset_repo_id:
            if hf_username:
                # Use HuggingFace username format for hub uploads
                dataset_repo_id = f"{hf_username}/{dataset_repo_id}"
                typer.echo(f"🔧 Fixed dataset_repo_id format: '{dataset_repo_id}'")
            else:
                # Use local format for local-only datasets
                dataset_repo_id = f"local/{dataset_repo_id}"
                typer.echo(f"🔧 Fixed dataset_repo_id format: '{dataset_repo_id}'")

        # Get task description
        task_description = Prompt.ask("Enter task description (e.g., 'Pick up the red cube and place it in the box')")
        
        # Get episode time
        episode_time = float(Prompt.ask("Duration of each recording episode in seconds", default="60"))
        
        # Get number of episodes
        num_episodes = int(Prompt.ask("Total number of episodes to record", default="10"))

        # Setup cameras
        camera_config = setup_cameras()

    # Save configuration before execution (if not using preconfigured settings)
    if not preconfigured:
        from solo.commands.robots.lerobot.mode_config import save_recording_config
        recording_args = {
            'robot_type': robot_type,
            'leader_port': leader_port,
            'follower_port': follower_port,
            'camera_config': camera_config,
            'leader_id': leader_id,
            'follower_id': follower_id,
            'dataset_repo_id': dataset_repo_id,
            'task_description': task_description,
            'episode_time': episode_time,
            'num_episodes': num_episodes,
            'fps': 30,
            'push_to_hub': push_to_hub,
            'should_resume': should_resume
        }
        save_recording_config(config, recording_args)

    # Step 3: Start recording
    typer.echo("\n🎬 Starting Data Recording")
    typer.echo(f"   • Dataset: {dataset_repo_id}")
    typer.echo(f"   • Task: {task_description}")
    typer.echo(f"   • Episode duration: {episode_time}s")
    typer.echo(f"   • Number of episodes: {num_episodes}")
    typer.echo(f"   • Push to hub: {push_to_hub}")
    typer.echo(f"   • Robot type: {robot_type.upper()}")
    
    # Import lerobot recording components
    from lerobot.scripts.lerobot_record import record
    
    try:
        
        # Create unified record configuration for recording mode
        record_config_kwargs = {
            'robot_type': robot_type,
            'leader_port': leader_port,
            'follower_port': follower_port,
            'camera_config': camera_config,
            'mode': "recording",
            'leader_id': leader_id,
            'follower_id': follower_id,
            'dataset_repo_id': dataset_repo_id,
            'task_description': task_description,
            'episode_time': episode_time,
            'num_episodes': num_episodes,
            'push_to_hub': push_to_hub,
            'fps': 30,
            'should_resume': should_resume,
        }
        
        # Add robot-type specific configuration
        if is_realman_robot(robot_type):
            # Add RealMan network configuration
            realman_config = config.get('realman_config') or config.get('lerobot', {}).get('realman_config')
            record_config_kwargs.update({
                'realman_config': realman_config,
            })
        elif is_bimanual_robot(robot_type):
            # Add bimanual ports
            lerobot_config = config.get('lerobot', {})
            record_config_kwargs.update({
                'left_leader_port': lerobot_config.get('left_leader_port'),
                'right_leader_port': lerobot_config.get('right_leader_port'),
                'left_follower_port': lerobot_config.get('left_follower_port'),
                'right_follower_port': lerobot_config.get('right_follower_port'),
            })
        
        record_config = unified_record_config(**record_config_kwargs)
        
        if should_resume:
            typer.echo("📝 Resuming — recording will continue from existing dataset")
        
        # Display keyboard shortcuts in a prominent panel
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        
        console = Console()
        
        # Create a table for keyboard shortcuts
        shortcuts_table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        shortcuts_table.add_column("Key", style="bold yellow", width=15)
        shortcuts_table.add_column("Action", style="white")
        
        shortcuts_table.add_row("→  Right Arrow", "Next episode (early stop current & proceed)")
        shortcuts_table.add_row("←  Left Arrow", "Re-record (cancel current episode)")
        shortcuts_table.add_row("ESC  Escape", "Stop & save (encode videos, upload dataset)")
        
        # Create panel with shortcuts
        shortcuts_panel = Panel(
            shortcuts_table,
            title="⌨️  Keyboard Shortcuts",
            title_align="left",
            border_style="bright_blue",
            padding=(1, 2),
        )
        
        console.print()
        console.print(shortcuts_panel)
        console.print()
        
        typer.echo("💡 Tip: Move the leader arm to control the follower")
        
        # Camera validation disabled - it was causing issues by opening/closing camera
        # right before lerobot needs it. If camera fails, lerobot will report the error.
        if camera_config and camera_config.get('enabled'):
            typer.echo("\n📷 Camera configured - if recording fails with camera error,")
            typer.echo("   make sure no other apps are using the camera.\n")
        
        # Start recording with retry logic
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                dataset = record(record_config)
                
                mode_text = "resumed and completed" if should_resume else "completed"
                typer.echo(f"✅ Recording {mode_text}!")
                
                # Get the actual dataset name from the record config for display
                actual_repo_id = record_config.dataset.repo_id
                typer.echo(f"📊 Dataset: {actual_repo_id}")
                typer.echo(f"📈 Total episodes in dataset: {dataset.num_episodes}")
                
                if push_to_hub:
                    typer.echo(f"🚀 Dataset pushed to HuggingFace Hub: https://huggingface.co/datasets/{actual_repo_id}")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                # Check if it's a port connection error
                if "Could not connect on port" in error_msg or "Make sure you are using the correct port" in error_msg:
                    if attempt < max_retries:
                        typer.echo(f"❌ Connection failed: {error_msg}")
                        typer.echo("🔄 Attempting to detect new ports...")
                        
                        # Detect new ports and retry
                        new_leader_port, new_follower_port = detect_and_retry_ports(leader_port, follower_port, config)
                        
                        if new_leader_port != leader_port or new_follower_port != follower_port:
                            # Update ports and recreate config
                            leader_port, follower_port = new_leader_port, new_follower_port
                            
                            # Build config kwargs
                            retry_config_kwargs = {
                                'robot_type': robot_type,
                                'leader_port': leader_port,
                                'follower_port': follower_port,
                                'camera_config': camera_config,
                                'mode': "recording",
                                'leader_id': leader_id,
                                'follower_id': follower_id,
                                'dataset_repo_id': dataset_repo_id,
                                'task_description': task_description,
                                'episode_time': episode_time,
                                'num_episodes': num_episodes,
                                'push_to_hub': push_to_hub,
                                'fps': 30,
                                'should_resume': should_resume,
                            }
                            
                            # Add robot-type specific configuration
                            if is_realman_robot(robot_type):
                                # Add RealMan network configuration
                                realman_config = config.get('realman_config') or config.get('lerobot', {}).get('realman_config')
                                retry_config_kwargs.update({
                                    'realman_config': realman_config,
                                })
                            elif is_bimanual_robot(robot_type):
                                # Add bimanual ports
                                lerobot_config = config.get('lerobot', {})
                                retry_config_kwargs.update({
                                    'left_leader_port': lerobot_config.get('left_leader_port'),
                                    'right_leader_port': lerobot_config.get('right_leader_port'),
                                    'left_follower_port': lerobot_config.get('left_follower_port'),
                                    'right_follower_port': lerobot_config.get('right_follower_port'),
                                })
                            
                            record_config = unified_record_config(**retry_config_kwargs)
                            typer.echo("🔄 Retrying recording with new ports...")
                            continue
                        else:
                            typer.echo("❌ Could not find new ports. Please check connections.")
                            cleanup_rerun()
                            return
                    else:
                        typer.echo(f"❌ Recording failed after retry: {error_msg}")
                        cleanup_rerun()
                        return
                elif "Cannot create a file when that file already exists" in error_msg:
                    typer.echo(f"❌ Dataset already exists: {dataset_repo_id}")
                    typer.echo("Please try running the command again.")
                    cleanup_rerun()
                    return
                else:
                    # Non-port related error
                    typer.echo(f"❌ Recording failed: {error_msg}")
                    typer.echo("Please check your robot connections and try again.")
                    cleanup_rerun()
                    return
        
        # Cleanup after successful recording too
        cleanup_rerun()
        
    except KeyboardInterrupt:
        typer.echo("\n🛑 Recording stopped by user.")
        cleanup_rerun()

