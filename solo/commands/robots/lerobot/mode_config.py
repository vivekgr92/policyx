"""
Mode-specific configuration utilities for LeRobot
Handles loading and saving of mode-specific configurations
"""

import json
import os
import typer
from rich.prompt import Confirm
from typing import Dict, Optional, Any
from solo.config import CONFIG_PATH


def load_mode_config(config: dict, mode: str) -> Optional[Dict]:
    """
    Load mode-specific configuration from the main config file.
    
    Args:
        config: Main configuration dictionary
        mode: Mode name (e.g., 'calibration', 'teleop', 'recording', 'training', 'inference')
    
    Returns:
        Mode-specific configuration dictionary or None if not found
    """
    lerobot_config = config.get('lerobot', {})
    mode_configs = lerobot_config.get('mode_configs', {})
    return mode_configs.get(mode)


def save_mode_config(config: dict, mode: str, mode_config: Dict) -> None:
    """
    Save mode-specific configuration to the main config file.
    
    Args:
        config: Main configuration dictionary
        mode: Mode name (e.g., 'calibration', 'teleop', 'recording', 'training', 'inference')
        mode_config: Mode-specific configuration to save
    """
    if 'lerobot' not in config:
        config['lerobot'] = {}
    
    if 'mode_configs' not in config['lerobot']:
        config['lerobot']['mode_configs'] = {}
    
    config['lerobot']['mode_configs'][mode] = mode_config
    
    # Save to file
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    


def use_preconfigured_args(config: dict, mode: str, mode_name: str, auto_use: bool = False) -> tuple[Optional[Dict], Optional[str]]:
    """
    Check if preconfigured arguments exist for a mode and ask user if they want to use them.
    
    Also validates that the saved robot_type matches the currently connected hardware.
    
    Args:
        config: Main configuration dictionary
        mode: Mode name (e.g., 'calibration', 'teleop', 'recording', 'training', 'inference')
        mode_name: Display name for the mode (e.g., 'Calibration', 'Teleoperation')
        auto_use: If True, automatically use preconfigured settings without prompting
    
    Returns:
        Tuple of (preconfigured_args, detected_robot_type):
        - preconfigured_args: Preconfigured arguments if user chooses to use them, None otherwise
        - detected_robot_type: The detected robot type from hardware (useful when mismatch detected)
    """
    mode_config = load_mode_config(config, mode)
    
    # Always try to detect current hardware
    detected_type = None
    try:
        from solo.commands.robots.lerobot.scan import auto_detect_robot_type
        detected_type, _ = auto_detect_robot_type(verbose=False)
    except Exception:
        pass
    
    if mode_config:
        saved_robot_type = mode_config.get('robot_type')
        
        # Show current hardware detection status
        typer.echo(f"\n📋 Found preconfigured {mode_name} settings:")
        
        # Display robot type prominently first, with mismatch warning
        if saved_robot_type:
            if detected_type and detected_type != saved_robot_type:
                typer.echo(f"   ⚠️  robot_type: {saved_robot_type.upper()} (MISMATCH - detected {detected_type.upper()})")
            else:
                typer.echo(f"   • robot_type: {saved_robot_type.upper()}")
        elif detected_type:
            typer.echo(f"   • robot_type: (not saved, detected {detected_type.upper()})")
        
        # Display the rest of the configuration
        for key, value in mode_config.items():
            if key == 'robot_type':
                continue  # Already displayed above
            if isinstance(value, dict):
                typer.echo(f"   • {key}: {len(value)} items")
            else:
                typer.echo(f"   • {key}: {value}")
        
        # Validate robot type against currently connected hardware
        if saved_robot_type and detected_type and detected_type != saved_robot_type:
            typer.echo(f"\n⚠️  Robot type mismatch detected!")
            typer.echo(f"   Your saved configuration is for {saved_robot_type.upper()}")
            typer.echo(f"   But the connected hardware is {detected_type.upper()}")
            typer.echo(f"   Using the wrong config will cause motor errors.")
            
            use_detected = Confirm.ask(f"Use detected {detected_type.upper()} instead?", default=True)
            if use_detected:
                typer.echo(f"✅ Will use {detected_type.upper()} configuration")
                # Return None for config but pass detected_type so caller uses it
                return (None, detected_type)
            else:
                typer.echo(f"⚠️  Continuing with saved {saved_robot_type.upper()} config (may fail)")
        
        # If auto_use is True, skip the prompt and use configs directly
        if auto_use:
            typer.echo(f"\n✅ Using preconfigured {mode_name} settings")
            return (mode_config, detected_type)
        else:
            typer.echo(f"\nRunning {mode_name} with new settings, or use -y to automatically use preconfigured settings")
            return (None, detected_type)
    
    return (None, detected_type)


def save_teleop_config(config: dict, leader_port: str, follower_port: str, robot_type: str, camera_config: Dict, leader_id: str | None = None, follower_id: str | None = None) -> None:
    """Save teleoperation-specific configuration."""
    teleop_config = {
        'leader_port': leader_port,
        'follower_port': follower_port,
        'robot_type': robot_type,
        'camera_config': camera_config,
        'use_cameras': camera_config.get('enabled', False) if camera_config else False,
        'leader_id': leader_id,
        'follower_id': follower_id,
    }
    save_mode_config(config, 'teleop', teleop_config)


def save_recording_config(config: dict, recording_args: Dict) -> None:
    """Save recording-specific configuration."""
    recording_config = {
        'robot_type': recording_args.get('robot_type'),
        'leader_port': recording_args.get('leader_port'),
        'follower_port': recording_args.get('follower_port'),
        'camera_config': recording_args.get('camera_config'),
        'leader_id': recording_args.get('leader_id'),
        'follower_id': recording_args.get('follower_id'),
        'dataset_repo_id': recording_args.get('dataset_repo_id'),
        'task_description': recording_args.get('task_description'),
        'episode_time': recording_args.get('episode_time'),
        'num_episodes': recording_args.get('num_episodes'),
        'fps': recording_args.get('fps'),
        'push_to_hub': recording_args.get('push_to_hub'),
        'should_resume': recording_args.get('should_resume')
    }
    save_mode_config(config, 'recording', recording_config)


def save_training_config(config: dict, training_args: Dict) -> None:
    """Save training-specific configuration."""
    training_config = {
        'dataset_repo_id': training_args.get('dataset_repo_id'),
        'output_dir': training_args.get('output_dir'),
        'policy_type': training_args.get('policy_type'),
        'training_args': training_args.get('training_args', {})
    }
    save_mode_config(config, 'training', training_config)


def save_inference_config(config: dict, inference_args: Dict) -> None:
    """Save inference-specific configuration."""
    inference_config = {
        'robot_type': inference_args.get('robot_type'),
        'leader_port': inference_args.get('leader_port'),
        'leader_id': inference_args.get('leader_id'),
        'follower_id': inference_args.get('follower_id'),
        'follower_port': inference_args.get('follower_port'),
        'camera_config': inference_args.get('camera_config'),
        'policy_path': inference_args.get('policy_path'),
        'task_description': inference_args.get('task_description'),
        'inference_time': inference_args.get('inference_time'),
        'fps': inference_args.get('fps'),
        'use_teleoperation': inference_args.get('use_teleoperation')
    }
    save_mode_config(config, 'inference', inference_config)


def save_replay_config(config: dict, replay_args: Dict) -> None:
    """Save replay-specific configuration."""
    replay_config = {
        'robot_type': replay_args.get('robot_type'),
        'follower_port': replay_args.get('follower_port'),
        'follower_id': replay_args.get('follower_id'),
        'dataset_repo_id': replay_args.get('dataset_repo_id'),
        'episode': replay_args.get('episode'),
        'fps': replay_args.get('fps'),
        'play_sounds': replay_args.get('play_sounds'),
        'save_replay_as': replay_args.get('save_replay_as'),
        'save_replay_resume': replay_args.get('save_replay_resume'),
        'camera_config': replay_args.get('camera_config'),
        'repeat': replay_args.get('repeat'),
        'perturb': replay_args.get('perturb'),
        'perturb_increment': replay_args.get('perturb_increment'),
    }
    save_mode_config(config, 'replay', replay_config)


def save_deployx_config(config: dict, deployx_args: Dict) -> None:
    """Save DeployX edge-agent-specific configuration."""
    deployx_config = {
        'robot_type': deployx_args.get('robot_type'),
        'follower_port': deployx_args.get('follower_port'),
        'follower_id': deployx_args.get('follower_id'),
        'camera_config': deployx_args.get('camera_config'),
        'server_url': deployx_args.get('server_url'),
        'task_description': deployx_args.get('task_description'),
        'fps': deployx_args.get('fps'),
        'session_time': deployx_args.get('session_time'),
        'push_repo_id': deployx_args.get('push_repo_id'),
        'openwam_server_url': deployx_args.get('openwam_server_url'),
    }
    save_mode_config(config, 'deployx', deployx_config)


def update_mode_config_port(config: dict, mode: str, key: str, value: str) -> None:
    """
    Update one port in one mode's saved configuration.

    Unlike :func:`update_all_mode_config_ports`, this leaves the other modes alone,
    so correcting (say) the teleop leader port for one robot type cannot point a
    saved configuration for a different robot type at the wrong hardware.
    """
    mode_config = config.get('lerobot', {}).get('mode_configs', {}).get(mode)
    if not isinstance(mode_config, dict) or mode_config.get(key) == value:
        return

    mode_config[key] = value
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    typer.echo(f"📝 Updated {key} in saved {mode} settings: {value}")


def update_all_mode_config_ports(config: dict, leader_port: Optional[str] = None, follower_port: Optional[str] = None) -> None:
    """
    Update ports in all mode-specific configurations.
    
    This is called when auto-detection finds new ports after a connection failure.
    Updates teleop, recording, inference, and replay configs with the new ports.
    
    Args:
        config: Main configuration dictionary
        leader_port: New leader port (if None, won't update leader ports)
        follower_port: New follower port (if None, won't update follower ports)
    """
    if 'lerobot' not in config or 'mode_configs' not in config['lerobot']:
        return
    
    mode_configs = config['lerobot']['mode_configs']
    updated_modes = []
    
    # Modes that have leader_port
    leader_modes = ['teleop', 'recording', 'inference']
    # Modes that have follower_port
    follower_modes = ['teleop', 'recording', 'inference', 'replay']
    
    for mode, mode_config in mode_configs.items():
        if not isinstance(mode_config, dict):
            continue
        
        updated = False
        
        # Update leader_port if applicable
        if leader_port and mode in leader_modes and 'leader_port' in mode_config:
            mode_config['leader_port'] = leader_port
            updated = True
        
        # Update follower_port if applicable
        if follower_port and mode in follower_modes and 'follower_port' in mode_config:
            mode_config['follower_port'] = follower_port
            updated = True
        
        if updated:
            updated_modes.append(mode)
    
    if updated_modes:
        # Save to file
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        typer.echo(f"📝 Updated ports in preconfigured settings: {', '.join(updated_modes)}")
