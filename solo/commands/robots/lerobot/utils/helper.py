"""
Helper utilities for LeRobot modes.

This module contains reusable functions to reduce code duplication across
teleoperation, recording, inference, replay, and calibration modes.
"""

import typer
from typing import Optional
from rich.prompt import Prompt, Confirm


# Robot type selection menu - used across multiple modes
ROBOT_TYPE_MENU = {
    1: ("SO101", "so101"),
    2: ("SO100", "so100"),
    3: ("Koch", "koch"),
    4: ("Bimanual SO100", "bi_so100"),
    5: ("Bimanual SO101", "bi_so101"),
    6: ("RealMan R1D2 - SO101 leader", "realman_r1d2"),
    7: ("Star Arm 102 leader - SO101 follower", "stararm102"),
}


def prompt_robot_type_selection(default: str = "so101") -> str:
    """
    Display robot type selection menu and return the selected type.
    
    Args:
        default: Default robot type if user just presses Enter
    
    Returns:
        Selected robot type string (e.g., "so101", "koch", "bi_so100")
    """
    typer.echo("\n🤖 Select your robot type:")
    for num, (label, _) in ROBOT_TYPE_MENU.items():
        typer.echo(f"{num}. {label}")
    
    # Find default number from default type
    default_num = "1"
    for num, (_, rtype) in ROBOT_TYPE_MENU.items():
        if rtype == default:
            default_num = str(num)
            break
    
    robot_choice = int(Prompt.ask("Enter robot type", default=default_num))
    
    if robot_choice in ROBOT_TYPE_MENU:
        return ROBOT_TYPE_MENU[robot_choice][1]
    return default


def auto_detect_robot(default: str = "so101") -> str:
    """
    Auto-detect robot type from connected hardware, or prompt for manual selection.
    
    Args:
        default: Default robot type if auto-detection fails and user accepts default
    
    Returns:
        Detected or selected robot type string
    """
    try:
        from solo.commands.robots.lerobot.scan import auto_detect_robot_type
        detected_type, port_info = auto_detect_robot_type(verbose=True)
        
        if detected_type:
            typer.echo(f"\n🤖 Auto-detected robot type: {detected_type.upper()}")
            use_detected = Confirm.ask("Use this robot type?", default=True)
            if use_detected:
                return detected_type
        
        # Manual selection if not detected or user declined
        return prompt_robot_type_selection(default=default)
    except Exception as e:
        typer.echo(f"⚠️  Auto-detection failed: {e}")
        return prompt_robot_type_selection(default=default)


def get_realman_configs(config: dict) -> dict:
    """
    Load RealMan configuration from YAML and merge with saved settings.
    
    Always loads fresh config from YAML to pick up changes (like invert_joints),
    but preserves saved network settings (ip/port) from the config.
    
    Args:
        config: Main configuration dictionary
    
    Returns:
        Merged RealMan configuration dictionary
    """
    from solo.commands.robots.lerobot.realman_config import load_realman_config
    
    realman_config = load_realman_config()
    
    # Merge with any saved network settings (ip/port) if they exist
    lerobot_config = config.get('lerobot', {})
    saved_realman = lerobot_config.get('realman_config', {})
    if saved_realman:
        realman_config['ip'] = saved_realman.get('ip', realman_config['ip'])
        realman_config['port'] = saved_realman.get('port', realman_config['port'])
    
    return realman_config


def port_detection(config: dict, arm_type: str, robot_type: str, current_port: Optional[str] = None) -> Optional[str]:
    """
    Detect port for an arm if not already set, and update config.
    
    Args:
        config: Main configuration dictionary (will be updated with detected port)
        arm_type: "leader" or "follower"
        robot_type: Robot type string (e.g., "so101", "koch")
        current_port: Current port value (if any)
    
    Returns:
        Detected or existing port string, or None if detection failed
    """
    from solo.commands.robots.lerobot.config import is_starai_robot

    if arm_type == "leader" and is_starai_robot(robot_type):
        # A saved leader port may belong to a different leader entirely, and the
        # FashionStar driver fails obscurely on the wrong port, so verify it.
        from solo.commands.robots.lerobot.starai_config import resolve_starai_leader_port
        resolved = resolve_starai_leader_port(current_port)
        if resolved:
            config[f'{arm_type}_port'] = resolved
        return resolved

    if current_port:
        return current_port

    from solo.commands.robots.lerobot.ports import detect_arm_port
    
    detected_port, _ = detect_arm_port(arm_type, robot_type=robot_type)
    if detected_port:
        config[f'{arm_type}_port'] = detected_port
    
    return detected_port


def prompt_arm_id(config: dict, arm_type: str, robot_type: str, current_id: Optional[str] = None) -> str:
    """
    Display known IDs and prompt user to select or enter an arm ID.
    
    Args:
        config: Main configuration dictionary
        arm_type: "leader" or "follower"
        robot_type: Robot type string for filtering known IDs
        current_id: Current ID value (if already set, returns immediately)
    
    Returns:
        Selected or entered arm ID string
    """
    if current_id:
        return current_id
    
    from solo.commands.robots.lerobot.config import get_known_ids, display_known_ids
    
    # Get known IDs for this robot type
    known_leader_ids, known_follower_ids = get_known_ids(config, robot_type=robot_type)
    known_ids = known_leader_ids if arm_type == "leader" else known_follower_ids
    
    # Get default ID from config or generate one. Only reuse the saved id when it
    # was recorded for this robot type — otherwise switching arms silently offers
    # the other arm's id, which points calibration at the wrong file.
    lerobot_config = config.get('lerobot', {})
    saved_id = lerobot_config.get(f'{arm_type}_id')
    if saved_id and lerobot_config.get('robot_type') not in (None, robot_type):
        saved_id = None
    default_id = saved_id or f"{robot_type}_{arm_type}"
    
    # Display known IDs
    display_known_ids(known_ids, arm_type, detected_robot_type=robot_type, config=config)
    
    # Prompt for ID
    return Prompt.ask(f"Enter {arm_type} id", default=default_id)

