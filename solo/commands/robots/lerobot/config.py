"""
Configuration utilities for LeRobot
"""

import json
import os
import typer
from typing import Optional, Tuple, TYPE_CHECKING, Dict, List
from rich.prompt import Prompt
from solo.config import CONFIG_PATH

if TYPE_CHECKING:
    from lerobot.scripts.lerobot_record import RecordConfig


def validate_lerobot_config(config: dict) -> tuple[Optional[str], Optional[str], bool, bool, str]:
    """
    Extract and validate lerobot configuration from main config.
    Returns: (leader_port, follower_port, leader_calibrated, follower_calibrated, robot_type)
    """
    lerobot_config = config.get('lerobot', {})
    leader_port = lerobot_config.get('leader_port')
    follower_port = lerobot_config.get('follower_port')
    leader_calibrated = lerobot_config.get('leader_calibrated', False)
    follower_calibrated = lerobot_config.get('follower_calibrated', False)
    robot_type = lerobot_config.get('robot_type')
    
    return leader_port, follower_port, leader_calibrated, follower_calibrated, robot_type


def save_lerobot_config(config: dict, arm_config: dict) -> None:
    """Save lerobot configuration to config file."""
    if 'lerobot' not in config:
        config['lerobot'] = {}
    config['lerobot'].update(arm_config)
    
    # Update server type
    if 'server' not in config:
        config['server'] = {}
    config['server']['type'] = 'lerobot'
    
    # Save to file
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    
    typer.echo(f"\nConfiguration saved to {CONFIG_PATH}")


def migrate_known_ids_to_structured(config: dict) -> bool:
    """
    Migrate legacy flat known_leader_ids/known_follower_ids lists 
    to the new structured known_ids_by_type format.
    
    Returns True if migration was performed and saved.
    """
    lerobot_config = config.get('lerobot', {})
    
    # Check if migration is needed
    legacy_leaders = lerobot_config.get('known_leader_ids', [])
    legacy_followers = lerobot_config.get('known_follower_ids', [])
    
    if not legacy_leaders and not legacy_followers:
        return False  # Nothing to migrate
    
    if 'known_ids_by_type' not in lerobot_config:
        lerobot_config['known_ids_by_type'] = {}
    
    known_ids_by_type = lerobot_config['known_ids_by_type']
    
    # Migrate leaders
    for lid in legacy_leaders:
        inferred_type = infer_robot_type_from_id(lid) or 'unknown'
        if inferred_type not in known_ids_by_type:
            known_ids_by_type[inferred_type] = {'leaders': [], 'followers': []}
        if lid not in known_ids_by_type[inferred_type].get('leaders', []):
            known_ids_by_type[inferred_type].setdefault('leaders', []).append(lid)
    
    # Migrate followers
    for fid in legacy_followers:
        inferred_type = infer_robot_type_from_id(fid) or 'unknown'
        if inferred_type not in known_ids_by_type:
            known_ids_by_type[inferred_type] = {'leaders': [], 'followers': []}
        if fid not in known_ids_by_type[inferred_type].get('followers', []):
            known_ids_by_type[inferred_type].setdefault('followers', []).append(fid)
    
    config['lerobot']['known_ids_by_type'] = known_ids_by_type
    
    # Save updated config
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    
    return True


def infer_robot_type_from_id(arm_id: str) -> Optional[str]:
    """Infer robot type from arm ID name pattern."""
    arm_id_lower = arm_id.lower()
    if 'koch' in arm_id_lower:
        return 'koch'
    elif 'so101' in arm_id_lower:
        return 'so101'
    elif 'so100' in arm_id_lower:
        return 'so100'
    elif 'realman' in arm_id_lower or 'r1d2' in arm_id_lower:
        return 'realman'
    elif 'stararm102' in arm_id_lower or 'star' in arm_id_lower:
        return 'stararm102'
    elif 'bi_so100' in arm_id_lower or 'biso100' in arm_id_lower:
        return 'bi_so100'
    elif 'bi_so101' in arm_id_lower or 'biso101' in arm_id_lower:
        return 'bi_so101'
    return None


def get_known_ids(config: dict, robot_type: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Return known leader and follower ids from config.
    
    If robot_type is provided, returns only IDs for that robot type.
    Otherwise returns all IDs (for backward compatibility).
    
    New structure: lerobot.known_ids_by_type = {
        "koch": {"leaders": [...], "followers": [...]},
        "so101": {"leaders": [...], "followers": [...]}
    }
    
    Also supports legacy flat lists for backward compatibility.
    """
    lerobot_config = config.get('lerobot', {})
    
    # Check for new structure first
    known_ids_by_type = lerobot_config.get('known_ids_by_type', {})
    
    if robot_type and robot_type in known_ids_by_type:
        # Return IDs specific to this robot type
        type_ids = known_ids_by_type[robot_type]
        return type_ids.get('leaders', []), type_ids.get('followers', [])
    
    # If robot_type specified but not found, or no robot_type specified
    # Aggregate all IDs from all robot types
    all_leaders = []
    all_followers = []
    
    for rtype, type_ids in known_ids_by_type.items():
        for lid in type_ids.get('leaders', []):
            if lid not in all_leaders:
                all_leaders.append(lid)
        for fid in type_ids.get('followers', []):
            if fid not in all_followers:
                all_followers.append(fid)
    
    # Also include legacy flat lists for backward compatibility
    legacy_leaders = lerobot_config.get('known_leader_ids', [])
    legacy_followers = lerobot_config.get('known_follower_ids', [])
    
    for lid in legacy_leaders:
        if lid not in all_leaders:
            all_leaders.append(lid)
    for fid in legacy_followers:
        if fid not in all_followers:
            all_followers.append(fid)
    
    return all_leaders, all_followers


def get_known_ids_by_type(config: dict) -> Dict[str, Dict[str, List[str]]]:
    """
    Return all known IDs organized by robot type.
    
    Returns: {"koch": {"leaders": [...], "followers": [...]}, ...}
    """
    lerobot_config = config.get('lerobot', {})
    known_ids_by_type = lerobot_config.get('known_ids_by_type', {}).copy()
    
    # Migrate legacy flat lists by inferring robot type from ID names
    legacy_leaders = lerobot_config.get('known_leader_ids', [])
    legacy_followers = lerobot_config.get('known_follower_ids', [])
    
    for lid in legacy_leaders:
        inferred_type = infer_robot_type_from_id(lid) or 'unknown'
        if inferred_type not in known_ids_by_type:
            known_ids_by_type[inferred_type] = {'leaders': [], 'followers': []}
        if lid not in known_ids_by_type[inferred_type].get('leaders', []):
            known_ids_by_type[inferred_type].setdefault('leaders', []).append(lid)
    
    for fid in legacy_followers:
        inferred_type = infer_robot_type_from_id(fid) or 'unknown'
        if inferred_type not in known_ids_by_type:
            known_ids_by_type[inferred_type] = {'leaders': [], 'followers': []}
        if fid not in known_ids_by_type[inferred_type].get('followers', []):
            known_ids_by_type[inferred_type].setdefault('followers', []).append(fid)
    
    return known_ids_by_type


def format_id_with_robot_type(arm_id: str, detected_robot_type: Optional[str] = None) -> str:
    """Format an arm ID with its inferred or detected robot type."""
    robot_type = infer_robot_type_from_id(arm_id)
    if robot_type:
        return f"{arm_id} ({robot_type})"
    elif detected_robot_type:
        # If ID doesn't have robot type in name, show detected type
        return f"{arm_id} ({detected_robot_type}?)"
    return arm_id


def display_known_ids(known_ids: List[str], arm_type: str, detected_robot_type: Optional[str] = None, config: Optional[dict] = None) -> None:
    """Display known IDs filtered by robot type.
    
    Only shows IDs that match the detected/selected robot type.
    IDs from other robot types are hidden to avoid confusion.
    
    Args:
        known_ids: List of known arm IDs (for backward compatibility, can be empty if using config)
        arm_type: "leader" or "follower"
        detected_robot_type: Currently detected/selected robot type
        config: Config dictionary to get IDs organized by type (preferred)
    """
    import typer
    
    # Try to auto-detect robot type if not provided
    if detected_robot_type is None:
        try:
            from solo.commands.robots.lerobot.scan import auto_detect_robot_type
            detected_robot_type, _ = auto_detect_robot_type(verbose=False)
        except Exception:
            pass
    
    # If config provided, use structured IDs by type
    if config:
        ids_by_type = get_known_ids_by_type(config)
        key = 'leaders' if arm_type == 'leader' else 'followers'
        
        # Only show IDs that match the detected robot type
        matching_ids = []
        
        if detected_robot_type and detected_robot_type in ids_by_type:
            # Get IDs for the detected robot type
            for arm_id in ids_by_type[detected_robot_type].get(key, []):
                matching_ids.append(arm_id)
        
        # Also include IDs marked as 'unknown' type (legacy IDs without type info)
        if 'unknown' in ids_by_type:
            for arm_id in ids_by_type['unknown'].get(key, []):
                if arm_id not in matching_ids:
                    matching_ids.append(arm_id)
        
        if matching_ids:
            typer.echo(f"📇 Known {arm_type} ids for {detected_robot_type.upper() if detected_robot_type else 'unknown'}:")
            for i, arm_id in enumerate(matching_ids, 1):
                typer.echo(f"   {i}. {arm_id}")
        return
    
    # Fallback to old behavior with flat list - filter by inferred type
    if known_ids:
        filtered_ids = []
        for kid in known_ids:
            inferred_type = infer_robot_type_from_id(kid)
            # Include if type matches, or if no type could be inferred (legacy ID)
            if inferred_type is None or inferred_type == detected_robot_type:
                filtered_ids.append(kid)
        
        if filtered_ids:
            typer.echo(f"📇 Known {arm_type} ids for {detected_robot_type.upper() if detected_robot_type else 'unknown'}:")
            for i, kid in enumerate(filtered_ids, 1):
                typer.echo(f"   {i}. {kid}")


def add_known_id(config: dict, arm_type: str, arm_id: str, robot_type: Optional[str] = None) -> None:
    """
    Persist a discovered or chosen id for leader/follower in the config.
    
    Args:
        config: Main configuration dictionary
        arm_type: "leader" or "follower"
        arm_id: The arm ID to add
        robot_type: Robot type to associate with this ID (if None, inferred from ID name)
    """
    if 'lerobot' not in config:
        config['lerobot'] = {}
    
    # Determine robot type - use provided, infer from ID, or default to 'unknown'
    effective_robot_type = robot_type or infer_robot_type_from_id(arm_id) or 'unknown'
    
    # Initialize known_ids_by_type structure if needed
    if 'known_ids_by_type' not in config['lerobot']:
        config['lerobot']['known_ids_by_type'] = {}
    
    if effective_robot_type not in config['lerobot']['known_ids_by_type']:
        config['lerobot']['known_ids_by_type'][effective_robot_type] = {'leaders': [], 'followers': []}
    
    # Add to appropriate list
    key = 'leaders' if arm_type == 'leader' else 'followers'
    existing: List[str] = config['lerobot']['known_ids_by_type'][effective_robot_type].get(key, [])
    
    if arm_id and arm_id not in existing:
        existing.append(arm_id)
        config['lerobot']['known_ids_by_type'][effective_robot_type][key] = existing
        
        # Also add to legacy flat list for backward compatibility
        legacy_key = 'known_leader_ids' if arm_type == 'leader' else 'known_follower_ids'
        legacy_list: List[str] = config['lerobot'].get(legacy_key, [])
        if arm_id not in legacy_list:
            legacy_list.append(arm_id)
            config['lerobot'][legacy_key] = legacy_list
        
        # Save immediately to disk
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)


def get_robot_config_classes(robot_type: str) -> Tuple[Optional[type], Optional[type]]:
    """
    Get the appropriate config classes for leader and follower based on robot type
    Returns (leader_config_class, follower_config_class)
    
    Uses lazy loading to only import config classes when actually needed.
    
    For RealMan robots, the leader is always SO101 (USB) and follower is RealMan (network).
    """
    if robot_type == "so100":
        from lerobot.teleoperators.so_leader import SO100LeaderConfig
        from lerobot.robots.so_follower import SO100FollowerConfig
        return SO100LeaderConfig, SO100FollowerConfig
    elif robot_type == "so101":
        from lerobot.teleoperators.so_leader import SO101LeaderConfig
        from lerobot.robots.so_follower import SO101FollowerConfig
        return SO101LeaderConfig, SO101FollowerConfig
    elif robot_type == "koch":
        from lerobot.teleoperators.koch_leader import KochLeaderConfig
        from lerobot.robots.koch_follower import KochFollowerConfig
        return KochLeaderConfig, KochFollowerConfig
    elif robot_type == "bi_so100":
        from lerobot.teleoperators.bi_so_leader import BiSO100LeaderConfig
        from lerobot.robots.bi_so_follower import BiSO100FollowerConfig
        return BiSO100LeaderConfig, BiSO100FollowerConfig
    elif robot_type == "bi_so101":
        from lerobot.teleoperators.bi_so_leader import BiSO101LeaderConfig
        from lerobot.robots.bi_so_follower import BiSO101FollowerConfig
        return BiSO101LeaderConfig, BiSO101FollowerConfig
    elif robot_type == "stararm102":
        # Star Arm 102 leader (FashionStar UART bus) driving an SO101 follower.
        # The leader speaks a different protocol and has one joint more than the
        # SO101, so it goes through Solo's remapping adapter rather than a stock
        # LeRobot teleoperator.
        from solo.commands.robots.lerobot.teleoperators import StarArm102SO101LeaderConfig
        from lerobot.robots.so_follower import SO101FollowerConfig
        return StarArm102SO101LeaderConfig, SO101FollowerConfig
    elif robot_type in ["realman_r1d2", "realman_rm65", "realman_rm75"]:
        # RealMan robots use SO101 as leader arm (USB serial)
        # and RealMan arm as follower (network connection)
        from lerobot.teleoperators.so_leader import SO101LeaderConfig
        from lerobot.robots.realman_follower import RealManFollowerConfig
        return SO101LeaderConfig, RealManFollowerConfig
    else:
        return None, None


def is_bimanual_robot(robot_type: str) -> bool:
    """Check if robot type is bimanual"""
    return robot_type in ["bi_so100", "bi_so101"]


def is_realman_robot(robot_type: str) -> bool:
    """
    Check if robot type is a RealMan robot (network-connected follower).
    
    RealMan robots connect via IP/port instead of USB serial.
    They use SO101 as the leader arm for teleoperation.
    """
    return robot_type in ["realman_r1d2", "realman_rm65", "realman_rm75"]


def is_starai_robot(robot_type: str) -> bool:
    """
    Check if robot type uses a Star Arm 102 (StarAI / Fashionstar) leader arm.

    These pair a FashionStar-bus leader with a standard SO101 follower, so the
    leader action has to be remapped onto SO101 joint names before it is sent.
    """
    from solo.commands.robots.lerobot.starai_config import is_starai_robot as _is_starai
    return _is_starai(robot_type)


def get_realman_model_from_type(robot_type: str) -> str:
    """
    Get the RealMan model name from robot type.
    
    Args:
        robot_type: Robot type string (e.g., "realman_r1d2")
        
    Returns:
        Model name (e.g., "R1D2")
    """
    model_map = {
        "realman_r1d2": "R1D2",
        "realman_rm65": "RM65",
        "realman_rm75": "RM75",
    }
    return model_map.get(robot_type, "R1D2")


def normalize_fps(requested_fps: float) -> int:
    """
    Normalize FPS to common supported values.
    Defaults to 30 FPS (most widely supported) unless specifically close to 60.
    """
    # Round to clean integer first
    fps_int = round(requested_fps)
    
    # If very close to 60, use 60 FPS
    if fps_int >= 55:
        return 60
    # Otherwise default to 30 FPS (most compatible)
    else:
        return 30


def build_camera_configuration(camera_config: Dict) -> Dict:
    """
    Build camera configuration dictionary from camera_config
    Returns cameras_dict for robot configuration
    """
    if not camera_config or not camera_config.get('enabled', False):
        return {}
    
    # Import camera configuration classes
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    
    cameras_dict = {}
    for cam in camera_config.get('cameras', []):
        camera_name = cam['angle']  # Use angle as camera name
        cam_info = cam['camera_info']
        
        # Create camera config based on type
        if cam['camera_type'] == 'OpenCV':
            stream_profile = cam_info.get('default_stream_profile') or {}
            requested_fps = stream_profile.get('fps', 30)
            # Normalize FPS to avoid hardware mismatch issues
            normalized_fps = normalize_fps(requested_fps)
            
            cameras_dict[camera_name] = OpenCVCameraConfig(
                index_or_path=cam_info.get('id', 0),
                width=stream_profile.get('width', 640),
                height=stream_profile.get('height', 480),
                fps=normalized_fps
            )
        elif cam['camera_type'] == 'RealSense':
            stream_profile = cam_info.get('default_stream_profile') or {}
            requested_fps = stream_profile.get('fps', 30)
            # Normalize FPS to avoid hardware mismatch issues  
            normalized_fps = normalize_fps(requested_fps)
            
            cameras_dict[camera_name] = RealSenseCameraConfig(
                serial_number_or_name=str(cam_info.get('id', '')),
                width=stream_profile.get('width', 640),
                height=stream_profile.get('height', 480),
                fps=normalized_fps
            )
    
    return cameras_dict


def create_follower_config(
    follower_config_class,
    follower_port: str,
    robot_type: str,
    camera_config: Dict = None,
    follower_id: Optional[str] = None,
):
    """
    Create follower configuration with optional camera support (single-arm robots)
    """
    cameras_dict = build_camera_configuration(camera_config or {})

    kwargs = {
        'port': follower_port,
        'id': follower_id or f"{robot_type}_follower",
    }
    if cameras_dict:
        kwargs['cameras'] = cameras_dict

    if is_starai_robot(robot_type):
        # The Star Arm 102 leader is not the SO101's twin, so a mis-signed or
        # badly scaled joint would otherwise command a large jump. Capping the
        # per-step move keeps a mapping mistake to a slow drift the operator can
        # stop, at the cost of one extra follower read per control step.
        from solo.commands.robots.lerobot.starai_config import starai_max_relative_target
        max_relative_target = starai_max_relative_target()
        if max_relative_target is not None:
            kwargs['max_relative_target'] = max_relative_target

    return follower_config_class(**kwargs)


def create_leader_config(
    leader_config_class,
    leader_port: str,
    robot_type: str,
    leader_id: Optional[str] = None,
    follower_id: Optional[str] = None,
):
    """
    Create leader configuration for single-arm robots.

    Star Arm 102 leaders carry the saved joint mapping (and the follower's travel
    limits, when the follower id is known); every other leader just needs a port
    and an id.
    """
    if is_starai_robot(robot_type):
        from solo.commands.robots.lerobot.starai_config import create_starai_leader_config
        return create_starai_leader_config(
            port=leader_port,
            leader_id=leader_id or f"{robot_type}_leader",
            follower_id=follower_id,
        )

    return leader_config_class(port=leader_port, id=leader_id or f"{robot_type}_leader")


def create_bimanual_leader_config(
    leader_config_class,
    left_leader_port: str,
    right_leader_port: str,
    robot_type: str,
    leader_id: Optional[str] = None,
):
    """
    Create bimanual leader configuration
    """
    return leader_config_class(
        left_arm_port=left_leader_port,
        right_arm_port=right_leader_port,
        id=leader_id or f"{robot_type}_leader"
    )


def create_bimanual_follower_config(
    follower_config_class,
    left_follower_port: str,
    right_follower_port: str,
    robot_type: str,
    camera_config: Dict = None,
    follower_id: Optional[str] = None,
):
    """
    Create bimanual follower configuration with optional camera support
    """
    cameras_dict = build_camera_configuration(camera_config or {})
    
    return follower_config_class(
        left_arm_port=left_follower_port,
        right_arm_port=right_follower_port,
        id=follower_id or f"{robot_type}_follower",
        cameras=cameras_dict if cameras_dict else {}
    )


def create_robot_configs(
    robot_type: str,
    leader_port: str,
    follower_port: str,
    camera_config: Dict = None,
    leader_id: Optional[str] = None,
    follower_id: Optional[str] = None,
) -> tuple[Optional[object], Optional[object]]:
    """
    Create leader and follower configurations for given robot type.
    Returns: (leader_config, follower_config)
    """
    leader_config_class, follower_config_class = get_robot_config_classes(robot_type)
    
    if leader_config_class is None or follower_config_class is None:
        typer.echo(f"❌ Unsupported robot type: {robot_type}")
        return None, None
    
    leader_config = create_leader_config(
        leader_config_class,
        leader_port,
        robot_type,
        leader_id=leader_id,
        follower_id=follower_id,
    )
    follower_config = create_follower_config(
        follower_config_class,
        follower_port,
        robot_type,
        camera_config,
        follower_id=follower_id,
    )
    
    return leader_config, follower_config
