"""
Teleoperation utilities for LeRobot

Note: Heavy lerobot imports are done lazily inside functions to speed up CLI startup.
"""

import typer
from rich.prompt import Confirm, Prompt
from typing import Optional

from solo.commands.robots.lerobot.config import (
    get_robot_config_classes,
    create_follower_config,
    create_leader_config,
    is_starai_robot,
    save_lerobot_config,
    is_bimanual_robot,
    is_realman_robot,
    create_bimanual_leader_config,
    create_bimanual_follower_config,
    validate_lerobot_config,
)
from solo.commands.robots.lerobot.mode_config import use_preconfigured_args
from solo.commands.robots.lerobot.ports import detect_and_retry_ports, detect_bimanual_arm_ports

def teleoperation(config: dict = None, auto_use: bool = False) -> bool:
    leader_id = None
    follower_id = None
    camera_config = None

    preconfigured, detected_robot_type = use_preconfigured_args(config, 'teleop', 'Teleoperation', auto_use=auto_use)
    if preconfigured:
        leader_port = preconfigured.get('leader_port')
        follower_port = preconfigured.get('follower_port')
        robot_type = preconfigured.get('robot_type')
        camera_config = preconfigured.get('camera_config')
        leader_id = preconfigured.get('leader_id')
        follower_id = preconfigured.get('follower_id')

        from solo.commands.robots.lerobot.starai_config import ensure_starai_leader_port
        leader_port = ensure_starai_leader_port(config, 'teleop', robot_type, leader_port)
        if not leader_port:
            return False


    if not preconfigured:
        # Validate configuration using utility function
        leader_port, follower_port, leader_calibrated, follower_calibrated, saved_robot_type = validate_lerobot_config(config)
        
        # Use detected robot type if available (e.g., from mismatch detection), otherwise use saved
        robot_type = detected_robot_type if detected_robot_type else saved_robot_type
        
        if not robot_type:
            from solo.commands.robots.lerobot.utils.helper import auto_detect_robot
            robot_type = auto_detect_robot(default="so101")
            config['robot_type'] = robot_type
        
        # Check if RealMan and handle network-based follower
        if is_realman_robot(robot_type):
            from solo.commands.robots.lerobot.utils.helper import get_realman_configs, port_detection
            
            # Leader is SO101 (USB)
            leader_port = port_detection(config, "leader", "so101", leader_port)
            
            # Follower is RealMan (network)
            config['realman_config'] = get_realman_configs(config)
            follower_port = None  # Network-based, no USB port
        
        # Check if bimanual and handle port detection accordingly
        elif is_bimanual_robot(robot_type):
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
            from solo.commands.robots.lerobot.utils.helper import port_detection
            leader_port = port_detection(config, "leader", robot_type, leader_port)
            follower_port = port_detection(config, "follower", robot_type, follower_port)
    
        # Prompt/select ids if not provided
        from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
        leader_id = prompt_arm_id(config, "leader", robot_type, leader_id)
        follower_id = prompt_arm_id(config, "follower", robot_type, follower_id)
        
        # Setup cameras if not provided
        if camera_config is None:
            # Check if cameras were previously configured
            lerobot_config = config.get('lerobot', {})
            prev_camera_config = lerobot_config.get('camera_config', {})
            cameras_previously_enabled = prev_camera_config.get('enabled', False)
            
            # Default to previous setting (no if never configured)
            use_camera = Confirm.ask("Would you like to setup cameras?", default=cameras_previously_enabled)
            if use_camera:
                from solo.commands.robots.lerobot.cameras import setup_cameras
                camera_config = setup_cameras()
            else:
                # Set empty camera config when user chooses not to use cameras
                camera_config = {'enabled': False, 'cameras': []}

    try:
        # Determine config classes based on robot type
        leader_config_class, follower_config_class = get_robot_config_classes(robot_type)
        
        if leader_config_class is None or follower_config_class is None:
            typer.echo(f"❌ Unsupported robot type for teleoperation: {robot_type}")
            return False
        
        # Debug: Show port/connection assignments
        typer.echo(f"\n🔌 Connection Configuration:")
        typer.echo(f"   • Leader port:   {leader_port}")
        if is_realman_robot(robot_type):
            realman_config = config.get('realman_config', {})
            typer.echo(f"   • Follower:      RealMan @ {realman_config.get('ip', 'N/A')}:{realman_config.get('port', 'N/A')}")
        else:
            typer.echo(f"   • Follower port: {follower_port}")
        
        # Create configurations based on robot type
        if is_realman_robot(robot_type):
            # RealMan: SO101 leader (USB) + RealMan follower (network)
            from solo.commands.robots.lerobot.realman_config import create_realman_follower_config
            
            # Create SO101 leader config
            leader_config = leader_config_class(port=leader_port, id=leader_id or "so101_leader")
            
            # Create RealMan follower config
            realman_config = config.get('realman_config', {})
            follower_config = create_realman_follower_config(
                realman_config,
                camera_config,
                follower_id=follower_id or "realman_r1d2_follower"
            )
        
        elif is_bimanual_robot(robot_type):
            # Create bimanual configurations
            lerobot_config = config.get('lerobot', {})
            left_leader_port = lerobot_config.get('left_leader_port')
            right_leader_port = lerobot_config.get('right_leader_port')
            left_follower_port = lerobot_config.get('left_follower_port')
            right_follower_port = lerobot_config.get('right_follower_port')
            
            leader_config = create_bimanual_leader_config(
                leader_config_class,
                left_leader_port,
                right_leader_port,
                robot_type,
                leader_id=leader_id
            )
            
            follower_config = create_bimanual_follower_config(
                follower_config_class,
                left_follower_port,
                right_follower_port,
                robot_type,
                camera_config,
                follower_id=follower_id
            )
        else:
            # Create single-arm configurations
            leader_config = create_leader_config(
                leader_config_class,
                leader_port,
                robot_type,
                leader_id=leader_id,
                follower_id=follower_id,
            )
            
            # Create robot config with cameras if enabled
            follower_config = create_follower_config(
                follower_config_class,
                follower_port,
                robot_type,
                camera_config,
                follower_id=follower_id,
            )
        
        # Lazy import heavy lerobot modules
        from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
    
        # Create teleoperation config
        teleop_config = TeleoperateConfig(
            teleop=leader_config,
            robot=follower_config,
            fps=60,
            display_data=True
        )
        
        # Save configuration before execution (if not using preconfigured settings)
        if config and not preconfigured:
            from .mode_config import save_teleop_config
            if is_bimanual_robot(robot_type):
                # For bimanual, we don't use the standard save_teleop_config
                # Configuration is already saved via save_lerobot_config
                pass
            else:
                save_teleop_config(
                    config,
                    leader_port,
                    follower_port,
                    robot_type,
                    camera_config,
                    leader_id,
                    follower_id,
                )
        
        if is_realman_robot(robot_type):
            typer.echo("🎮 Starting RealMan teleoperation... Press Ctrl+C to stop.")
            typer.echo("📋 Move the SO101 leader arm to control the RealMan follower arm.")
            typer.echo("⚠️  Note: RealMan uses network connection - ensure robot is powered and connected.")
        elif is_bimanual_robot(robot_type):
            typer.echo("🎮 Starting bimanual teleoperation... Press Ctrl+C to stop.")
            typer.echo("📋 Move BOTH leader arms to control BOTH follower arms.")
        elif is_starai_robot(robot_type):
            from solo.commands.robots.lerobot.starai_config import describe_starai_map
            typer.echo("🎮 Starting teleoperation... Press Ctrl+C to stop.")
            typer.echo("📋 Move the Star Arm 102 leader to control the SO101 follower.")
            describe_starai_map()
            typer.echo("\n💡 If a joint runs backwards or over/under-travels, stop and run 'solo robo --star-tune'.")
        else:
            typer.echo("🎮 Starting teleoperation... Press Ctrl+C to stop.")
            typer.echo("📋 Move the leader arm to control the follower arm.")
        
        # Start teleoperation with retry logic
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                # Use standard lerobot teleoperate (stability is now handled in lerobot itself)
                teleoperate(teleop_config)
                
                return True
                
            except Exception as e:
                error_msg = str(e)
                
                # Enhanced error reporting for sync_read failures
                if "sync read" in error_msg.lower() or "sync_read" in error_msg.lower():
                    typer.echo(f"\n⚠️  Motor communication error detected!")
                    typer.echo(f"   Error: {error_msg}")
                    typer.echo("")
                    typer.echo("🔍 Running quick diagnostics...")
                    try:
                        from solo.commands.robots.lerobot.scan import diagnose_connection
                        typer.echo(f"\n   Leader port ({leader_port}):")
                        diagnose_connection(leader_port, verbose=True)
                        typer.echo(f"\n   Follower port ({follower_port}):")
                        diagnose_connection(follower_port, verbose=True)
                    except Exception as diag_err:
                        typer.echo(f"   (Diagnostic failed: {diag_err})")
                    typer.echo("")
                    typer.echo("💡 Possible causes:")
                    typer.echo("   1. Motor power issue - check 12V supply")
                    typer.echo("   2. USB timing - try unplugging and waiting 2 seconds")
                    typer.echo("   3. Port swapped - run 'solo robo --scan' to verify")
                    typer.echo("   4. Loose cable in daisy chain")
                    return False
                
                # Check if it's a port connection error
                if "Could not connect on port" in error_msg or "Make sure you are using the correct port" in error_msg:
                    if attempt < max_retries:
                        typer.echo(f"❌ Connection failed: {error_msg}")
                        typer.echo("🔄 Attempting to detect new ports...")
                        
                        # Detect new ports and retry
                        new_leader_port, new_follower_port = detect_and_retry_ports(leader_port, follower_port, config)
                        
                        if new_leader_port != leader_port or new_follower_port != follower_port:
                            # Update ports and recreate configs
                            leader_port, follower_port = new_leader_port, new_follower_port
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
                            teleop_config = TeleoperateConfig(
                                teleop=leader_config,
                                robot=follower_config,
                                fps=60,
                                display_data=True
                            )
                            typer.echo("🔄 Retrying teleoperation with new ports...")
                            continue
                        else:
                            typer.echo("❌ Could not find new ports. Please check connections.")
                            return False
                    else:
                        typer.echo(f"❌ Teleoperation failed after retry: {error_msg}")
                        return False
                else:
                    # Non-port related error
                    typer.echo(f"❌ Teleoperation failed: {error_msg}")
                    return False
        
    except KeyboardInterrupt:
        typer.echo("\n🛑 Teleoperation stopped by user.")
        return True
