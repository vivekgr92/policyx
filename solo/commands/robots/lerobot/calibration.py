"""
Calibration utilities for LeRobot

Note: Heavy lerobot imports are done lazily inside functions to speed up CLI startup.
      A background preloader starts the import while the user answers prompts.
"""

import typer
from rich.prompt import Prompt, Confirm
from typing import Dict
from typing import Optional

# Light imports - these don't load torch/transformers
from solo.commands.robots.lerobot.ports import detect_arm_port, detect_bimanual_arm_ports
from solo.commands.robots.lerobot.config import (
    get_robot_config_classes,
    create_leader_config,
    is_starai_robot,
    save_lerobot_config,
    get_known_ids,
    add_known_id,
    is_bimanual_robot,
    is_realman_robot,
    create_bimanual_leader_config,
    create_bimanual_follower_config,
)
from solo.commands.robots.lerobot.realman_config import load_realman_config, prompt_realman_config, test_realman_connection
from solo.commands.robots.lerobot.utils.preload import start_lerobot_preload, wait_for_lerobot_preload

def calibrate_arm(arm_type: str, port: str, robot_type: str = "so100", arm_id: Optional[str] = None) -> bool:
    """
    Calibrate a specific arm using the lerobot calibration system
    """
    wait_for_lerobot_preload()
    from lerobot.scripts.lerobot_calibrate import calibrate, CalibrateConfig
    
    try:
        # Determine the appropriate config class based on arm type and robot type
        leader_config_class, follower_config_class = get_robot_config_classes(robot_type)
        
        if leader_config_class is None or follower_config_class is None:
            typer.echo(f"❌ Unsupported robot type for {arm_type}: {robot_type}")
            return False

        if arm_type == "leader":
            arm_config = create_leader_config(
                leader_config_class, port, robot_type, leader_id=arm_id
            )
            calibrate_config = CalibrateConfig(teleop=arm_config)
        else:
            arm_config = follower_config_class(port=port, id=arm_id or f"{robot_type}_{arm_type}")
            calibrate_config = CalibrateConfig(robot=arm_config)
        
        typer.echo(f"🔧 Calibrating {arm_type} arm on port {port}...")
        typer.echo("⚠️  Please follow the calibration instructions.")

        if arm_type == "leader" and is_starai_robot(robot_type):
            typer.echo(
                "   Sweep every Star Arm 102 joint through its full travel — the recorded\n"
                "   range sets the midpoint each joint angle is measured from, and the SO101\n"
                "   follower is driven from those angles."
            )
        
        calibrate(calibrate_config)
        typer.echo(f"✅ {arm_type.title()} arm calibrated successfully!")
        return True
        
    except Exception as e:
        typer.echo(f"❌ {arm_type.title()} arm calibration failed: {e}")
        return False


def calibrate_realman_follower(realman_cfg: Dict, follower_id: str) -> bool:
    """
    Calibrate RealMan follower arm by recording joint ranges and center position.
    
    Args:
        realman_cfg: Dictionary containing RealMan configuration (ip, port, etc.)
        follower_id: ID to assign to this follower arm
    
    Returns:
        True if calibration succeeded, False otherwise
    """
    
    wait_for_lerobot_preload()
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.realman_follower import RealManFollowerConfig
    
    try:
        # Create RealManFollowerConfig
        follower_config = RealManFollowerConfig(
            ip=realman_cfg['ip'],
            port=realman_cfg['port'],
            model=realman_cfg['model'],
            velocity=realman_cfg.get('velocity', 100),
            id=follower_id
        )
        
        # Create robot from config
        robot = make_robot_from_config(follower_config)
        
        # Connect to robot (calibrate=False to avoid double calibration)
        typer.echo("\n🌐 Connecting to RealMan follower...")
        robot.connect(calibrate=False)
        
        # Run calibration (records joint ranges and center position)
        typer.echo("📏 Starting joint range calibration...")
        typer.echo("⚠️  You will be prompted to move each joint to its min and max positions.\n")
        robot.calibrate()
        
        # Disconnect
        robot.disconnect()
        
        typer.echo(f"\n✅ RealMan follower calibrated successfully!")
        return True
    
    except Exception as e:
        typer.echo(f"❌ RealMan follower calibration failed: {str(e)}")
        import traceback
        typer.echo(traceback.format_exc())
        return False


def calibrate_bimanual_arm(
    arm_type: str, 
    left_port: str, 
    right_port: str, 
    robot_type: str = "bi_so100", 
    arm_id: Optional[str] = None
) -> bool:
    """
    Calibrate a bimanual arm (both left and right) using the lerobot calibration system
    """
    
    wait_for_lerobot_preload()
    from lerobot.scripts.lerobot_calibrate import calibrate, CalibrateConfig
    leader_config_class, follower_config_class = get_robot_config_classes(robot_type)
    
    try:
        if leader_config_class is None or follower_config_class is None:
            typer.echo(f"❌ Unsupported robot type: {robot_type}")
            return False

        if arm_type == "leader":
            arm_config = create_bimanual_leader_config(
                leader_config_class,
                left_port,
                right_port,
                robot_type,
                leader_id=arm_id or f"{robot_type}_{arm_type}"
            )
            calibrate_config = CalibrateConfig(teleop=arm_config)
        else:  # follower
            arm_config = create_bimanual_follower_config(
                follower_config_class,
                left_port,
                right_port,
                robot_type,
                camera_config=None,
                follower_id=arm_id or f"{robot_type}_{arm_type}"
            )
            calibrate_config = CalibrateConfig(robot=arm_config)
        
        typer.echo(f"🔧 Calibrating bimanual {arm_type} arms...")
        typer.echo(f"   • Left: {left_port}  • Right: {right_port}")
        typer.echo("⚠️  Follow the instructions. LEFT arm first, then RIGHT.")
        
        calibrate(calibrate_config)
        typer.echo(f"✅ Bimanual {arm_type} arms calibrated successfully!")
        return True
        
    except Exception as e:
        typer.echo(f"❌ Bimanual {arm_type} arms calibration failed: {e}")
        return False


def setup_motors_for_arm(arm_type: str, port: str, robot_type: str = "so100") -> bool:
    """
    Setup motor IDs for a specific arm (leader or follower)
    Returns True if successful, False otherwise
    """
    
    wait_for_lerobot_preload()
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.robots import make_robot_from_config

    try:
        # Determine the appropriate config class based on arm type and robot type
        leader_config_class, follower_config_class = get_robot_config_classes(robot_type)
        
        if leader_config_class is None or follower_config_class is None:
            typer.echo(f"❌ Unsupported robot type for {arm_type}: {robot_type}")
            return False

        if arm_type == "leader":
            if is_starai_robot(robot_type):
                # Star Arm 102 servos ship with fixed bus ids (0-6) and the FashionStar
                # bus offers no id-assignment flow, so there is nothing to set up here.
                typer.echo(
                    "ℹ️  The Star Arm 102 leader has factory-assigned servo ids — no motor setup needed."
                )
                typer.echo("   Run 'solo robo --calibrate leader' next.")
                return True
            config_class = leader_config_class
            make_device = make_teleoperator_from_config
        else:
            config_class = follower_config_class
            make_device = make_robot_from_config
        
        # Create device config and instance
        device_config = config_class(port=port, id=f"{robot_type}_{arm_type}")
        device = make_device(device_config)
        
        typer.echo(f"🔧 Setting up motors for {arm_type} arm — connect each motor when prompted.")
        
        device.setup_motors()
        typer.echo(f"✅ {arm_type.title()} arm motor setup completed!")
        return True
        
    except Exception as e:
        typer.echo(f"❌ {arm_type.title()} arm motor setup failed: {e}")
        return False


def setup_motors_for_bimanual_arm(
    arm_type: str,
    left_port: str,
    right_port: str,
    robot_type: str = "bi_so100"
) -> bool:
    """
    Setup motor IDs for bimanual arm (both left and right)
    Returns True if successful, False otherwise
    """
    
    wait_for_lerobot_preload()
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.robots import make_robot_from_config
    leader_config_class, follower_config_class = get_robot_config_classes(robot_type)
    
    try:
        if leader_config_class is None or follower_config_class is None:
            typer.echo(f"❌ Unsupported robot type: {robot_type}")
            return False

        if arm_type == "leader":
            device_config = create_bimanual_leader_config(
                leader_config_class,
                left_port,
                right_port,
                robot_type,
                leader_id=f"{robot_type}_{arm_type}"
            )
            device = make_teleoperator_from_config(device_config)
        else:
            device_config = create_bimanual_follower_config(
                follower_config_class,
                left_port,
                right_port,
                robot_type,
                camera_config=None,
                follower_id=f"{robot_type}_{arm_type}"
            )
            device = make_robot_from_config(device_config)
        
        typer.echo(f"🔧 Setting up motors for bimanual {arm_type} arms — connect each motor when prompted.")
        typer.echo(f"   • Left: {left_port}  • Right: {right_port}")
        
        device.setup_motors()
        typer.echo(f"✅ Bimanual {arm_type} arms motor setup completed!")
        return True
        
    except Exception as e:
        typer.echo(f"❌ Bimanual {arm_type} arms motor setup failed: {e}")
        return False


def calibration(main_config: dict = None, arm_type: str = None) -> Dict:
    """
    Setup process for arm calibration with selective arm support
    Supports both single-arm and bimanual robots
    Returns configuration dictionary with arm setup details
    """
    # Start loading heavy lerobot libraries in the background while the user
    start_lerobot_preload()
    
    config = {}

    if arm_type is not None and arm_type not in ("leader", "follower", "all"):
        raise ValueError(f"Invalid arm type: {arm_type}, please use 'leader', 'follower', or 'all'")
    
    # Gather any existing config and ask once to reuse
    lerobot_config = main_config.get('lerobot', {}) if main_config else {}
    existing_robot_type = lerobot_config.get('robot_type')
    existing_leader_port = lerobot_config.get('leader_port')
    existing_follower_port = lerobot_config.get('follower_port')
    existing_left_leader_port = lerobot_config.get('left_leader_port')
    existing_right_leader_port = lerobot_config.get('right_leader_port')
    existing_left_follower_port = lerobot_config.get('left_follower_port')
    existing_right_follower_port = lerobot_config.get('right_follower_port')
    
    reuse_all = False
    if existing_robot_type or existing_leader_port or existing_follower_port or existing_left_leader_port:
        typer.echo("\n📦 Found existing configuration:")
        if existing_robot_type:
            typer.echo(f"   • Robot type: {existing_robot_type}")
        # Show ports based on whether bimanual or not
        if is_bimanual_robot(existing_robot_type):
            if existing_left_leader_port:
                typer.echo(f"   • Left leader port: {existing_left_leader_port}")
            if existing_right_leader_port:
                typer.echo(f"   • Right leader port: {existing_right_leader_port}")
            if existing_left_follower_port:
                typer.echo(f"   • Left follower port: {existing_left_follower_port}")
            if existing_right_follower_port:
                typer.echo(f"   • Right follower port: {existing_right_follower_port}")
        else:
            # Only show relevant port(s) based on arm_type
            if arm_type == "leader" and existing_leader_port:
                typer.echo(f"   • Leader port: {existing_leader_port}")
            elif arm_type == "follower" and existing_follower_port:
                typer.echo(f"   • Follower port: {existing_follower_port}")
            else:
                if existing_leader_port:
                    typer.echo(f"   • Leader port: {existing_leader_port}")
                if existing_follower_port:
                    typer.echo(f"   • Follower port: {existing_follower_port}")
        reuse_all = Confirm.ask("Use these settings?", default=True)
    
    if reuse_all and existing_robot_type:
        robot_type = existing_robot_type
    else:
        # Try auto-detection first
        try:
            from solo.commands.robots.lerobot.scan import auto_detect_robot_type
            detected_type, port_info = auto_detect_robot_type(verbose=True)
            
            if detected_type:
                # Ask user to confirm auto-detected type
                typer.echo(f"\n🤖 Auto-detected robot type: {detected_type.upper()}")
                use_detected = Confirm.ask("Use this robot type?", default=True)
                if use_detected:
                    robot_type = detected_type
                else:
                    detected_type = None
            
            if not detected_type:
                # Check if no ports were found at all - warn user
                if not port_info:
                    typer.echo("\n⚠️  No robot arms detected on any serial port.")
                    typer.echo("   This could mean:")
                    typer.echo("   • Robot arm is not connected via USB")
                    typer.echo("   • Robot arm is not powered on")
                    typer.echo("   • USB drivers are not installed")
                    typer.echo("   • Another application is using the port")
                    
                    # Check if this might be a RealMan setup (network-based)
                    typer.echo("\n💡 If you're using a RealMan robot (network-based), you can continue.")
                    if not Confirm.ask("Continue with manual robot selection anyway?", default=False):
                        typer.echo("\n🔌 Please connect your robot arm and try again.")
                        typer.echo("   Run 'solo robo --scan' to check for connected motors.")
                        return {}
                
                # Manual selection
                from solo.commands.robots.lerobot.utils.helper import prompt_robot_type_selection
                robot_type = prompt_robot_type_selection(default="so101")
        except Exception as e:
            typer.echo(f"⚠️  Auto-detection failed: {e}")
            # Fall back to manual selection
            from solo.commands.robots.lerobot.utils.helper import prompt_robot_type_selection
            robot_type = prompt_robot_type_selection(default="so101")
    
    config['robot_type'] = robot_type
    is_bimanual = is_bimanual_robot(robot_type)
    is_realman = is_realman_robot(robot_type)
    
    # Determine which arms to calibrate based on arm_type parameter
    if arm_type == "leader":
        setup_leader = True
        setup_follower = False
    elif arm_type == "follower":
        setup_leader = False
        setup_follower = True
    else:
        # setup both arms
        setup_leader = True
        setup_follower = True
    
    # Handle RealMan robots (SO101 leader + RealMan follower via network)
    if is_realman:
        from solo.commands.robots.lerobot.realman_config import (
            load_realman_config,
            prompt_realman_config,
            test_realman_connection,
        )
        
        typer.echo("\n🤖 RealMan R1D2 Setup")
        typer.echo("   Leader: SO101 (USB) - will be calibrated")
        typer.echo("   Follower: RealMan (network) - no calibration needed")
        
        # Setup SO101 leader arm (USB) - needs calibration
        if setup_leader:
            leader_port = existing_leader_port if reuse_all and existing_leader_port else None
            if not leader_port:
                typer.echo("\n🔍 Detecting SO101 leader arm...")
                leader_port, _ = detect_arm_port("leader", robot_type="so101")
            
            if not leader_port:
                typer.echo("❌ Failed to detect SO101 leader arm. Skipping leader calibration.")
            else:
                config['leader_port'] = leader_port
                from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
                leader_id = prompt_arm_id(main_config or {}, "leader", "so101")
                
                # Calibrate SO101 leader
                if calibrate_arm("leader", leader_port, "so101", leader_id):
                    config['leader_calibrated'] = True
                    config['leader_id'] = leader_id
                    # Add known ID - ensure we have proper main_config
                    target_config = main_config if main_config is not None else {}
                    add_known_id(target_config, 'leader', leader_id, robot_type="so101")
                    if main_config is None:
                        main_config = target_config
                else:
                    config['leader_calibrated'] = False
        
        # Setup RealMan follower (network) - needs calibration for joint mapping
        if setup_follower:
            typer.echo("\n🌐 Configuring RealMan follower (network connection)...")
            
            # Load or prompt for RealMan config
            realman_cfg = load_realman_config()
            if not realman_cfg or not Confirm.ask(f"Use RealMan at {realman_cfg['ip']}:{realman_cfg['port']}?", default=True):
                realman_cfg = prompt_realman_config(realman_cfg)
            
            # Test connection
            if test_realman_connection(realman_cfg):
                # Store RealMan config in main config
                if not config.get('lerobot'):
                    config['lerobot'] = {}
                config['lerobot']['realman_config'] = realman_cfg
                config['realman_config'] = realman_cfg  # Also store at top level for easy access
                
                # Set follower ID
                from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
                follower_id = prompt_arm_id(main_config or {}, "follower", robot_type)
                config['follower_id'] = follower_id
                
                # Add known ID - ensure we have proper main_config
                target_config = main_config if main_config is not None else {}
                add_known_id(target_config, 'follower', follower_id, robot_type=robot_type)
                if main_config is None:
                    main_config = target_config
                
                typer.echo(f"✅ RealMan follower connection test successful: {realman_cfg['model']} at {realman_cfg['ip']}:{realman_cfg['port']}")
                
                # Run calibration to record joint ranges
                if calibrate_realman_follower(realman_cfg, follower_id):
                    config['follower_calibrated'] = True
                else:
                    config['follower_calibrated'] = False
            else:
                typer.echo("❌ Failed to connect to RealMan follower. Please check network settings.")
                config['follower_calibrated'] = False
        
        # Save config - ensure main_config exists
        if main_config is None:
            main_config = {}
        save_lerobot_config(main_config, config)
        return config
    
    if is_bimanual:
        # Bimanual calibration workflow
        if setup_leader:
            # Use existing ports or detect new ones
            left_leader_port = existing_left_leader_port if reuse_all and existing_left_leader_port else None
            right_leader_port = existing_right_leader_port if reuse_all and existing_right_leader_port else None
            
            if not left_leader_port or not right_leader_port:
                left_leader_port, right_leader_port = detect_bimanual_arm_ports("leader")
            
            if not left_leader_port or not right_leader_port:
                typer.echo("❌ Failed to detect bimanual leader arms. Skipping leader calibration.")
            else:
                config['left_leader_port'] = left_leader_port
                config['right_leader_port'] = right_leader_port
                
                # Select leader id
                from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
                leader_id = prompt_arm_id(main_config or {}, "leader", robot_type)
                
                # Calibrate bimanual leader arms
                if calibrate_bimanual_arm("leader", left_leader_port, right_leader_port, robot_type, leader_id):
                    config['leader_calibrated'] = True
                    config['leader_id'] = leader_id
                    add_known_id(main_config or config, 'leader', leader_id, robot_type=robot_type)
                else:
                    config['leader_calibrated'] = False
        
        if setup_follower:
            # Use existing ports or detect new ones
            left_follower_port = existing_left_follower_port if reuse_all and existing_left_follower_port else None
            right_follower_port = existing_right_follower_port if reuse_all and existing_right_follower_port else None
            
            if not left_follower_port or not right_follower_port:
                left_follower_port, right_follower_port = detect_bimanual_arm_ports("follower")
            
            if not left_follower_port or not right_follower_port:
                typer.echo("❌ Failed to detect bimanual follower arms. Skipping follower calibration.")
            else:
                config['left_follower_port'] = left_follower_port
                config['right_follower_port'] = right_follower_port
                
                # Select follower id
                from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
                follower_id = prompt_arm_id(main_config or {}, "follower", robot_type)
                
                # Calibrate bimanual follower arms
                if calibrate_bimanual_arm("follower", left_follower_port, right_follower_port, robot_type, follower_id):
                    config['follower_calibrated'] = True
                    config['follower_id'] = follower_id
                    add_known_id(main_config or config, 'follower', follower_id, robot_type=robot_type)
                else:
                    config['follower_calibrated'] = False
    
    else:
        # Single-arm calibration workflow
        if setup_leader:
            # Use consolidated decision for leader port
            leader_port = existing_leader_port if reuse_all and existing_leader_port else None
            if leader_port and is_starai_robot(robot_type):
                # A reused port may belong to whichever leader was configured last.
                from solo.commands.robots.lerobot.starai_config import resolve_starai_leader_port
                leader_port = resolve_starai_leader_port(leader_port)
            if not leader_port:
                leader_port, detected_type = detect_arm_port("leader", robot_type=robot_type)
                # Update robot_type if auto-detected and not already set
                if detected_type and robot_type is None:
                    robot_type = detected_type
                    config['robot_type'] = robot_type
            
            if not leader_port:
                typer.echo("❌ Failed to detect leader arm. Skipping leader calibration.")
            else:
                config['leader_port'] = leader_port
                # Select leader id
                from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
                leader_id = prompt_arm_id(main_config or {}, "leader", robot_type)
                
                # Calibrate leader arm
                if calibrate_arm("leader", leader_port, robot_type, leader_id):
                    config['leader_calibrated'] = True
                    config['leader_id'] = leader_id
                    add_known_id(main_config or config, 'leader', leader_id, robot_type=robot_type)
                else:
                    config['leader_calibrated'] = False
        
        if setup_follower:
            # Use consolidated decision for follower port
            follower_port = existing_follower_port if reuse_all and existing_follower_port else None
            if not follower_port:
                follower_port, detected_type = detect_arm_port("follower", robot_type=robot_type)
                # Update robot_type if auto-detected and not already set
                if detected_type and robot_type is None:
                    robot_type = detected_type
                    config['robot_type'] = robot_type
            
            if not follower_port:
                typer.echo("❌ Failed to detect follower arm. Skipping follower calibration.")
            else:
                config['follower_port'] = follower_port
                # Select follower id
                from solo.commands.robots.lerobot.utils.helper import prompt_arm_id
                follower_id = prompt_arm_id(main_config or {}, "follower", robot_type)
                
                # Calibrate follower arm
                if calibrate_arm("follower", follower_port, robot_type, follower_id):
                    config['follower_calibrated'] = True
                    config['follower_id'] = follower_id
                    add_known_id(main_config or config, 'follower', follower_id, robot_type=robot_type)
                else:
                    config['follower_calibrated'] = False
    
    return config


def display_calibration_error():
    """Display standard calibration error message."""
    typer.echo("❌ Arms are not properly calibrated.")
    typer.echo("Please run the following commands in order:")
    typer.echo("   • 'solo robo --motors all' - Setup motor IDs for both arms")
    typer.echo("   • 'solo robo --motors leader' - Setup motor IDs for leader arm only")
    typer.echo("   • 'solo robo --motors follower' - Setup motor IDs for follower arm only")
    typer.echo("   • 'solo robo --calibrate all' - Calibrate both arms")
    typer.echo("   • 'solo robo --calibrate leader' - Calibrate leader arm only")
    typer.echo("   • 'solo robo --calibrate follower' - Calibrate follower arm only")


def display_arms_status(robot_type: str, leader_port: str, follower_port: str, arm_type: str = None):
    """Display current arms configuration status."""
    typer.echo("✅ Found calibrated arms:")
    typer.echo(f"   • Robot type: {robot_type.upper()}")
    
    # Only show relevant arm(s) based on arm_type
    if arm_type == "leader":
        if leader_port:
            typer.echo(f"   • Leader arm: {leader_port}")
    elif arm_type == "follower":
        if follower_port:
            typer.echo(f"   • Follower arm: {follower_port}")
    else:
        # Show both arms when arm_type is "all"
        if leader_port:
            typer.echo(f"   • Leader arm: {leader_port}")
        if follower_port:
            typer.echo(f"   • Follower arm: {follower_port}")


def check_calibration_success(arm_config: dict, setup_motors: bool = False) -> None:
    """Check and report calibration success status with appropriate messages."""
    leader_configured = arm_config.get('leader_port') and arm_config.get('leader_calibrated')
    # For RealMan, follower uses network (realman_config) instead of USB port
    follower_configured = (
        (arm_config.get('follower_port') or arm_config.get('realman_config')) 
        and arm_config.get('follower_calibrated')
    )
    
    if leader_configured and follower_configured:
        typer.echo("🎉 All arms calibrated successfully!")
        
        if setup_motors:
            leader_motors = arm_config.get('leader_motors_setup', False)
            follower_motors = arm_config.get('follower_motors_setup', False)
            if leader_motors and follower_motors:
                typer.echo("✅ Motor IDs have been set up for both leader and follower arm.")
            else:
                typer.echo("⚠️  Some motor setups may have failed, but calibration completed.")
        
        typer.echo("🎮 You can now run 'solo robo --teleop' to start teleoperation.")
    elif leader_configured:
        typer.echo("✅ Leader arm calibrated successfully!")
    elif follower_configured:
        typer.echo("✅ Follower arm calibrated successfully!")
    else:
        typer.echo("Run 'solo robo --calibrate all' to retry.")
