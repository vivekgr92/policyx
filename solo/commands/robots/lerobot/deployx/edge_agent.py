"""
DeployX edge agent - runs on the Mac mini next to the SO-101 arm.

Control loop: observe -> send `predict` -> receive `action` -> validate against
safety limits -> execute -> capture the resulting observation -> log to
telemetry -> repeat. The wire format is defined in protocol.py (shared with
the Runpod-side policy server, built separately) and must not be reinterpreted
here.

Hardware setup reuses the same building blocks as modes/inference.py and
utils/record_config.py: `create_follower_config`/`get_robot_config_classes`
from config.py to build a robot config object, then lerobot's own
`make_robot_from_config` to instantiate and connect it. No new hardware I/O is
implemented in this file - it stays focused on orchestrating the loop, with
safety checks living in safety.py and logging in telemetry.py.
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.prompt import Confirm, Prompt

from solo.commands.robots.lerobot.deployx import protocol
from solo.commands.robots.lerobot.deployx.safety import (
    EmergencyStop,
    EmergencyStopTriggered,
    JointLimits,
    Watchdog,
    load_joint_limits,
    safe_shutdown,
    validate_action,
)
from solo.commands.robots.lerobot.deployx.telemetry import (
    StepRecord,
    TelemetryLogger,
    encode_jpeg,
    push_session_to_hub,
)


class EdgeAgentError(Exception):
    """Raised for protocol/connection problems the control loop can't recover from."""


def _build_follower_robot(
    robot_type: str,
    follower_port: Optional[str],
    follower_id: str,
    camera_config: Dict,
    realman_config: Optional[Dict] = None,
):
    """
    Construct and connect the follower robot object.

    Reuses config.py's own builders (the same ones record_config.py and
    calibration.py use) rather than a separate construction path, so DeployX
    stays in sync with however those config classes evolve.
    """
    from lerobot.robots import make_robot_from_config
    from solo.commands.robots.lerobot.config import (
        create_follower_config,
        get_robot_config_classes,
        is_bimanual_robot,
        is_realman_robot,
    )

    if is_bimanual_robot(robot_type):
        raise EdgeAgentError(
            "DeployX does not support bimanual robots yet (protocol.py's observation/action "
            "shape is single-arm)."
        )

    if is_realman_robot(robot_type):
        from solo.commands.robots.lerobot.realman_config import create_realman_follower_config

        follower_config = create_realman_follower_config(
            realman_config or {}, camera_config, follower_id=follower_id
        )
    else:
        _, follower_config_class = get_robot_config_classes(robot_type)
        if follower_config_class is None:
            raise EdgeAgentError(f"Unsupported robot type: {robot_type}")
        follower_config = create_follower_config(
            follower_config_class, follower_port, robot_type, camera_config, follower_id=follower_id
        )

    robot = make_robot_from_config(follower_config)
    robot.connect(calibrate=True)
    return robot


def _split_observation(obs: dict, limits: JointLimits) -> tuple[list[float], dict]:
    """Split a robot.get_observation() dict into (ordered joint state, camera frames).

    Joint order follows `limits.names` (the calibration file's own order) so
    the wire payload and the safety limits always index the same joint the
    same way, regardless of dict iteration order.
    """
    positions = {key[: -len(".pos")]: float(value) for key, value in obs.items() if key.endswith(".pos")}
    missing = [name for name in limits.names if name not in positions]
    if missing:
        raise EdgeAgentError(f"Observation missing calibrated joint(s): {missing}")
    state = [positions[name] for name in limits.names]
    frames = {key: value for key, value in obs.items() if not key.endswith(".pos")}
    return state, frames


def _encode_frames_b64(frames: dict) -> dict:
    return {name: base64.b64encode(encode_jpeg(frame)).decode("ascii") for name, frame in frames.items()}


async def _send_json(ws, message: dict) -> None:
    await ws.send(json.dumps(message))


async def _recv_json(ws, timeout: float) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(raw)


def _log_openwam_result(telemetry: TelemetryLogger, step: int, request_id: str, task: "asyncio.Task") -> None:
    """Done-callback for a background OpenWAM prediction task - runs on the
    event loop once the task finishes, decoupled from the control loop's own
    timing. Never raises: a failed/timed-out OpenWAM call is logged, not fatal."""
    if task.cancelled():
        return
    exc = task.exception()
    prediction = {"error": str(exc)} if exc else task.result()
    telemetry.log_step(
        StepRecord(
            step=step,
            request_id=request_id,
            timestamp=time.time(),
            openwam_prediction=prediction,
        )
    )


async def run_edge_agent(
    server_url: str,
    robot_type: str,
    follower_port: Optional[str],
    follower_id: str,
    camera_config: Dict,
    task_description: str = "",
    fps: int = 30,
    session_time_s: Optional[float] = None,
    realman_config: Optional[Dict] = None,
    push_repo_id: Optional[str] = None,
    openwam_server_url: Optional[str] = None,
) -> None:
    """Connect, run the observe/predict/validate/execute/log loop, and always
    safe-stop + flush telemetry on the way out (normal end, error, or e-stop).

    When openwam_server_url is set, each step's action chunk is ALSO fired at
    an OpenWAM instance (openwam_bridge.py) via asyncio.create_task() - a
    fire-and-forget background task, never awaited here, so it cannot add
    latency to the observe/validate/execute hot path or interfere with
    protocol.PREDICT_TIMEOUT_S/the watchdog. Its result lands in telemetry
    whenever it completes, via _log_openwam_result. Default (None) is a no-op:
    zero extra work, identical behavior to before this parameter existed."""
    import websockets

    estop = EmergencyStop()
    estop.arm()

    robot = None
    telemetry = TelemetryLogger()
    watchdog = Watchdog(timeout_s=protocol.PREDICT_TIMEOUT_S)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30
    step = 0
    openwam_tasks: set = set()

    typer.echo(f"📝 Telemetry session: {telemetry.session_dir}")

    try:
        limits = load_joint_limits(robot_type, follower_id)
        typer.echo(f"✅ Loaded safety limits for {len(limits)} joints from calibration")

        robot = _build_follower_robot(robot_type, follower_port, follower_id, camera_config, realman_config)
        typer.echo(f"✅ Connected to {robot_type} follower '{follower_id}'")

        typer.echo(f"🔌 Connecting to DeployX policy server at {server_url}")
        async with websockets.connect(server_url, max_size=None) as ws:
            await _send_json(ws, {"type": protocol.MSG_PING})
            pong = await _recv_json(ws, protocol.PREDICT_TIMEOUT_S)
            if pong.get("type") != protocol.MSG_PONG or not pong.get("ready", False):
                raise EdgeAgentError(f"Policy server not ready: {pong}")
            watchdog.reset()
            typer.echo("✅ Policy server ready")

            await _send_json(ws, {"type": protocol.MSG_RESET})
            reset_ack = await _recv_json(ws, protocol.PREDICT_TIMEOUT_S)
            if reset_ack.get("type") != protocol.MSG_RESET_ACK:
                raise EdgeAgentError(f"Policy server reset failed: {reset_ack}")
            watchdog.reset()

            start_time = time.monotonic()
            last_activity = time.monotonic()

            while True:
                estop.check()

                if session_time_s and (time.monotonic() - start_time) >= session_time_s:
                    typer.echo("⏱️  Session duration elapsed, ending normally.")
                    break

                if watchdog.is_timed_out():
                    raise EdgeAgentError(f"Watchdog timeout: no server response in over {watchdog.timeout_s}s")

                loop_start = time.monotonic()

                obs = robot.get_observation()
                current_state, frames = _split_observation(obs, limits)
                image_paths = {
                    name: telemetry.save_frame(step, name, encode_jpeg(frame)) for name, frame in frames.items()
                }
                request_id = str(step)

                request: protocol.PredictRequest = {
                    "type": protocol.MSG_PREDICT,
                    "observation": {
                        "images": _encode_frames_b64(frames),
                        "state": current_state,
                        "task": task_description,
                    },
                    "request_id": request_id,
                }

                error_text: Optional[str] = None
                is_safe = True
                safety_reason: Optional[str] = None
                commanded_action: Optional[list[float]] = None
                resulting_state: Optional[list[float]] = None
                success = False
                latency_ms = None

                try:
                    await _send_json(ws, request)
                    response = await _recv_json(ws, protocol.PREDICT_TIMEOUT_S)
                    watchdog.reset()
                    last_activity = time.monotonic()

                    if response.get("type") != protocol.MSG_ACTION or response.get("request_id") != request_id:
                        raise EdgeAgentError(f"Unexpected/mismatched response: {response}")
                    if response.get("error"):
                        raise EdgeAgentError(f"Policy server error: {response['error']}")

                    action_chunk = response.get("action_chunk") or []
                    latency_ms = response.get("latency_ms")
                    if not action_chunk:
                        raise EdgeAgentError("Empty action_chunk from policy server")

                    if openwam_server_url:
                        from solo.commands.robots.lerobot.deployx.openwam_bridge import (
                            predict_observation_openwam,
                        )

                        owam_task = asyncio.create_task(
                            predict_observation_openwam(
                                server_url=openwam_server_url,
                                images=request["observation"]["images"],
                                action_chunk=action_chunk,
                                request_id=request_id,
                                task=task_description,
                            )
                        )
                        def _on_owam_done(t, s=step, rid=request_id):
                            openwam_tasks.discard(t)
                            _log_openwam_result(telemetry, s, rid, t)

                        openwam_tasks.add(owam_task)
                        owam_task.add_done_callback(_on_owam_done)

                    step_state = current_state
                    for sub_action in action_chunk:
                        estop.check()
                        ok, clamped, reason = validate_action(sub_action, step_state, dt, limits)
                        if reason:
                            safety_reason = reason
                        if not ok:
                            is_safe = False
                            raise EmergencyStopTriggered(f"Unsafe action rejected: {reason}")

                        action_dict = {f"{name}.pos": clamped[i] for i, name in enumerate(limits.names)}
                        sent = robot.send_action(action_dict)
                        step_state = [float(sent[f"{name}.pos"]) for name in limits.names]
                        await asyncio.sleep(0)  # yield to the event loop between sub-steps

                    commanded_action = step_state
                    resulting_obs = robot.get_observation()
                    resulting_state, _ = _split_observation(resulting_obs, limits)
                    success = True

                except EmergencyStopTriggered:
                    raise
                except (asyncio.TimeoutError, TimeoutError, ConnectionError) as e:
                    error_text = f"Connection error: {e}"
                except EdgeAgentError as e:
                    error_text = str(e)
                except Exception as e:  # noqa: BLE001 - log and stop, don't crash silently
                    error_text = f"Unexpected error: {e}"

                telemetry.log_step(
                    StepRecord(
                        step=step,
                        request_id=request_id,
                        timestamp=time.time(),
                        task=task_description,
                        observation_state=current_state,
                        image_paths=image_paths,
                        commanded_action=commanded_action,
                        is_safe=is_safe,
                        safety_reason=safety_reason,
                        resulting_state=resulting_state,
                        latency_ms=latency_ms,
                        success=success,
                        error=error_text,
                    )
                )

                if error_text:
                    raise EdgeAgentError(error_text)

                step += 1

                elapsed = time.monotonic() - loop_start
                if time.monotonic() - last_activity >= protocol.HEARTBEAT_INTERVAL_S:
                    await _send_json(ws, {"type": protocol.MSG_PING})
                    await _recv_json(ws, protocol.PREDICT_TIMEOUT_S)
                    watchdog.reset()
                    last_activity = time.monotonic()

                sleep_time = dt - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

    except EmergencyStopTriggered as e:
        typer.echo(f"🛑 {e}")
    except Exception as e:  # noqa: BLE001 - top-level session guard
        typer.echo(f"❌ DeployX session ended with error: {e}")
    finally:
        for t in openwam_tasks:
            if not t.done():
                t.cancel()  # background/out-of-band, never worth blocking session end on
        estop.disarm()
        safe_shutdown(robot, reason="session end")
        telemetry.close()
        typer.echo(f"📝 Telemetry saved to {telemetry.session_dir}")
        if push_repo_id:
            push_session_to_hub(telemetry.session_dir, push_repo_id)


def deployx_run_mode(config: dict, server_url: str, auto_use: bool = False) -> None:
    """
    CLI entry point for `solo robo --deployx-run <ws_url>`.

    Prompts for whatever robot/camera config isn't already saved, mirroring
    modes/inference.py's preconfigured-vs-fresh handling, then runs the async
    control loop.
    """
    from solo.commands.robots.lerobot.config import (
        is_bimanual_robot,
        is_realman_robot,
        validate_lerobot_config,
    )
    from solo.commands.robots.lerobot.cameras import setup_cameras
    from solo.commands.robots.lerobot.mode_config import save_deployx_config, use_preconfigured_args
    from solo.commands.robots.lerobot.utils.helper import auto_detect_robot, port_detection, prompt_arm_id

    preconfigured, detected_robot_type = use_preconfigured_args(config, "deployx", "DeployX", auto_use=auto_use)

    realman_config = None

    if preconfigured:
        robot_type = preconfigured.get("robot_type")
        follower_port = preconfigured.get("follower_port")
        follower_id = preconfigured.get("follower_id")
        camera_config = preconfigured.get("camera_config") or {}
        task_description = preconfigured.get("task_description") or ""
        fps = preconfigured.get("fps") or 30
        session_time = preconfigured.get("session_time")
        push_repo_id = preconfigured.get("push_repo_id")
        openwam_server_url = preconfigured.get("openwam_server_url")

        if is_realman_robot(robot_type):
            realman_config = config.get("realman_config") or config.get("lerobot", {}).get("realman_config")

        if not (robot_type and follower_id and (follower_port or is_realman_robot(robot_type))):
            typer.echo("❌ Preconfigured DeployX settings missing required configuration")
            preconfigured = None

    if not preconfigured:
        _, follower_port, _, follower_calibrated, saved_robot_type = validate_lerobot_config(config)
        robot_type = detected_robot_type if detected_robot_type else saved_robot_type

        if not robot_type:
            robot_type = auto_detect_robot(default="so101")
            config["robot_type"] = robot_type

        if is_bimanual_robot(robot_type):
            typer.echo("❌ DeployX does not support bimanual robots yet.")
            return

        if is_realman_robot(robot_type):
            from solo.commands.robots.lerobot.utils.helper import get_realman_configs

            realman_config = get_realman_configs(config)
            config["realman_config"] = realman_config
            follower_port = None
            typer.echo("✅ Found RealMan follower (network):")
            typer.echo(f"   • {realman_config.get('ip')}:{realman_config.get('port')}")
        else:
            follower_port = port_detection(config, "follower", robot_type, follower_port)
            if not follower_port:
                typer.echo("❌ Could not detect follower arm port.")
                return
            typer.echo(f"✅ Found follower arm on {follower_port}")

        follower_id = prompt_arm_id(config, "follower", robot_type)

        if not follower_calibrated:
            typer.echo(
                "⚠️  Follower arm is not marked as calibrated in the saved config. "
                "DeployX needs a calibration file to load safety limits."
            )
            if not Confirm.ask("Continue anyway?", default=False):
                return

        camera_config = setup_cameras()

        task_description = Prompt.ask("Enter task description", default="")
        fps = int(Prompt.ask("Control loop FPS", default="30"))
        duration_str = Prompt.ask(
            "Session duration in seconds (0 = run until stopped with Ctrl+C)", default="0"
        )
        session_time = float(duration_str) if float(duration_str) > 0 else None

        push_repo_id = None
        if Confirm.ask("Push telemetry session to HuggingFace Hub when it ends?", default=False):
            push_repo_id = Prompt.ask("Dataset repo ID for telemetry (e.g. username/deployx-session)")

        openwam_server_url = Prompt.ask(
            "OpenWAM server URL for observation-prediction comparison (leave blank to skip)",
            default="",
        ) or None

        save_deployx_config(
            config,
            {
                "robot_type": robot_type,
                "follower_port": follower_port,
                "follower_id": follower_id,
                "camera_config": camera_config,
                "server_url": server_url,
                "task_description": task_description,
                "fps": fps,
                "session_time": session_time,
                "push_repo_id": push_repo_id,
                "openwam_server_url": openwam_server_url,
            },
        )

    typer.echo("\n🚀 Starting DeployX Edge Agent")
    typer.echo(f"   • Server: {server_url}")
    typer.echo(f"   • Robot: {robot_type.upper()} ({follower_id})")
    typer.echo(f"   • Task: {task_description or 'Not specified'}")
    typer.echo(f"   • FPS: {fps}")
    typer.echo(f"   • Duration: {session_time if session_time else 'until stopped'}")
    if openwam_server_url:
        typer.echo(f"   • OpenWAM prediction: {openwam_server_url}")
    typer.echo("💡 Press Ctrl+C to trigger an emergency stop at any time.\n")

    try:
        asyncio.run(
            run_edge_agent(
                server_url=server_url,
                robot_type=robot_type,
                follower_port=follower_port,
                follower_id=follower_id,
                camera_config=camera_config,
                task_description=task_description,
                fps=fps,
                session_time_s=session_time,
                realman_config=realman_config,
                push_repo_id=push_repo_id,
                openwam_server_url=openwam_server_url,
            )
        )
    except KeyboardInterrupt:
        # Second Ctrl+C (after the loop's own EmergencyStop handler already
        # unwound once) - nothing left to clean up beyond what run_edge_agent's
        # own finally block already did.
        typer.echo("\n🛑 DeployX stopped by user.")
