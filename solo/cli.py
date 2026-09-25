import typer
from typing import Optional

app = typer.Typer()

# Lazy-loaded commands to improve CLI startup performance

@app.command()
def robo(
    motors: Optional[str] = typer.Option(
        None,
        "--motors",
        help="Setup motor IDs: 'leader', 'follower', or 'all'",
    ),
    calibrate: Optional[str] = typer.Option(
        None,
        "--calibrate",
        help="Calibrate robot arms: 'leader', 'follower', or 'all' (requires motor setup)",
    ),
    teleop: bool = typer.Option(False, "--teleop", help="Start teleoperation (requires calibrated arms)"),
    record: bool = typer.Option(False, "--record", help="Record data for training (requires calibrated arms)"),
    train: bool = typer.Option(False, "--train", help="Train a model (requires recorded data)"),
    inference: bool = typer.Option(False, "--inference", help="Run inference on a pre-trained model"),
    replay: bool = typer.Option(False, "--replay", help="Replay actions from a recorded dataset episode"),
    scan: bool = typer.Option(False, "--scan", help="Scan for connected motors on all serial ports"),
    diagnose: bool = typer.Option(False, "--diagnose", help="Run detailed connection diagnostics on all ports"),
    star_tune: bool = typer.Option(False, "--star-tune", help="Tune the Star Arm 102 leader -> SO101 follower joint mapping"),
    deployx_run: Optional[str] = typer.Option(
        None,
        "--deployx-run",
        help="Run the DeployX edge agent against a policy server, e.g. ws://<pod-host>:8849",
    ),
    deployx_serve: Optional[str] = typer.Option(
        None,
        "--deployx-serve",
        help="Start the DeployX policy server for the given checkpoint (local path, 'org/model' HF repo id, or 'solo:org/model')",
    ),
    deployx_port: Optional[int] = typer.Option(
        None,
        "--deployx-port",
        help="Port for the DeployX policy server (default: protocol.DEFAULT_PORT)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Automatically use saved settings if available"),
    # Replay-specific options (non-interactive)
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Dataset repository ID for replay (e.g., 'organize_fennel_seed')"),
    episode: Optional[str] = typer.Option(
        None,
        "--episode",
        help="Episode(s) to replay: a number ('3'), a comma list ('0,2,5'), a range ('0-10'), a combination ('0-2,5,7-9'), or 'all' (default: 0)",
    ),
    follower_id: Optional[str] = typer.Option(None, "--follower-id", help="Follower arm ID for replay (e.g., 'follower_right')"),
    fps: Optional[int] = typer.Option(None, "--fps", help="Frames per second for replay (default: 30)"),
    save_replay_as: Optional[str] = typer.Option(
        None,
        "--save-replay-as",
        help="Also record the replayed run(s) as new episode(s) in this dataset repo id (requires cameras)",
    ),
    repeat: Optional[int] = typer.Option(
        None,
        "--repeat",
        help="How many times to replay each selected episode (default: 1)",
    ),
    perturb: Optional[float] = typer.Option(
        None,
        "--perturb",
        help="Perturb replayed actions by this fraction of each joint's safe range, "
        "for motion diversity (default: 0, disabled). Safety-validated/clamped like a live policy's output.",
    ),
    perturb_increment: Optional[float] = typer.Option(
        None,
        "--perturb-increment",
        help="Increase --perturb by this amount on each successive repeat (default: 0, constant). "
        "E.g. --perturb 0.02 --repeat 2 --perturb-increment 0.01 -> repeat 1 uses 0.02, repeat 2 uses 0.03.",
    ),
):
    """
    Robotics operations: motor setup, calibration, teleoperation, data recording, training, replay, and inference
    """
    if scan:
        from solo.commands.robots.lerobot.scan import scan_motors
        scan_motors()
        return
    if diagnose:
        from solo.commands.robots.lerobot.scan import diagnose_all_ports
        diagnose_all_ports()
        return
    if star_tune:
        import json, os
        from solo.config import CONFIG_PATH
        from solo.commands.robots.lerobot.starai_tune import tune_starai_map
        saved_config = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH) as f:
                    saved_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                saved_config = {}
        tune_starai_map(saved_config)
        return
    if deployx_serve is not None:
        from solo.commands.robots.lerobot.deployx.policy_server import run_policy_server
        run_policy_server(deployx_serve, port=deployx_port)
        return
    from solo.commands.robo import robo as _robo
    _robo(motors, calibrate, teleop, record, train, inference, replay, yes, dataset, episode, follower_id, fps, deployx_run, save_replay_as, repeat, perturb, perturb_increment)


@app.command()
def setup():
    """
    Set up Solo CLI environment with interactive prompts and saves configuration to config.json.
    """
    from solo.main import setup as _setup
    _setup()


@app.command()
def serve(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="""Model name or path. Can be:
    - HuggingFace repo ID (e.g., 'meta-llama/Llama-3.2-1B-Instruct')
    - Ollama model Registry (e.g., 'llama3.2')
    - Local path to a model file (e.g., '/path/to/model.gguf')
    If not specified, the default model from configuration will be used."""),
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Server type (ollama, vllm, llama.cpp)"), 
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port to run the server on"),
    ui: Optional[bool] = typer.Option(True, "--ui", help="Start the UI for the server")
):
    """Start a model server with the specified model.
    
    If no server is specified, uses the server type from configuration.
    To set up your configuration, run 'solo setup' first.
    """
    from solo.commands.serve import serve as _serve
    _serve(model, server, port, ui)


@app.command()
def status(
    model: Optional[str] = typer.Argument(None, help="Model identifier (org/model_name) to check training status on Solo Hub"),
):
    """Check system status, or check a model's training status on Solo Hub.

    Without arguments: shows running models, system status, and configuration.
    With a model identifier: checks the model's training status on Solo Hub.
    """
    if model:
        from solo.commands.model_status import model_status as _model_status
        _model_status(model)
    else:
        from solo.commands.status import status as _status
        _status()


@app.command(name="list")
def list_models():
    """
    List all downloaded models available in HuggingFace cache and Ollama.
    """
    from solo.commands.models_list import list as _list
    _list()


@app.command()
def test(
    timeout: Optional[int] = typer.Option(None, "--timeout", "-t", help="Request timeout in seconds. Default is 30s for vLLM/Llama.cpp and 120s for Ollama.")
):
    """
    Test if the Solo CLI is running correctly.
    Performs an inference test to verify server functionality.
    """
    from solo.commands.test import test as _test
    _test(timeout)


@app.command()
def stop(name: str = typer.Option("", help="Server type to stop (e.g., 'ollama', 'vllm', 'llama.cpp')")):
    """
    Stops Solo CLI services. If a server type is specified (e.g., 'ollama', 'vllm', 'llama.cpp'),
    only that specific service will be stopped. Otherwise, all Solo services will be stopped.
    """
    from solo.commands.stop import stop as _stop
    _stop(name)


@app.command()
def login(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-authentication even if already logged in"),
):
    """
    Log in to Solo Hub using device-code authentication.
    """
    from solo.commands.login import login as _login
    _login(force=force)


@app.command()
def logout():
    """
    Log out of Solo Hub by removing stored credentials.
    """
    from solo.commands.logout import logout as _logout
    _logout()


@app.command()
def whoami():
    """
    Display your Solo Hub profile, organization, and subscription info.
    """
    from solo.commands.whoami import whoami as _whoami
    _whoami()


@app.command()
def download(
    model: str = typer.Argument(..., help="Model identifier: 'org/model_name' or 'solo:org/model_name'"),
    local_dir: str = typer.Option(None, "--local-dir", "-d", help="Download into a local directory instead of the cache"),
):
    """
    Downloads a model from Solo Hub.

    Accepts both 'org/model_name' and 'solo:org/model_name' formats.
    Requires authentication via 'solo login' first.
    """
    from solo.commands.download import download as _download
    _download(model, local_dir=local_dir)


@app.command(name="setup-usb")
def setup_usb_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt (Linux only)")
):
    """
    Set up USB permissions for LeRobot-compatible robot arms.
    
    On Linux: Installs udev rules and adds user to dialout group.
    On macOS: Checks for connected devices and provides driver info.
    
    Supports Koch (Dynamixel), SO100/SO101 (Feetech/Waveshare) arms.
    Run once after installing solo-cli.
    """
    from solo.commands.setup_usb import setup_usb
    setup_usb(auto_confirm=yes)


if __name__ == "__main__":
    app()
