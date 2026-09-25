"""
Remote training on a Runpod GPU pod.

This talks to the Runpod REST API (https://rest.runpod.io/v1) directly with a
user API key - it is independent of the Claude Code Runpod plugin/MCP, since
`solo robo --train` runs standalone outside of any Claude Code session.
"""

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.prompt import Confirm, Prompt

from solo.config import CONFIG_PATH

API_BASE = "https://rest.runpod.io/v1"
# Separate host/version from the pod-CRUD REST v1 API above - the GPU catalog
# (availability + pricing) only exists under this v2 host, confirmed via
# Runpod's own docs (rest.runpod.io has no catalog endpoint at all).
CATALOG_API_BASE = "https://api.runpod.io/v2"
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_GPU_TYPE_ID = "NVIDIA GeForce RTX 4090"
ALL_POLICIES = ["smolvla", "act", "pi0", "tdmpc", "diffusion", "vqbet", "pi0_fast", "pi05"]


# ---------------------------------------------------------------------------
# Config / credentials
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}


def _save_config(config: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def get_api_key() -> str:
    """Get the Runpod API key from env, saved config, or prompt for it once."""
    env_key = os.environ.get("RUNPOD_API_KEY")
    if env_key:
        return env_key

    config = _load_config()
    key = config.get("runpod", {}).get("api_key")
    if key:
        return key

    typer.echo("\n🔑 A Runpod API key is required to manage pods from solo.")
    typer.echo("   Create one at https://www.runpod.io/console/user/settings (API Keys tab)")
    key = Prompt.ask("Enter your Runpod API key")
    config.setdefault("runpod", {})["api_key"] = key
    _save_config(config)
    return key


def _local_hf_token() -> Optional[str]:
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def _local_wandb_key() -> Optional[str]:
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth:
            return auth[2]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Runpod REST client
# ---------------------------------------------------------------------------

class RunpodClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {get_api_key()}",
            "Content-Type": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs):
        resp = self.session.request(method, f"{API_BASE}{path}", timeout=30, **kwargs)
        if resp.status_code == 401:
            raise RuntimeError("Runpod rejected the API key (401). Check it and try again.")
        if not resp.ok:
            # requests' default raise_for_status() message doesn't include the response
            # body, which is where Runpod actually explains what went wrong (e.g. a
            # capacity error) - surface it so failures are diagnosable from the CLI.
            try:
                parsed = resp.json()
                detail = parsed.get("detail", resp.text) if isinstance(parsed, dict) else parsed
            except ValueError:
                detail = resp.text
            error = requests.HTTPError(f"{resp.status_code} {method} {path}: {detail}", response=resp)
            raise error
        return resp.json() if resp.text else None

    def list_pods(self) -> list:
        return self._request("GET", "/pods") or []

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}")

    def start_pod(self, pod_id: str) -> dict:
        return self._request("POST", f"/pods/{pod_id}/start")

    def stop_pod(self, pod_id: str) -> dict:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def update_pod(self, pod_id: str, **fields) -> dict:
        return self._request("PATCH", f"/pods/{pod_id}", data=json.dumps(fields))

    def create_pod(self, name: str, public_key: str, gpu_type_ids: list, cloud_type: str = "SECURE") -> dict:
        body = {
            "name": name,
            "imageName": DEFAULT_IMAGE,
            "gpuTypeIds": gpu_type_ids,
            "gpuCount": 1,
            "cloudType": cloud_type,
            "containerDiskInGb": 30,
            "volumeInGb": 30,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp", "8888/http"],
            "env": {"PUBLIC_KEY": public_key},
        }
        return self._request("POST", "/pods", data=json.dumps(body))

    def list_gpu_types(self, cloud_type: str = "SECURE") -> list:
        """GPU catalog with live availability + pricing for the given cloud tier.

        Lives on a different host/API version than pod CRUD (see
        CATALOG_API_BASE) - reuses this client's session/auth but bypasses
        `_request()`'s API_BASE prefix.
        """
        resp = self.session.get(
            f"{CATALOG_API_BASE}/catalog/gpus",
            params={"include": "AVAILABILITY", "product": "POD", "cloud": cloud_type},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("gpus", [])


# ---------------------------------------------------------------------------
# SSH key management
# ---------------------------------------------------------------------------

def ensure_local_ssh_key() -> tuple[Path, str]:
    """Ensure a local SSH keypair exists for connecting to Runpod pods."""
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    key_path = ssh_dir / "id_ed25519_runpod"
    pub_path = key_path.with_suffix(".pub")

    if not key_path.exists():
        typer.echo("🔑 Generating a new SSH keypair for Runpod pod access...")
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", "solo-cli-runpod"],
            check=True,
            capture_output=True,
        )
    return key_path, pub_path.read_text().strip()


# ---------------------------------------------------------------------------
# Pod selection
# ---------------------------------------------------------------------------

def _select_cloud_and_gpu(client: RunpodClient) -> tuple[str, str]:
    """Prompt for Secure vs Community cloud, then show real available GPUs
    (live stock + $/hr for that tier) to pick from, instead of a blind
    free-text GPU id. Falls back to manual entry if the catalog call fails or
    the user wants a GPU not in the (possibly filtered/stale) listing.
    """
    cloud_type = Prompt.ask(
        "Cloud type - Secure (Runpod-owned datacenters, more reliable) or "
        "Community (community-hosted, often cheaper but less consistent)",
        choices=["secure", "community"],
        default="community",
    ).upper()

    try:
        gpus = client.list_gpu_types(cloud_type=cloud_type)
    except requests.RequestException as e:
        typer.echo(f"⚠️  Could not fetch live GPU catalog ({e}) - falling back to manual entry.")
        gpu_type_id = Prompt.ask(
            "GPU type ID (find options at https://www.runpod.io/console/pods)",
            default=DEFAULT_GPU_TYPE_ID,
        )
        return cloud_type, gpu_type_id

    price_key = cloud_type.lower()
    available = [g for g in gpus if g.get("availability") not in (None, "NONE")]
    available.sort(key=lambda g: (g.get("price", {}).get(price_key) is None, g.get("price", {}).get(price_key, 0)))

    if not available:
        typer.echo(f"⚠️  No GPUs currently show availability on {cloud_type} cloud - falling back to manual entry.")
        gpu_type_id = Prompt.ask(
            "GPU type ID (find options at https://www.runpod.io/console/pods)",
            default=DEFAULT_GPU_TYPE_ID,
        )
        return cloud_type, gpu_type_id

    typer.echo(f"\n📋 Available GPUs on {cloud_type} cloud (live stock + pricing):")
    for i, g in enumerate(available, 1):
        price = g.get("price", {}).get(price_key)
        price_str = f"${price:.2f}/hr" if price is not None else "price n/a"
        typer.echo(
            f"   {i}. {g['id']}  ({g.get('memory', '?')}GB)  {price_str}  "
            f"[{g.get('availability', '?')} availability]"
        )
    typer.echo(f"   {len(available) + 1}. Enter a GPU type ID manually")

    choice = Prompt.ask(
        "Select a GPU",
        choices=[str(i) for i in range(1, len(available) + 2)],
        default="1",
    )
    idx = int(choice) - 1
    if idx < len(available):
        gpu_type_id = available[idx]["id"]
    else:
        gpu_type_id = Prompt.ask(
            "GPU type ID (find options at https://www.runpod.io/console/pods)",
            default=DEFAULT_GPU_TYPE_ID,
        )
    return cloud_type, gpu_type_id


def _create_pod_interactive(client: RunpodClient, public_key: str, max_attempts: int = 3) -> dict:
    name = Prompt.ask("Name for the new pod", default="solo-train")
    cloud_type, gpu_type_id = _select_cloud_and_gpu(client)
    typer.echo(
        f"💰 About to create a pod with 1x {gpu_type_id} on {cloud_type} cloud. Check current $/hr pricing at "
        "https://www.runpod.io/console/pods before confirming - this will start billing immediately."
    )
    if not Confirm.ask("Proceed with pod creation?", default=True):
        raise typer.Abort()

    for attempt in range(1, max_attempts + 1):
        try:
            pod = client.create_pod(name=name, public_key=public_key, gpu_type_ids=[gpu_type_id], cloud_type=cloud_type)
            typer.echo(f"✅ Created pod {pod['id']}")
            return pod
        except requests.HTTPError as e:
            if not _is_capacity_error(e) or attempt == max_attempts:
                raise
            # The scheduler auto-picks a host per attempt, so simply retrying
            # (without pinning a data center) can land somewhere with free capacity.
            typer.echo(f"⚠️  No capacity on the host the scheduler picked (attempt {attempt}/{max_attempts}). Retrying...")
    raise RuntimeError("Could not create a pod after multiple attempts - all attempted hosts were out of capacity.")


def select_or_create_pod(client: RunpodClient, public_key: str) -> dict:
    pods = client.list_pods()
    if pods:
        typer.echo("\n📦 Existing Runpod pods:")
        for i, pod in enumerate(pods, 1):
            gpu = (pod.get("gpu") or {}).get("id", "?")
            typer.echo(f"   {i}. {pod['name']}  ({gpu})  [{pod.get('desiredStatus', '?')}]  id={pod['id']}")
        typer.echo(f"   {len(pods) + 1}. Create a new pod")

        choice = Prompt.ask(
            "Select a pod",
            choices=[str(i) for i in range(1, len(pods) + 2)],
            default="1",
        )
        idx = int(choice) - 1
        pod = pods[idx] if idx < len(pods) else _create_pod_interactive(client, public_key)
    else:
        typer.echo("\n📦 No existing Runpod pods found.")
        pod = _create_pod_interactive(client, public_key)

    pod_id = pod["id"]

    # Make sure our SSH key is authorized on the pod (older pods may predate this key,
    # and there is no REST endpoint to register an account-level key - PUBLIC_KEY env
    # var is what Runpod's pod images use to seed authorized_keys on boot).
    current_env = pod.get("env") or {}
    if current_env.get("PUBLIC_KEY") != public_key:
        typer.echo("🔧 Updating pod's authorized SSH key...")
        client.update_pod(pod_id, env={**current_env, "PUBLIC_KEY": public_key})

    if pod.get("desiredStatus") != "RUNNING":
        typer.echo(f"▶️  Starting pod {pod_id}...")
        try:
            client.start_pod(pod_id)
        except requests.HTTPError as e:
            if not _is_capacity_error(e):
                raise
            typer.echo(f"⚠️  Pod {pod_id} could not start - its host is out of GPU capacity right now.")
            if Confirm.ask("Create a new pod on a different host instead?", default=True):
                pod = _create_pod_interactive(client, public_key)
                pod_id = pod["id"]
            else:
                raise RuntimeError(f"Pod {pod_id}'s host has no free GPU capacity. Try again later or pick another pod.")

    typer.echo("⏳ Waiting for pod to come online...")
    return _wait_until_ready(client, pod_id)


def _is_capacity_error(exc: "requests.HTTPError") -> bool:
    resp = exc.response
    if resp is None or resp.status_code not in (400, 500):
        return False
    text = resp.text.lower()
    return any(phrase in text for phrase in (
        "not enough free gpu", "does not have the resources", "no free", "not enough resources",
    ))


def _wait_until_ready(client: RunpodClient, pod_id: str, timeout: int = 600) -> dict:
    # Direct SSH (public IP + port mapping) can take several minutes to appear
    # after a pod reports RUNNING - observed ~6 minutes on Secure Cloud.
    elapsed = 0
    last_status = None
    while elapsed < timeout:
        try:
            pod = client.get_pod(pod_id)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise RuntimeError(
                    f"Pod {pod_id} no longer exists (it may have been stopped/terminated by "
                    "someone or something else while this command was waiting on it). "
                    "Please re-run to create a fresh pod."
                ) from e
            raise
        status = pod.get("desiredStatus")
        if status != last_status:
            typer.echo(f"   ... pod status: {status}")
            last_status = status
        if status == "RUNNING" and (pod.get("portMappings") or {}).get("22"):
            return pod
        time.sleep(10)
        elapsed += 10
    raise TimeoutError(
        f"Pod {pod_id} did not get a direct SSH port mapping within {timeout}s. "
        "Note: Runpod's SSH proxy (ssh.runpod.io) is not used here because it forces an "
        "interactive shell and ignores one-shot commands - only direct IP:port SSH works "
        "for scripted execution."
    )


# ---------------------------------------------------------------------------
# SSH / rsync operations
# ---------------------------------------------------------------------------

def _ssh_conn(pod: dict) -> tuple[str, int]:
    return pod["publicIp"], pod["portMappings"]["22"]


def wait_for_ssh(pod: dict, key_path: Path, timeout: int = 180) -> None:
    host, port = _ssh_conn(pod)
    elapsed = 0
    while elapsed < timeout:
        result = subprocess.run(
            ["ssh", "-i", str(key_path), "-p", str(port),
             "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=5",
             f"root@{host}", "echo ready"],
            capture_output=True,
        )
        if result.returncode == 0:
            return
        time.sleep(5)
        elapsed += 5
    raise TimeoutError(
        "Could not SSH into the pod. Make sure your public key is registered with your "
        "Runpod account at https://www.runpod.io/console/user/settings"
    )


def run_remote(pod: dict, key_path: Path, command: str) -> int:
    """Run a command on the pod, streaming its output live to this terminal."""
    host, port = _ssh_conn(pod)
    args = [
        "ssh", "-tt", "-i", str(key_path), "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new", f"root@{host}", command,
    ]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        typer.echo(line.rstrip())
    return process.wait()


def run_remote_to_file(pod: dict, key_path: Path, command: str, log_path: Path) -> int:
    """
    Run a command on the pod, writing its output to a local log file instead of
    streaming to the terminal. Used for concurrent multi-pod runs, where several
    pods streaming full training logs live at once would interleave into noise.
    """
    host, port = _ssh_conn(pod)
    args = [
        "ssh", "-tt", "-i", str(key_path), "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new", f"root@{host}", command,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        process = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
        return process.wait()


def rsync_to_pod(pod: dict, key_path: Path, local_path: str, remote_path: str) -> None:
    host, port = _ssh_conn(pod)
    ssh_cmd = f"ssh -i {key_path} -p {port} -o StrictHostKeyChecking=accept-new"
    subprocess.run(
        ["rsync", "-avz", "-e", ssh_cmd, f"{local_path}/", f"root@{host}:{remote_path}/"],
        check=True,
    )


def rsync_from_pod(pod: dict, key_path: Path, remote_path: str, local_path: str) -> None:
    host, port = _ssh_conn(pod)
    ssh_cmd = f"ssh -i {key_path} -p {port} -o StrictHostKeyChecking=accept-new"
    os.makedirs(local_path, exist_ok=True)
    subprocess.run(
        ["rsync", "-avz", "-e", ssh_cmd, f"root@{host}:{remote_path}/", f"{local_path}/"],
        check=True,
    )


# ---------------------------------------------------------------------------
# Bootstrap + dataset sync
# ---------------------------------------------------------------------------

SOLO_REPO_URL = "https://github.com/vivekgr92/policyx.git"


# pi0 / pi0_fast / pi05 all build on PaliGemma-family modeling code that requires a
# patched transformers fork (lerobot's own "pi" extra) - plain PyPI transformers is
# missing the module their __init__ checks for and fails with a cryptic ValueError
# deep inside lerobot. Installed unconditionally for these policies at bootstrap time
# since nothing in the remote training path calls solo-cli's own preflight/dependency
# check (that only runs on the local inference path), so there's no other point where
# this would get caught or auto-installed before training crashes on it.
_TRANSFORMERS_PI_PATCH_SPEC = (
    "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
)
_POLICIES_NEEDING_TRANSFORMERS_PI_PATCH = {"pi0", "pi0_fast", "pi05"}


def bootstrap_pod(
    pod: dict,
    key_path: Path,
    log_path: Optional[Path] = None,
    policy_name: Optional[str] = None,
) -> None:
    """
    Install solo-cli + lerobot on the pod if not already present.

    Installs from the solo-cli git remote (`SOLO_REPO_URL`), i.e. whatever is
    currently pushed to origin's default branch - push local changes before
    training remotely, or the pod will run stale code.

    When `log_path` is given, install output goes to that file instead of the
    terminal (for concurrent multi-pod runs). `policy_name`, when one of
    pi0/pi0_fast/pi05, triggers installing the patched transformers fork those
    policies require (see `_TRANSFORMERS_PI_PATCH_SPEC`).
    """
    typer.echo("🔍 Checking remote environment...")
    already_installed = run_remote(pod, key_path, "which solo") == 0
    if already_installed and policy_name not in _POLICIES_NEEDING_TRANSFORMERS_PI_PATCH:
        typer.echo("✅ solo-cli already installed on pod.")
        return
    if already_installed:
        typer.echo("✅ solo-cli already installed on pod.")
        _install_transformers_pi_patch(pod, key_path, log_path)
        return

    typer.echo("📦 Installing solo-cli + lerobot on pod (first run only, this can take a few minutes)...")
    install_cmd = (
        # uv resolves lerobot's large dependency tree dramatically faster than plain
        # pip (minutes vs 20+ minutes observed with pip on this exact install). The
        # Runpod pytorch image ships uv already; the `command -v` fallback installs
        # it via pip if a future/different image doesn't. `--system` targets the
        # container's system Python directly (no venv); `--break-system-packages` is
        # ALSO required separately (uv has its own flag mirroring pip's - `--system`
        # alone does not bypass Ubuntu 24.04's PEP668 externally-managed-environment
        # marker, confirmed by a real bootstrap failure on a live pod) - fine to
        # override here since this is a disposable training container, not a shared
        # system.
        "(command -v uv >/dev/null 2>&1 || pip install -q uv) && "
        f"uv pip install --system --break-system-packages -q 'solo-cli @ git+{SOLO_REPO_URL}'"
    )
    rc = (
        run_remote_to_file(pod, key_path, install_cmd, log_path)
        if log_path
        else run_remote(pod, key_path, install_cmd)
    )
    if rc != 0:
        raise RuntimeError(
            f"Failed to bootstrap solo-cli on the pod."
            + (f" See {log_path} for details." if log_path else "")
        )
    typer.echo("✅ Pod bootstrap complete.")

    if policy_name in _POLICIES_NEEDING_TRANSFORMERS_PI_PATCH:
        _install_transformers_pi_patch(pod, key_path, log_path)


def _install_transformers_pi_patch(pod: dict, key_path: Path, log_path: Optional[Path] = None) -> None:
    typer.echo(f"📦 Installing patched transformers fork (required for pi0/pi0_fast/pi05)...")
    install_cmd = (
        "(command -v uv >/dev/null 2>&1 || pip install -q uv) && "
        f"uv pip install --system --break-system-packages -q '{_TRANSFORMERS_PI_PATCH_SPEC}'"
    )
    rc = (
        run_remote_to_file(pod, key_path, install_cmd, log_path)
        if log_path
        else run_remote(pod, key_path, install_cmd)
    )
    if rc != 0:
        raise RuntimeError(
            "Failed to install the patched transformers fork required for this policy."
            + (f" See {log_path} for details." if log_path else "")
        )
    typer.echo("✅ transformers patch installed.")


def sync_dataset_if_local(pod: dict, key_path: Path, dataset_repo_id: str) -> None:
    if not dataset_repo_id.startswith("local/"):
        typer.echo(f"☁️  Dataset '{dataset_repo_id}' will be pulled from HuggingFace Hub on the pod.")
        return

    from lerobot.utils.constants import HF_LEROBOT_HOME
    local_path = HF_LEROBOT_HOME / dataset_repo_id
    if not local_path.exists():
        raise RuntimeError(f"Local dataset not found at {local_path}")

    typer.echo(f"📤 Syncing local dataset '{dataset_repo_id}' to pod...")
    remote_path = f"~/.cache/huggingface/lerobot/{dataset_repo_id}"
    run_remote(pod, key_path, f"mkdir -p {remote_path}")
    rsync_to_pod(pod, key_path, str(local_path), remote_path)
    typer.echo("✅ Dataset synced.")


def push_training_config(pod: dict, key_path: Path, training_config: dict) -> None:
    host, port = _ssh_conn(pod)
    tmp_path = Path("/tmp/solo_runpod_train_config.json")
    tmp_path.write_text(json.dumps({"lerobot": {"mode_configs": {"training": training_config}}}, indent=4))
    run_remote(pod, key_path, "mkdir -p ~/.solo")
    subprocess.run(
        ["scp", "-i", str(key_path), "-P", str(port), "-o", "StrictHostKeyChecking=accept-new",
         str(tmp_path), f"root@{host}:~/.solo/config.json"],
        check=True,
    )
    tmp_path.unlink()


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------

def run_training_on_runpod(
    dataset_repo_id: str,
    policy_name: str,
    training_steps: int,
    batch_size: int,
    output_dir: str,
    push_to_hub: bool,
    policy_repo_id: str,
    use_wandb: bool,
    wandb_project: str,
    pretrained_policy_path: Optional[str] = None,
) -> None:
    typer.echo("\n☁️  Runpod Training")

    client = RunpodClient()
    key_path, public_key = ensure_local_ssh_key()

    pod = select_or_create_pod(client, public_key)
    host, port = _ssh_conn(pod)
    typer.echo(f"✅ Pod ready: {pod['name']} ({pod['id']}) at {host}:{port}")

    try:
        wait_for_ssh(pod, key_path)
    except TimeoutError as e:
        typer.echo(f"❌ {e}")
        typer.echo(f"\nYour public key (add it at https://www.runpod.io/console/user/settings):\n{public_key}")
        return

    bootstrap_pod(pod, key_path, policy_name=policy_name)
    sync_dataset_if_local(pod, key_path, dataset_repo_id)

    remote_output_dir = f"/workspace/train/{Path(output_dir).name}"
    training_config = {
        "dataset_repo_id": dataset_repo_id,
        "output_dir": remote_output_dir,
        "policy_type": policy_name,
        "training_args": {
            "training_steps": training_steps,
            "batch_size": batch_size,
            "push_to_hub": push_to_hub,
            "policy_repo_id": policy_repo_id,
            "use_wandb": use_wandb,
            "wandb_project": wandb_project,
            "pretrained_path": pretrained_policy_path,
        },
    }
    push_training_config(pod, key_path, training_config)

    env_vars = {"SOLO_REMOTE_TRAINING": "1"}
    if push_to_hub:
        hf_token = _local_hf_token()
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token
        else:
            typer.echo("⚠️  No local HuggingFace token found - remote push_to_hub may fail.")
    if use_wandb:
        wandb_key = _local_wandb_key()
        if wandb_key:
            env_vars["WANDB_API_KEY"] = wandb_key
        else:
            typer.echo("⚠️  No local WandB API key found - remote WandB logging may fail.")

    env_prefix = "".join(f"{k}={v} " for k, v in env_vars.items())
    typer.echo(f"\n🚀 Starting remote training on {pod['name']}...\n")
    rc = run_remote(pod, key_path, f"cd ~ && {env_prefix}solo robo --train --yes")

    if rc != 0:
        typer.echo("❌ Remote training exited with an error. Checkpoints (if any) will still be synced back.")

    typer.echo(f"\n📥 Syncing checkpoints back to {output_dir}...")
    try:
        rsync_from_pod(pod, key_path, remote_output_dir, output_dir)
        typer.echo(f"✅ Checkpoints synced to {output_dir}")
    except subprocess.CalledProcessError as e:
        typer.echo(f"⚠️  Failed to sync checkpoints back: {e}")

    if push_to_hub and policy_repo_id:
        typer.echo(f"🚀 Model pushed to HuggingFace Hub: https://huggingface.co/{policy_repo_id}")

    if Confirm.ask(f"\n🛑 Stop pod {pod['name']} now to stop billing?", default=True):
        try:
            client.stop_pod(pod["id"])
            typer.echo("✅ Pod stopped.")
        except (requests.RequestException, RuntimeError) as e:
            # Everything that matters (training, checkpoint sync) has already
            # happened by this point - a network hiccup here shouldn't crash
            # the command, but the pod may still be billing, so say so clearly.
            typer.echo(f"⚠️  Could not stop pod {pod['name']} ({pod['id']}): {e}")
            typer.echo(
                f"   It may still be running and billing - stop it manually at "
                f"https://www.runpod.io/console/pods (pod: {pod['name']}, id: {pod['id']})"
            )


# ---------------------------------------------------------------------------
# Multi-policy comparison: one pod per policy, same dataset/steps/batch_size
# ---------------------------------------------------------------------------

def _create_pod_noninteractive(
    client: RunpodClient, name: str, gpu_type_id: str, public_key: str, cloud_type: str = "SECURE", max_attempts: int = 3
) -> dict:
    """Like _create_pod_interactive, but no prompts - used when creation was already confirmed once for a whole batch."""
    for attempt in range(1, max_attempts + 1):
        try:
            return client.create_pod(name=name, public_key=public_key, gpu_type_ids=[gpu_type_id], cloud_type=cloud_type)
        except requests.HTTPError as e:
            if not _is_capacity_error(e) or attempt == max_attempts:
                raise
            time.sleep(2)
    raise RuntimeError(f"Could not create pod '{name}' after {max_attempts} attempts - all attempted hosts were out of capacity.")


def _run_one_policy_comparison(
    client: RunpodClient,
    public_key: str,
    key_path: Path,
    gpu_type_id: str,
    policy_name: str,
    dataset_repo_id: str,
    training_steps: int,
    batch_size: int,
    push_to_hub: bool,
    use_wandb: bool,
    wandb_project: str,
    log_lock: threading.Lock,
    cloud_type: str = "SECURE",
) -> dict:
    def log(message: str) -> None:
        with log_lock:
            typer.echo(f"[{policy_name}] {message}")

    output_dir = f"outputs/train/{dataset_repo_id.replace('/', '_')}_{policy_name}"
    log_path = Path(output_dir) / "remote_install.log"

    try:
        log("Creating pod...")
        pod = _create_pod_noninteractive(client, f"solo-compare-{policy_name}", gpu_type_id, public_key, cloud_type=cloud_type)
        log(f"Pod {pod['id']} created, waiting for it to come online (this can take several minutes)...")
        pod = _wait_until_ready(client, pod["id"])
        host, port = _ssh_conn(pod)
        log(f"Pod ready at {host}:{port}")

        wait_for_ssh(pod, key_path)
        log("Bootstrapping solo-cli + lerobot (output logged to file, not streamed)...")
        bootstrap_pod(pod, key_path, log_path=log_path, policy_name=policy_name)

        pretrained_policy_path = "lerobot/smolvla_base" if policy_name == "smolvla" else None
        remote_output_dir = f"/workspace/train/{Path(output_dir).name}"
        training_config = {
            "dataset_repo_id": dataset_repo_id,
            "output_dir": remote_output_dir,
            "policy_type": policy_name,
            "training_args": {
                "training_steps": training_steps,
                "batch_size": batch_size,
                "push_to_hub": push_to_hub,
                "policy_repo_id": "",
                "use_wandb": use_wandb,
                "wandb_project": wandb_project,
                "pretrained_path": pretrained_policy_path,
            },
        }
        push_training_config(pod, key_path, training_config)

        env_vars = {"SOLO_REMOTE_TRAINING": "1"}
        if push_to_hub:
            hf_token = _local_hf_token()
            if hf_token:
                env_vars["HF_TOKEN"] = hf_token
        if use_wandb:
            wandb_key = _local_wandb_key()
            if wandb_key:
                env_vars["WANDB_API_KEY"] = wandb_key
        env_prefix = "".join(f"{k}={v} " for k, v in env_vars.items())

        train_log_path = Path(output_dir) / "remote_train.log"
        log(f"Training started (log: {train_log_path})...")
        rc = run_remote_to_file(
            pod, key_path, f"cd ~ && {env_prefix}solo robo --train --yes", train_log_path
        )
        log(f"Training exited with code {rc}")

        log(f"Syncing checkpoints to {output_dir}...")
        rsync_from_pod(pod, key_path, remote_output_dir, output_dir)
        log("Checkpoints synced.")

        try:
            client.stop_pod(pod["id"])
            log("Pod stopped.")
        except (requests.RequestException, RuntimeError) as e:
            log(f"⚠️  Could not stop pod {pod['id']}: {e} - stop manually at https://www.runpod.io/console/pods")

        return {"policy": policy_name, "status": "success" if rc == 0 else "training_failed", "output_dir": output_dir}
    except Exception as e:
        log(f"❌ Failed: {e}")
        return {"policy": policy_name, "status": "error", "error": str(e)}


def run_policy_comparison_on_runpod(
    dataset_repo_id: str,
    training_steps: int,
    batch_size: int,
    push_to_hub: bool,
    use_wandb: bool,
    wandb_project: str,
) -> None:
    """Train every supported policy type on the same dataset/steps/batch_size, each on its own fresh Runpod pod, in parallel."""
    typer.echo(f"\n☁️  Runpod Policy Comparison - {len(ALL_POLICIES)} policies, one pod each")
    typer.echo(f"Policies: {', '.join(ALL_POLICIES)}")

    client = RunpodClient()
    key_path, public_key = ensure_local_ssh_key()

    cloud_type, gpu_type_id = _select_cloud_and_gpu(client)
    typer.echo(
        f"💰 About to create {len(ALL_POLICIES)} pods, each 1x {gpu_type_id} on {cloud_type} cloud. Check current "
        f"$/hr pricing at https://www.runpod.io/console/pods - this is {len(ALL_POLICIES)}x a single pod's hourly "
        "rate, running concurrently, and starts billing immediately for all of them."
    )
    if not Confirm.ask(f"Proceed with creating {len(ALL_POLICIES)} pods?", default=True):
        raise typer.Abort()

    log_lock = threading.Lock()
    results = []
    with ThreadPoolExecutor(max_workers=len(ALL_POLICIES)) as executor:
        futures = {
            executor.submit(
                _run_one_policy_comparison,
                client, public_key, key_path, gpu_type_id, policy_name,
                dataset_repo_id, training_steps, batch_size, push_to_hub, use_wandb, wandb_project,
                log_lock, cloud_type,
            ): policy_name
            for policy_name in ALL_POLICIES
        }
        for future in as_completed(futures):
            results.append(future.result())

    typer.echo("\n📊 Comparison run summary:")
    for r in sorted(results, key=lambda r: r["policy"]):
        icon = "✅" if r["status"] == "success" else "❌"
        detail = r.get("output_dir") or r.get("error", r["status"])
        typer.echo(f"   {icon} {r['policy']}: {detail}")
