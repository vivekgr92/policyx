"""
Provision (or reuse) a Runpod GPU pod and start the DeployX policy server on it.

Thin orchestration layer around runpod_train.py's already-working Runpod REST
client, SSH key management, and pod selection/bootstrap helpers - the pod
lifecycle (create/select/wait-for-ssh/install solo-cli) is identical to the
training flow, so it's imported and reused here rather than duplicated.
"""

import os
import shlex
import time
from pathlib import Path
from typing import Optional

import typer

from solo.commands.robots.lerobot.deployx import protocol
from solo.commands.robots.lerobot.runpod_train import (
    RunpodClient,
    bootstrap_pod,
    ensure_local_ssh_key,
    rsync_to_pod,
    run_remote,
    select_or_create_pod,
    wait_for_ssh,
)

# Base ports runpod_train.py's create_pod() exposes on a freshly created pod -
# used as the fallback when a pod has no recorded `ports` field to extend.
_BASE_PORTS = ["22/tcp", "8888/http"]


def _ensure_port_exposed(client: RunpodClient, pod: dict, port: int, timeout: int = 120) -> dict:
    """Make sure the DeployX WebSocket port is exposed on the pod, then wait for
    Runpod to hand back a public port mapping for it."""
    port_spec = f"{port}/tcp"
    current_ports = pod.get("ports") or _BASE_PORTS
    if port_spec not in current_ports:
        typer.echo(f"🔧 Exposing port {port} on pod {pod['id']}...")
        client.update_pod(pod["id"], ports=[*current_ports, port_spec])

    if (pod.get("portMappings") or {}).get(str(port)):
        return pod

    typer.echo(f"⏳ Waiting for Runpod to assign a public mapping for port {port}...")
    elapsed = 0
    while elapsed < timeout:
        pod = client.get_pod(pod["id"])
        if (pod.get("portMappings") or {}).get(str(port)):
            return pod
        time.sleep(5)
        elapsed += 5
    raise TimeoutError(
        f"Pod {pod['id']} did not get a public mapping for port {port}/tcp within {timeout}s."
    )


def _sync_checkpoint_if_local(pod: dict, key_path: Path, checkpoint_path: str) -> str:
    """If `checkpoint_path` is a local directory, rsync it to the pod and return the
    remote path to pass to `solo robo --deployx-serve`. Otherwise (an HF repo id or
    a `solo:org/model` Hub reference) return it unchanged - policy_server.py already
    resolves those itself on the pod."""
    if not os.path.isdir(checkpoint_path):
        typer.echo(f"☁️  Checkpoint '{checkpoint_path}' will be resolved on the pod (Hub reference).")
        return checkpoint_path

    remote_path = f"~/deployx_checkpoints/{Path(checkpoint_path).name}"
    typer.echo(f"📤 Syncing local checkpoint '{checkpoint_path}' to pod...")
    run_remote(pod, key_path, f"mkdir -p {remote_path}")
    rsync_to_pod(pod, key_path, checkpoint_path, remote_path)
    typer.echo("✅ Checkpoint synced.")
    return remote_path


def _start_server_remote(pod: dict, key_path: Path, checkpoint_ref: str, port: int) -> None:
    """Launch the policy server detached (nohup + disown) so this SSH command
    returns immediately instead of streaming the server's stdout forever."""
    log_path = "~/deployx_server.log"
    checkpoint_arg = shlex.quote(checkpoint_ref)
    cmd = (
        f"nohup solo robo --deployx-serve {checkpoint_arg} --deployx-port {port} "
        f"> {log_path} 2>&1 < /dev/null & disown; sleep 1; echo DEPLOYX_SERVER_STARTED"
    )
    rc = run_remote(pod, key_path, cmd)
    if rc != 0:
        raise RuntimeError(
            f"Failed to start the DeployX policy server on the pod (check {log_path} on the pod)."
        )


def deploy_policy_server_to_runpod(checkpoint_path: str, port: Optional[int] = None) -> str:
    """
    Provision/reuse a Runpod pod, install solo-cli on it, start the DeployX
    policy server for `checkpoint_path`, and return the ws:// URL the edge
    agent should connect to.
    """
    port = port or protocol.DEFAULT_PORT
    typer.echo("\n☁️  DeployX: Runpod policy server deployment")

    client = RunpodClient()
    key_path, public_key = ensure_local_ssh_key()

    pod = select_or_create_pod(client, public_key)
    typer.echo(f"✅ Pod ready: {pod['name']} ({pod['id']}) at {pod['publicIp']}:{pod['portMappings']['22']}")

    try:
        wait_for_ssh(pod, key_path)
    except TimeoutError as e:
        typer.echo(f"❌ {e}")
        typer.echo(f"\nYour public key (add it at https://www.runpod.io/console/user/settings):\n{public_key}")
        raise

    bootstrap_pod(pod, key_path)
    pod = _ensure_port_exposed(client, pod, port)

    checkpoint_ref = _sync_checkpoint_if_local(pod, key_path, checkpoint_path)

    typer.echo(f"🚀 Starting DeployX policy server on {pod['name']}...")
    _start_server_remote(pod, key_path, checkpoint_ref, port)

    ws_url = f"ws://{pod['publicIp']}:{pod['portMappings'][str(port)]}"
    typer.echo(f"\n✅ DeployX policy server starting on pod {pod['name']} ({pod['id']}).")
    typer.echo(f"   Edge agent connection URL: {ws_url}")
    typer.echo(
        "   The checkpoint may still be loading in the background - poll with a "
        "'ping' message until the response has ready: true."
    )
    return ws_url
