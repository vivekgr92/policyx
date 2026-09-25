"""
DeployX policy server.

Runs on the Runpod GPU pod. Loads a lerobot policy checkpoint once at startup
and serves predict/ping/reset requests from the Mac-mini edge agent over the
WebSocket protocol defined in `deployx/protocol.py` (not modified here).

Checkpoint loading follows the inference-mode pattern already used in
`solo/commands/robots/lerobot/utils/record_config.py` (PreTrainedConfig.from_pretrained
+ solo: Hub-ref resolution). Observation batching and the pre/postprocessor
pipeline follow lerobot's own `lerobot.async_inference.policy_server` - that
module is the first-party precedent for "serve a lerobot policy over the
wire", so its normalization/tokenization contract (make_pre_post_processors,
predict_action_chunk, per-timestep postprocessing) is reused here rather than
re-deriving normalization by hand.
"""

import asyncio
import base64
import io
import json
import logging
import time
from typing import Any, Optional

import typer

from solo.commands.robots.lerobot.deployx import protocol

logger = logging.getLogger("deployx.policy_server")


class _ServerState:
    """Shared, mutable server state. `ready` flips to True once the checkpoint
    (weights + pre/postprocessor pipelines) has finished loading."""

    def __init__(self):
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = "cpu"
        self.ready = False
        self.load_error: Optional[str] = None


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve a `solo:org/model` Hub reference to a local snapshot dir, exactly
    like the inference-mode branch of `unified_record_config` does."""
    from solo.hub import is_solo_ref, parse_solo_ref

    if is_solo_ref(checkpoint_path):
        from solo.hub import solo_snapshot_download

        clean_id = parse_solo_ref(checkpoint_path)
        logger.info(f"Downloading model from Solo Hub: {clean_id}...")
        local_path = solo_snapshot_download(repo_id=clean_id)
        logger.info(f"Model cached at: {local_path}")
        return local_path
    return checkpoint_path


def _load_checkpoint(checkpoint_path: str, state: "_ServerState") -> None:
    """Load policy weights + processors onto the GPU (or CPU if none). Runs in a
    background thread so the WebSocket server is already accepting connections
    (and answering `ping` with `ready: false`) while this is in flight."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    try:
        local_path = _resolve_checkpoint_path(checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        policy_config = PreTrainedConfig.from_pretrained(local_path)
        policy_config.pretrained_path = local_path
        policy_config.device = device

        policy_class = get_policy_class(policy_config.type)
        policy = policy_class.from_pretrained(local_path, config=policy_config)
        # from_pretrained already moves the model to config.device and calls .eval()

        # Loaded from the checkpoint dir (saved at train time), so normalization
        # stats / tokenizer config match what the policy was actually trained on.
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=local_path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )

        state.policy = policy
        state.preprocessor = preprocessor
        state.postprocessor = postprocessor
        state.device = device
        state.ready = True
        logger.info(f"Policy '{policy_config.type}' loaded on {device} from {local_path}")
    except Exception as e:
        # Don't crash the process - the server stays up and answers ping/error so
        # a detached remote run is still debuggable (and diagnosable) over the wire.
        state.load_error = str(e)
        logger.exception("Failed to load policy checkpoint")


def _decode_base64_jpeg(b64_jpeg: str):
    import numpy as np
    from PIL import Image

    raw = base64.b64decode(b64_jpeg)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image)  # (H, W, C) uint8


def _build_batch(policy, observation: dict) -> dict:
    """Raw observation payload -> tensor batch dict, following the same tensor
    contract as `lerobot.utils.control_utils.predict_action` /
    `lerobot.async_inference.helpers.raw_observation_to_observation`: images as
    float32 in [0, 1], channel-first, batch dim added; state batched the same way."""
    import torch
    import torch.nn.functional as F

    batch: dict[str, Any] = {}

    state = observation.get("state") or []
    batch["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    image_features = policy.config.image_features  # {"observation.images.<cam>": PolicyFeature(shape=(C,H,W))}
    for camera_name, b64_jpeg in (observation.get("images") or {}).items():
        key = f"observation.images.{camera_name}"
        img = _decode_base64_jpeg(b64_jpeg)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W) in [0, 1]

        target = image_features.get(key)
        if target is not None and tuple(img_t.shape[1:]) != tuple(target.shape[1:]):
            # Edge camera resolution may not match what the policy was trained on.
            img_t = F.interpolate(
                img_t.unsqueeze(0), size=target.shape[1:], mode="bilinear", align_corners=False
            ).squeeze(0)
        batch[key] = img_t.unsqueeze(0)

    task = observation.get("task")
    if task:
        batch["task"] = task

    return batch


def _run_inference(state: "_ServerState", observation: dict) -> list:
    """Blocking (CPU/GPU-bound) inference call - run via an executor so it doesn't
    block the asyncio event loop."""
    import torch

    batch = _build_batch(state.policy, observation)
    batch = {k: (v.to(state.device) if hasattr(v, "to") else v) for k, v in batch.items()}
    batch = state.preprocessor(batch)

    with torch.inference_mode():
        action_chunk = state.policy.predict_action_chunk(batch)
    if action_chunk.ndim == 2:
        action_chunk = action_chunk.unsqueeze(0)  # (1, chunk_len, action_dim)

    # Postprocessor (unnormalization) expects one (B, action_dim) slice at a time -
    # mirrors lerobot.async_inference.policy_server._predict_action_chunk exactly.
    _, chunk_len, _ = action_chunk.shape
    processed = [state.postprocessor(action_chunk[:, i, :]) for i in range(chunk_len)]
    action_chunk = torch.stack(processed, dim=1).squeeze(0)  # (chunk_len, action_dim)

    return action_chunk.detach().cpu().tolist()


async def _send_error(websocket, request_id: Optional[str], message: str) -> None:
    payload = {"type": protocol.MSG_ERROR, "error": message}
    if request_id is not None:
        payload["request_id"] = request_id
    await websocket.send(json.dumps(payload))


async def _handle_predict(websocket, state: "_ServerState", lock: asyncio.Lock, msg: dict) -> None:
    request_id = msg.get("request_id")
    if not state.ready:
        reason = state.load_error or "policy checkpoint not loaded yet"
        await _send_error(websocket, request_id, reason)
        return

    observation = msg.get("observation") or {}
    loop = asyncio.get_running_loop()
    start = time.monotonic()
    async with lock:  # serialize inference: one policy/GPU, one connection at a time
        action_chunk = await loop.run_in_executor(None, _run_inference, state, observation)
    latency_ms = (time.monotonic() - start) * 1000

    await websocket.send(json.dumps({
        "type": protocol.MSG_ACTION,
        "request_id": request_id,
        "action_chunk": action_chunk,
        "latency_ms": latency_ms,
    }))


async def _handle_reset(websocket, state: "_ServerState", lock: asyncio.Lock) -> None:
    async with lock:
        if state.policy is not None:
            state.policy.reset()
        if state.preprocessor is not None:
            state.preprocessor.reset()
        if state.postprocessor is not None:
            state.postprocessor.reset()
    await websocket.send(json.dumps({"type": protocol.MSG_RESET_ACK}))


async def _handle_connection(websocket, state: "_ServerState", lock: asyncio.Lock) -> None:
    from websockets.exceptions import ConnectionClosed

    peer = getattr(websocket, "remote_address", None)
    logger.info(f"client connected: {peer}")
    try:
        async for raw in websocket:
            start = time.monotonic()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as e:
                await _send_error(websocket, None, f"invalid JSON: {e}")
                continue

            msg_type = msg.get("type")
            try:
                if msg_type == protocol.MSG_PING:
                    await websocket.send(json.dumps({"type": protocol.MSG_PONG, "ready": state.ready}))
                elif msg_type == protocol.MSG_PREDICT:
                    await _handle_predict(websocket, state, lock, msg)
                elif msg_type == protocol.MSG_RESET:
                    await _handle_reset(websocket, state, lock)
                else:
                    await _send_error(websocket, msg.get("request_id"), f"unknown message type: {msg_type}")
            except Exception as e:
                # A bad frame or a single failed inference call must not drop the
                # connection or take down the server - hardware-in-the-loop control
                # depends on the edge agent's watchdog seeing an error, not a hang.
                logger.exception(f"error handling message type={msg_type}")
                await _send_error(websocket, msg.get("request_id"), str(e))
            finally:
                latency_ms = (time.monotonic() - start) * 1000
                logger.info(f"type={msg_type} latency_ms={latency_ms:.1f}")
    except ConnectionClosed:
        logger.info(f"client disconnected: {peer}")


def run_policy_server(checkpoint_path: str, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
    """Entry point wired from `solo robo --deployx-serve <checkpoint_path>`."""
    import websockets

    port = port or protocol.DEFAULT_PORT
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    async def _main():
        state = _ServerState()
        lock = asyncio.Lock()
        loop = asyncio.get_running_loop()

        typer.echo(f"Loading checkpoint '{checkpoint_path}'...")
        load_future = loop.run_in_executor(None, _load_checkpoint, checkpoint_path, state)

        async def handler(websocket):
            await _handle_connection(websocket, state, lock)

        async with websockets.serve(handler, host, port):
            typer.echo(f"DeployX policy server listening on ws://{host}:{port}")
            await load_future
            if state.ready:
                typer.echo("Checkpoint loaded - ready to serve predictions.")
            else:
                typer.echo(
                    f"Checkpoint failed to load ({state.load_error}). "
                    "Server stays up and will report this on ping/predict."
                )
            await asyncio.Future()  # run forever

    asyncio.run(_main())
