"""
Minimal async client for OpenWAM's "given action -> predicted observation"
capability (github.com/OpenWAM-Official/OpenWAM).

This is a SEPARATE server/protocol from protocol.py (DeployX's own Runpod
policy-server wire format) - OpenWAM is a different codebase built and
maintained separately, so its message schema lives in its own module here
rather than being folded into protocol.py.

Used out-of-band by edge_agent.py: once policy_server.py returns an action
chunk, edge_agent.py fires a background asyncio task at this module so
OpenWAM can predict what observation that action *should* produce, in
parallel with the action executing for real on hardware. Never called
synchronously from the control loop.

Message shape mirrors protocol.py's style (typed dicts, a `type` field, a
`request_id` for correlation) for consistency across this project, but the
exact field names below are a best guess at OpenWAM's actual contract - the
OpenWAM-side endpoint for this capability is being built in parallel by a
separate agent. All message construction/parsing is isolated in
`_build_predict_request` / `_parse_predict_response` below so a schema change
on their end is a one-place edit here.
"""

import asyncio
import json
from typing import Optional, TypedDict


class GivenActionRequest(TypedDict, total=False):
    type: str                    # "predict_given_action"
    observation: dict            # {"images": {cam_name: base64 jpeg}, "task": str}
    action: list                 # the raw action chunk to condition on, (chunk_len, action_dim)
    request_id: str


class GivenActionResponse(TypedDict, total=False):
    type: str                    # "prediction"
    request_id: str
    video: list                  # predicted future frames, shape/encoding TBD by OpenWAM's side
    actions: list                # echoed/refined action sequence, if OpenWAM's server returns one
    error: Optional[str]


# Message type constants, kept separate from protocol.py's MSG_* constants
# since this is a different wire format.
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_PREDICT_GIVEN_ACTION = "predict_given_action"
MSG_PREDICTION = "prediction"
MSG_RESET = "reset"
MSG_RESET_ACK = "reset_ack"

# OpenWAM's default port per protocol.py's comment (kept distinct from
# DeployX's own DEFAULT_PORT=8849 so both servers can run on one pod).
DEFAULT_PORT = 8848

# Given-action prediction can involve generating video frames, so it's
# allowed more time than protocol.PREDICT_TIMEOUT_S - this call only ever
# runs out-of-band (see module docstring), never blocking the control loop.
PREDICT_TIMEOUT_S = 15.0


class OpenWAMBridgeError(Exception):
    """Raised for OpenWAM protocol/connection problems the caller can't recover from."""


def _build_predict_request(images: dict, action_chunk: list, request_id: str, task: str = "") -> dict:
    """Build the given-action predict request. Isolated so a field-name change
    on OpenWAM's side (once their endpoint contract is finalized) is a
    one-place edit."""
    return {
        "type": MSG_PREDICT_GIVEN_ACTION,
        "observation": {"images": images, "task": task},
        "action": action_chunk,
        "request_id": request_id,
    }


def _parse_predict_response(response: dict) -> dict:
    """Validate and unwrap a predict response. Isolated alongside
    `_build_predict_request` for the same reason."""
    if response.get("error"):
        raise OpenWAMBridgeError(f"OpenWAM server error: {response['error']}")
    return {
        "request_id": response.get("request_id"),
        "video": response.get("video"),
        "actions": response.get("actions"),
    }


async def _send_json(ws, message: dict) -> None:
    await ws.send(json.dumps(message))


async def _recv_json(ws, timeout: float) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(raw)


async def ping_openwam(server_url: str, timeout_s: float = PREDICT_TIMEOUT_S) -> bool:
    """Connect, ping, and report readiness. Used for standalone health checks, not the hot path."""
    import websockets

    try:
        async with websockets.connect(server_url, max_size=None) as ws:
            await _send_json(ws, {"type": MSG_PING})
            pong = await _recv_json(ws, timeout_s)
            return pong.get("type") == MSG_PONG and bool(pong.get("ready", False))
    except (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError) as e:
        raise OpenWAMBridgeError(f"OpenWAM ping failed: {e}") from e


async def reset_openwam(server_url: str, timeout_s: float = PREDICT_TIMEOUT_S) -> None:
    """Reset OpenWAM's rollout/session state, mirroring protocol.py's reset handshake."""
    import websockets

    try:
        async with websockets.connect(server_url, max_size=None) as ws:
            await _send_json(ws, {"type": MSG_RESET})
            ack = await _recv_json(ws, timeout_s)
            if ack.get("type") != MSG_RESET_ACK:
                raise OpenWAMBridgeError(f"OpenWAM reset failed: {ack}")
    except (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError) as e:
        raise OpenWAMBridgeError(f"OpenWAM reset failed: {e}") from e


async def predict_observation_openwam(
    server_url: str,
    images: dict,
    action_chunk: list,
    request_id: str,
    task: str = "",
    timeout_s: float = PREDICT_TIMEOUT_S,
) -> dict:
    """
    Send a given action chunk (plus the observation it was predicted from) to
    an OpenWAM instance and return its predicted-observation response.

    Opens and closes its own short-lived connection per call rather than
    holding a persistent socket, so concurrent fire-and-forget calls from
    edge_agent.py (one per control step) never need to share/serialize access
    to a single connection. Always meant to be called from a background
    asyncio task - never awaited inline in the control loop.
    """
    import websockets

    try:
        async with websockets.connect(server_url, max_size=None) as ws:
            request = _build_predict_request(images, action_chunk, request_id, task)
            await _send_json(ws, request)
            response = await _recv_json(ws, timeout_s)
    except (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError) as e:
        raise OpenWAMBridgeError(f"OpenWAM request failed: {e}") from e

    if response.get("request_id") not in (None, request_id):
        raise OpenWAMBridgeError(f"Mismatched OpenWAM response: {response}")

    return _parse_predict_response(response)
