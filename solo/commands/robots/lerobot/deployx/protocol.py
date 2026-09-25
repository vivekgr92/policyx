"""
Wire protocol shared by the DeployX policy server (runs on a Runpod GPU pod)
and the DeployX edge agent (runs on the Mac mini next to the SO-101 arms).

Both sides import this module - it is the single source of truth for message
shape, so the two halves can be built independently without the schema
drifting apart. Transport is a JSON WebSocket connection, one message per
line, mirroring the request/response pattern OpenWAM's policy server uses
(ping / predict / reset) since that shape is already proven in this project.

Observation/action key names follow lerobot's own dataset feature naming
convention (`observation.state`, `observation.images.<camera>`, `action`)
so a batch built here needs no remapping to feed a lerobot policy directly.
"""

from typing import Optional, TypedDict


class ObservationPayload(TypedDict, total=False):
    images: dict[str, str]       # camera_name -> base64-encoded JPEG
    state: list[float]           # observation.state, raw joint positions
    task: str                    # task description string, e.g. "pick up the red cube"


class PredictRequest(TypedDict):
    type: str                    # "predict"
    observation: ObservationPayload
    request_id: str              # edge agent's own step counter/uuid, echoed back for correlation


class PredictResponse(TypedDict, total=False):
    type: str                    # "action"
    request_id: str              # echoes PredictRequest.request_id
    action_chunk: list[list[float]]  # (chunk_len, action_dim)
    latency_ms: float
    error: Optional[str]         # set (and action_chunk omitted) if inference failed


class PingRequest(TypedDict):
    type: str                    # "ping"


class PingResponse(TypedDict):
    type: str                    # "pong"
    ready: bool                  # true once the policy checkpoint is loaded and GPU-resident


class ResetRequest(TypedDict):
    type: str                    # "reset"


class ResetResponse(TypedDict):
    type: str                    # "reset_ack"


# Message type constants - use these instead of hardcoding the strings.
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_PREDICT = "predict"
MSG_ACTION = "action"
MSG_RESET = "reset"
MSG_RESET_ACK = "reset_ack"
MSG_ERROR = "error"

# Default WebSocket port for the DeployX policy server (distinct from
# OpenWAM's 8848 default, so both can run on the same Runpod pod if needed).
DEFAULT_PORT = 8849

# Edge agent -> server: how long to wait for a predict response before the
# watchdog treats it as a lost connection and triggers a safe-stop.
PREDICT_TIMEOUT_S = 5.0

# Edge agent -> server: ping interval used to detect a silently-dead
# connection between control steps (e.g. pod stopped, network partition).
HEARTBEAT_INTERVAL_S = 2.0
