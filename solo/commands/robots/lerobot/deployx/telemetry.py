"""
Per-session telemetry logging for the DeployX edge agent.

Each control step gets one JSONL record under
outputs/deployx/<session_id>/telemetry.jsonl. Camera frames are written as
JPEG files under the same session directory and referenced by relative path
from the record - never inlined as bytes/base64 into the JSONL, so the log
stays small and diffable and the frames can be uploaded as plain dataset
files.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import typer

DEFAULT_TELEMETRY_ROOT = Path("outputs") / "deployx"


def new_session_id() -> str:
    """Timestamp + short random suffix, so sessions sort chronologically and never collide."""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass
class StepRecord:
    """One executed action - the edge agent's unit of telemetry.

    A single `predict` response carries an action *chunk* (several actions),
    so `step` is a global monotonically increasing counter while
    `request_id` groups every StepRecord back to the predict call that
    produced it.
    """

    step: int
    request_id: str
    timestamp: float
    task: str = ""
    observation_state: List[float] = field(default_factory=list)
    image_paths: Dict[str, str] = field(default_factory=dict)  # camera_name -> path relative to session dir
    commanded_action: Optional[List[float]] = None
    is_safe: Optional[bool] = None
    safety_reason: Optional[str] = None
    resulting_state: Optional[List[float]] = None
    latency_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    # OpenWAM's given-action observation prediction for this step's request_id,
    # logged out-of-band once the background task completes (see edge_agent.py's
    # openwam_server_url handling) - None on this step's own record when unset
    # or still pending; a later record with the same request_id may carry it.
    openwam_prediction: Optional[dict] = None

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


class TelemetryLogger:
    """Owns one session's telemetry.jsonl and images/ directory."""

    def __init__(self, session_id: Optional[str] = None, root: Path = DEFAULT_TELEMETRY_ROOT):
        self.session_id = session_id or new_session_id()
        self.session_dir = Path(root) / self.session_id
        self.images_dir = self.session_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.session_dir / "telemetry.jsonl"
        # Line-buffered so a crash mid-session still leaves completed steps on disk.
        self._fh = open(self.jsonl_path, "a", buffering=1)

    def save_frame(self, step: int, camera_name: str, jpeg_bytes: bytes) -> str:
        """Write one camera frame to disk, return its path relative to session_dir."""
        rel_path = f"images/{camera_name}_{step:06d}.jpg"
        (self.session_dir / rel_path).write_bytes(jpeg_bytes)
        return rel_path

    def log_step(self, record: StepRecord) -> None:
        self._fh.write(record.to_json_line() + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def encode_jpeg(frame, quality: int = 90) -> bytes:
    """Encode an HxWx3 uint8 numpy frame (as returned by robot.get_observation()) to JPEG bytes."""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to JPEG-encode camera frame")
    return buf.tobytes()


def push_session_to_hub(session_dir: Path, repo_id: str) -> Optional[str]:
    """
    Push a completed session's telemetry (JSONL + images) to the Hub as a dataset repo.

    Reuses this codebase's existing HuggingFace auth flow (same one
    modes/recording.py relies on for pushing recorded datasets) rather than
    inventing a separate credential path.
    """
    from solo.commands.robots.lerobot.auth import authenticate_huggingface

    login_success, hf_username = authenticate_huggingface()
    if not login_success:
        typer.echo("❌ HuggingFace authentication required to push telemetry.")
        return None

    if "/" not in repo_id:
        repo_id = f"{hf_username}/{repo_id}"

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(session_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"DeployX session {session_dir.name}",
        )
    except Exception as e:
        typer.echo(f"❌ Failed to push telemetry session to hub: {e}")
        return None

    url = f"https://huggingface.co/datasets/{repo_id}"
    typer.echo(f"✅ Telemetry pushed to hub: {url}")
    return url
