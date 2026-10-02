"""
Standalone Gradio playground for manually testing the local Ollama VLM judge
model (see vlm_judge.py) against a LIVE camera feed - type an arbitrary
perception question (e.g. "is the cup upright?") and ask it against whatever
the camera currently sees, without running a full replay session each time.

This is a dev/debug tool only - not wired into `solo robo`/`cli.py`. Run it
as a module, NOT as a direct script path:

    python3 -m solo.commands.robots.lerobot.vlm_judge_playground

(Direct-path invocation - `python3 solo/commands/robots/lerobot/vlm_judge_playground.py`
- genuinely breaks here: Python prepends the script's own directory to
sys.path, and this exact directory already contains an unrelated
`lerobot.py` file - `import lerobot` then resolves to that single-file
module instead of the real installed `lerobot` package, crashing with
"No module named 'lerobot.cameras'; 'lerobot' is not a package". Verified
this live - -m module invocation has no such collision.)

Reuses the production judge's real Ollama-calling constants/helpers
(_OLLAMA_API_BASE, _OLLAMA_JUDGE_MODEL, _OLLAMA_NUM_CTX, _OLLAMA_KEEP_ALIVE,
_frame_to_base64_jpeg) and this repo's own camera-detection helper
(find_available_cameras) rather than duplicating either - this talks to the
exact same local model the real judge uses, just with a free-form question
instead of the fixed judging prompt.
"""

import json
import threading

import cv2
import gradio as gr
import requests

from solo.commands.robots.lerobot.cameras import find_available_cameras
from solo.commands.robots.lerobot.vlm_judge import (
    _OLLAMA_API_BASE,
    _OLLAMA_JUDGE_MODEL,
    _OLLAMA_NUM_CTX,
    _OLLAMA_KEEP_ALIVE,
    _frame_to_base64_jpeg,
)

DEFAULT_QUESTION = "Is the cup upright?"


def _pick_camera_index() -> int:
    """Reuse this repo's own camera detection rather than guessing an OpenCV
    index - picks the first camera found, same default a single-camera setup
    (the common case here) would resolve to in the real recording/replay
    flow."""
    cameras = find_available_cameras()
    if not cameras:
        raise RuntimeError(
            "No cameras detected (checked OpenCV + RealSense, same as the real "
            "recording/replay flow). Plug in a camera and restart this tool."
        )
    cam_id = cameras[0].get("id", 0)
    print(f"📷 Using camera: {cameras[0].get('type', 'Unknown')} (ID: {cam_id})")
    return cam_id


class _LiveCamera:
    """Keeps one OpenCV capture open for the life of the app and tracks the
    most recently read frame, so the Timer-driven preview and the "Ask"
    button both read the same live frame without reopening the camera per
    call (reopening repeatedly was observed elsewhere in this project to
    need a real settle delay - see cameras.py's validate_camera_accessible)."""

    def __init__(self, camera_index: int):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")
        self._lock = threading.Lock()
        self._latest_rgb = None

    def read_latest(self):
        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return self._latest_rgb
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._latest_rgb = frame_rgb
        return frame_rgb

    def get_latest(self):
        with self._lock:
            return self._latest_rgb


def _ask_ollama(frame_rgb, question: str) -> str:
    """Direct call to the same local Ollama endpoint/model the real judge
    uses, with a free-form question instead of the fixed judging prompt -
    same request shape (messages[0].images as base64 JPEG, matching
    num_ctx/keep_alive) as vlm_judge._judge_episode_ollama(), simplified to a
    single non-streaming call since this is a manual one-off test, not a
    long episode-judging call that needs progress/timeout handling."""
    if frame_rgb is None:
        return "⚠️ No frame available yet - is the camera actually producing frames?"
    if not question or not question.strip():
        return "⚠️ Enter a question first."

    image_b64 = _frame_to_base64_jpeg(frame_rgb)
    payload = {
        "model": _OLLAMA_JUDGE_MODEL,
        "messages": [{"role": "user", "content": question.strip(), "images": [image_b64]}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": _OLLAMA_NUM_CTX},
        "keep_alive": _OLLAMA_KEEP_ALIVE,
    }
    try:
        resp = requests.post(f"{_OLLAMA_API_BASE}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return content.strip() or "(empty response from the model)"
    except requests.exceptions.ConnectionError:
        return (
            f"⚠️ Could not reach Ollama at {_OLLAMA_API_BASE} - is it installed "
            f"and running? (https://ollama.com, then `ollama pull {_OLLAMA_JUDGE_MODEL}`)"
        )
    except Exception as e:
        return f"⚠️ Ollama call failed: {e}"


def build_app(camera: _LiveCamera) -> gr.Blocks:
    with gr.Blocks(title="VLM Judge Playground") as demo:
        gr.Markdown(
            f"## VLM Judge Playground\n"
            f"Live feed + free-form questions against the local judge model "
            f"(`{_OLLAMA_JUDGE_MODEL}` via Ollama) - for manually poking at "
            f"perception accuracy without running a full replay session."
        )
        with gr.Row():
            image = gr.Image(label="Live camera feed", interactive=False)
            with gr.Column():
                question = gr.Textbox(label="Question", value=DEFAULT_QUESTION, lines=2)
                ask_btn = gr.Button("Ask", variant="primary")
                answer = gr.Textbox(label="Model response", lines=8, interactive=False)

        timer = gr.Timer(0.5)
        timer.tick(fn=camera.read_latest, outputs=image)

        def _on_ask(question_text):
            frame = camera.get_latest()
            return _ask_ollama(frame, question_text)

        ask_btn.click(fn=_on_ask, inputs=question, outputs=answer)

    return demo


def main():
    camera_index = _pick_camera_index()
    camera = _LiveCamera(camera_index)
    demo = build_app(camera)
    demo.launch()


if __name__ == "__main__":
    main()
