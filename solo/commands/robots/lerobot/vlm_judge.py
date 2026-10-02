"""
VLM-based episode quality judge for replay-generated training data.

Given a handful of frames sampled from a replayed episode plus a task
description, asks a vision-capable model whether the episode shows a valid
completion of that task - used to filter out perturbation-induced failures
(dropped object, missed grasp, wrong direction) before they're saved into a
training dataset.

Default backend is now a local Ollama server on the Mac mini itself (set
VLM_JUDGE_BACKEND=ollama, or just leave it unset) - zero per-call cost, no
network round trip, no cold-start-vs-idle-cost tradeoff to manage, per an
explicit request to stop spending time/money on Runpod for this while it's
still being tested. Requires Ollama installed (https://ollama.com) and the
judge model pulled locally (`ollama pull qwen3-vl:4b` by default - see
OLLAMA_JUDGE_MODEL below to use a different one). See
_judge_episode_ollama()'s docstring for the model choice rationale and real
caveats on this hardware.

Runpod serverless (set VLM_JUDGE_BACKEND=runpod) remains available as an
explicit opt-in for later - a self-hosted Qwen2.5-VL-7B-Instruct model
(endpoint "qwen-vlm-judge", id lt8yd7ssvip3y9, GPU pool ADA_24 / RTX 4090 at
$1.10/hr serverless, deployed via the official runpod-workers/worker-vllm Hub
image), originally chosen over a frontier API because this judge call runs on
every perturbed replay episode, potentially thousands of times during a real
data-collection campaign, where per-call frontier-API token costs add up
fast. The endpoint is true scale-to-zero (workersMin=0, idleTimeout=120s): no
cost between data-collection sessions, but a real cold start on the first
call of each new session - see _judge_episode_runpod()'s docstring for the
measured real numbers. Worth revisiting once local quality/speed is proven
insufficient for real use.

(The original prototype also had an Anthropic Claude fallback backend; it was
removed per explicit request to keep this file to only the backends actually
in use - local Ollama now, Runpod later.)
"""

import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import requests
import typer
from tqdm import tqdm

DEFAULT_JUDGE_TASK_DESCRIPTION = "Pick cup and place"


@dataclass
class JudgeResult:
    """A judge call's outcome: the verdict (True=valid, False=invalid, None=the
    call itself could not be completed/verified) plus a short human-readable
    reason - the model's own stated reasoning when a verdict was reached, or an
    explanation of what went wrong when it wasn't."""

    verdict: Optional[bool]
    reason: str

# Local Ollama backend (default) - see module docstring for cost/rationale.
# qwen3-vl:4b chosen as the final pick (confirmed by the user) over
# qwen2.5vl:7b (same Qwen-VL lineage already proven on the Runpod backend,
# but 7B's ~5-9GB real runtime footprint leaves uncomfortably little headroom
# on this machine's real hardware - confirmed via sysctl/system_profiler:
# Mac mini, Apple M4, 16GB unified memory) and over llava:7b (older
# architecture, no longer the best fit now that Ollama has first-class
# Qwen3-VL support). Qwen3-VL is a newer generation than Qwen2.5-VL, and
# Ollama's own team describes its smaller sizes as working "exceptionally
# well for their size" - the 4B size trades some raw quality for
# substantially more memory headroom on 16GB unified memory. Override via
# OLLAMA_JUDGE_MODEL if a different local model fits better in practice.
_OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
_OLLAMA_JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "qwen3-vl:4b")

# Runpod serverless backend (opt-in) - see module docstring for cost/rationale.
RUNPOD_JUDGE_ENDPOINT_ID = os.environ.get("RUNPOD_JUDGE_ENDPOINT_ID", "lt8yd7ssvip3y9")
_RUNPOD_JUDGE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
_RUNPOD_API_BASE = "https://api.runpod.ai/v2"
_RUNPOD_POLL_INTERVAL_S = 3.0
_RUNPOD_POLL_TIMEOUT_S = 300.0  # real measured cold start: 189s delay + ~1s exec (see docstring below)

# Frame-selection tuning. No real failure-case data has validated these numbers
# yet (the zero-shot prototype only had clean-success episodes to test against)
# - adjust once real perturbation-induced failures have been judged and checked
# against what a human would've caught.
BASELINE_FRAME_COUNT = 10  # evenly-spaced frames across the whole episode, for general coverage
TOP_PERTURBATION_FRAMES = 5  # highest-realized-perturbation-magnitude frames to zoom in on
PERTURBATION_CONTEXT_WINDOW = 1  # also include this many frames before/after each top-perturbation frame
MAX_TOTAL_JUDGE_FRAMES = 25  # hard cap on total frames sent to the judge per episode (cost/latency)


def select_judge_frame_indices(
    num_frames: int,
    perturbation_magnitudes: Optional[list] = None,
    baseline_count: int = BASELINE_FRAME_COUNT,
    top_k: int = TOP_PERTURBATION_FRAMES,
    context_window: int = PERTURBATION_CONTEXT_WINDOW,
    max_total: int = MAX_TOTAL_JUDGE_FRAMES,
) -> list:
    """Pick frame indices to send to the judge: `baseline_count` evenly-spaced
    frames for general coverage of the whole episode, PLUS - if per-step
    realized perturbation magnitudes are known - the `top_k` highest-magnitude
    frames (each with `context_window` frames of padding before/after), so a
    brief mid-episode failure (a drop-and-recatch, a near-miss that resolves by
    luck) that uniform sampling alone could land between samples and miss
    entirely is still likely to get a frame near it. Capped at `max_total`
    frames total for cost/latency; if the union would exceed that, the
    perturbation-focused frames are kept in full and the baseline set is
    thinned to fit the remaining budget.

    `perturbation_magnitudes`, when given, must have length `num_frames` - one
    scalar per frame, the realized per-step perturbation (not the constant
    target fraction, which is the same for a whole episode and so useless for
    localizing where in the episode perturbation actually hit hardest).
    """
    if num_frames <= 0:
        return []
    if num_frames <= baseline_count:
        baseline = list(range(num_frames))
    else:
        fractions = [i / (baseline_count - 1) for i in range(baseline_count)] if baseline_count > 1 else [0.0]
        baseline = sorted(set(int(round(f * (num_frames - 1))) for f in fractions))

    perturbation_focused = set()
    if perturbation_magnitudes and len(perturbation_magnitudes) == num_frames and top_k > 0:
        ranked = sorted(range(num_frames), key=lambda i: perturbation_magnitudes[i], reverse=True)
        for idx in ranked[:top_k]:
            for offset in range(-context_window, context_window + 1):
                neighbor = idx + offset
                if 0 <= neighbor < num_frames:
                    perturbation_focused.add(neighbor)

    combined = set(baseline) | perturbation_focused
    if len(combined) <= max_total:
        return sorted(combined)

    # Over budget: keep all perturbation-focused frames, thin the baseline to
    # fill whatever's left.
    remaining_budget = max(0, max_total - len(perturbation_focused))
    thinned_baseline = baseline[:: max(1, len(baseline) // max(1, remaining_budget))][:remaining_budget]
    return sorted(perturbation_focused | set(thinned_baseline))


def _frame_to_base64_jpeg(frame) -> str:
    """Encode a (H, W, 3) uint8 RGB numpy array as a base64 JPEG string."""
    from PIL import Image

    image = Image.fromarray(frame.astype("uint8"))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_judge_prompt_text(task_description: str, num_frames: int) -> str:
    return (
        f"You are judging a robot arm demonstration recorded for training data. "
        f"The task description is: \"{task_description}\".\n\n"
        f"The robot arm is set up with two zones on a table, A (left side) and B "
        f"(right side). The task involves picking an object and moving it between "
        f"these zones as described.\n\n"
        f"Below are {num_frames} frames sampled across the episode, in "
        f"chronological order (first frame = episode start, last frame = episode end).\n\n"
        f"Look at how the scene changes from the first frame to the last. Judge "
        f"whether this episode shows a VALID, successful completion of the task "
        f"(the object was actually grasped, moved in the correct direction, and "
        f"released at the target) versus an INVALID one (dropped object, missed "
        f"grasp, wrong direction, or other failure - these can happen when the "
        f"recorded actions have been perturbed with random noise for data "
        f"augmentation).\n\n"
        f"Respond with EXACTLY one line starting with \"VALID\" or \"INVALID\", "
        f"followed by a dash and a one-sentence reason. Example:\n"
        f"VALID - the object moved from the left zone to the right zone and the "
        f"gripper is empty at the end.\n"
        f"Do not include anything else in your response."
    )


def _parse_verdict(verdict_text: str) -> JudgeResult:
    """Parse a model response of the form "VALID - <reason>" / "INVALID -
    <reason>" into a JudgeResult, preserving the model's actual stated reason
    rather than discarding it."""
    typer.echo(f"🧑‍⚖️  VLM judge: {verdict_text}")
    stripped = verdict_text.strip()
    upper = stripped.upper()

    reason = stripped
    if "-" in stripped:
        _, _, after_dash = stripped.partition("-")
        after_dash = after_dash.strip()
        if after_dash:
            reason = after_dash

    if upper.startswith("VALID"):
        return JudgeResult(True, reason)
    if upper.startswith("INVALID"):
        return JudgeResult(False, reason)

    typer.echo("⚠️  VLM judge: unexpected response format - could not verify.")
    return JudgeResult(None, f"unexpected response format: {stripped[:200]}")


def get_runpod_api_key() -> str:
    """Reuses runpod_train.py's existing get_api_key() (same env/config/prompt
    pattern already used for pod management) rather than duplicating it."""
    from solo.commands.robots.lerobot.runpod_train import get_api_key

    return get_api_key()


# Max gap between streamed chunks before treating the call as truly stuck
# rather than just slow under real system load. This machine's 16GB unified
# memory is shared between the solo robo --replay process itself (camera
# capture, rerun visualization) and the model inference all at once during a
# live session - real observed call durations have ranged from ~13s up to a
# genuine >180s read-timeout failure under contention, which a flat
# total-duration timeout can't distinguish from an actual hang. Streaming lets
# us reset the clock on every real chunk received instead: requests' own
# per-read timeout (passed to a stream=True request) already applies to each
# individual socket read rather than the whole response, so a single `timeout=`
# value here gives inactivity semantics for free - verified empirically, not
# assumed.
_OLLAMA_INACTIVITY_TIMEOUT_S = 90.0


def _extract_real_stats(final_chunk: dict) -> str:
    """Build a short real-numbers summary from Ollama's actual final streamed
    chunk (done=true) - real field names confirmed by inspecting a live
    response, not guessed: total_duration/load_duration/prompt_eval_count/
    prompt_eval_duration/eval_count/eval_duration (durations in nanoseconds)."""
    eval_count = final_chunk.get("eval_count")
    eval_duration_ns = final_chunk.get("eval_duration")
    total_duration_ns = final_chunk.get("total_duration")
    parts = []
    if total_duration_ns:
        parts.append(f"{total_duration_ns / 1e9:.1f}s total")
    if eval_count and eval_duration_ns:
        parts.append(f"{eval_count} tokens @ {eval_count / (eval_duration_ns / 1e9):.1f} tok/s")
    return ", ".join(parts) if parts else "no timing stats in final chunk"


def _judge_episode_ollama(frames: list, task_description: str) -> JudgeResult:
    """
    Judge via a local Ollama server (https://ollama.com) running a
    vision-capable model directly on this machine - no network round trip, no
    per-call cost, no cold-start-vs-idle-cost tradeoff. Requires Ollama
    installed and running (`ollama serve`, or the menu-bar app which runs it
    automatically) and the model already pulled (`ollama pull qwen3-vl:4b`,
    or whatever OLLAMA_JUDGE_MODEL is set to).

    Streams the response (see _OLLAMA_INACTIVITY_TIMEOUT_S) and renders two
    real tqdm bars: a determinate one while base64-encoding the known frame
    count, and an indeterminate one (total=None, manually incremented) over
    streamed chunks as they arrive, with the model's real generated text
    echoed live via tqdm.write() so it doesn't corrupt the bar. Real measured
    numbers on this machine (Mac mini, Apple M4, 16GB unified memory, no
    discrete GPU) - see the commit this comment was added in for the actual
    first-call vs warm-call timing and `ollama ps` memory footprint. Judgment
    accuracy is still unverified against real perturbation-induced failures
    (only clean-success episodes were available to test against, same caveat
    as the Runpod path originally had) - treat verdicts as unproven until
    checked against known-bad data.
    """
    if not frames:
        typer.echo("⚠️  VLM judge: no frames to judge - could not verify.")
        return JudgeResult(None, "no camera frames captured")

    typer.echo(f"📤 Sending {len(frames)} frames to {_OLLAMA_JUDGE_MODEL} for judging...")
    images_b64 = [
        _frame_to_base64_jpeg(frame)
        for frame in tqdm(frames, desc="Encoding frames", unit="frame")
    ]
    payload = {
        "model": _OLLAMA_JUDGE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": _build_judge_prompt_text(task_description, len(frames)),
                "images": images_b64,
            }
        ],
        "stream": True,
        # Ollama's default context window (4096 tokens) is too small for a
        # multi-frame judge request - 5 real frames alone measured at 5628
        # tokens (real error: "request (5628 tokens) exceeds the available
        # context size (4096 tokens)"). Sized with real headroom above the
        # MAX_TOTAL_JUDGE_FRAMES=25 worst case (~1075 tokens/image observed).
        "options": {"num_ctx": 32768},
    }

    try:
        resp = requests.post(
            f"{_OLLAMA_API_BASE}/api/chat",
            json=payload,
            timeout=_OLLAMA_INACTIVITY_TIMEOUT_S,
            stream=True,
        )
        resp.raise_for_status()

        content_parts = []
        final_chunk = None
        thinking_started = False
        content_started = False
        with tqdm(total=None, desc="Generating", unit="chunk") as bar:
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                bar.update(1)
                msg = chunk.get("message", {})
                thinking_delta = msg.get("thinking") or ""
                content_delta = msg.get("content") or ""
                if thinking_delta:
                    if not thinking_started:
                        tqdm.write("🧠 thinking: ", end="")
                        thinking_started = True
                    tqdm.write(thinking_delta, end="")
                if content_delta:
                    if not content_started:
                        tqdm.write("\n💬 answer: ", end="")
                        content_started = True
                    tqdm.write(content_delta, end="")
                    content_parts.append(content_delta)
                if chunk.get("done"):
                    final_chunk = chunk
                    break
        if thinking_started or content_started:
            tqdm.write("")  # final newline after the live-streamed text

        if final_chunk is not None:
            typer.echo(f"📊 {_extract_real_stats(final_chunk)}")

        verdict_text = "".join(content_parts).strip()
        if not verdict_text:
            reason = "Ollama produced no final-answer content (only reasoning/thinking tokens)"
            typer.echo(f"⚠️  VLM judge: {reason}")
            return JudgeResult(None, reason)
        return _parse_verdict(verdict_text)
    except requests.exceptions.ConnectionError:
        reason = (
            f"could not reach Ollama at {_OLLAMA_API_BASE} - is it installed and "
            f"running? (https://ollama.com, then `ollama pull {_OLLAMA_JUDGE_MODEL}`)"
        )
        typer.echo(f"⚠️  VLM judge: {reason}")
        return JudgeResult(None, reason)
    except requests.exceptions.ReadTimeout:
        reason = (
            f"Ollama produced no streamed output for {_OLLAMA_INACTIVITY_TIMEOUT_S:.0f}s - "
            f"likely genuinely stuck, not just slow under real system load"
        )
        typer.echo(f"⚠️  VLM judge: {reason}")
        return JudgeResult(None, reason)
    except Exception as e:
        reason = f"Ollama call failed: {e}"
        typer.echo(f"⚠️  VLM judge: {reason}")
        return JudgeResult(None, reason)


def _judge_episode_runpod(frames: list, task_description: str) -> JudgeResult:
    """
    Judge via a self-hosted Qwen2.5-VL-7B-Instruct model on Runpod serverless
    (see module docstring for why this is the default backend over a frontier
    API). The endpoint scales to zero when idle (no cost between sessions) and
    cold-starts on the first call after being idle.

    Real measured numbers (timed directly against this exact endpoint, not
    estimated): a genuinely cold call (confirmed via worker logs - vLLM
    loading Qwen2.5-VL from scratch, not a crash loop) took 189.4s queue/start
    delay + 1.1s execution (~190s total). The very next call, with the worker
    still warm, took 73ms delay + 524ms execution (~0.6s total) - a ~300x
    difference. _RUNPOD_POLL_TIMEOUT_S (300s) is set with real margin above
    the measured 190s cold start so a normal cold start is never mistaken for
    a failure. A genuine timeout (exceeding even a cold start by a wide
    margin) almost certainly means real trouble (capacity exhausted, worker
    crash-looping) rather than "just starting up" - in that case, same as any
    other judge-call failure, the caller fails CLOSED (discards the episode)
    rather than risking an unverified, possibly perturbation-corrupted episode
    silently entering the training set.
    """
    if not frames:
        typer.echo("⚠️  VLM judge: no frames to judge - could not verify.")
        return JudgeResult(None, "no camera frames captured")

    try:
        api_key = get_runpod_api_key()
    except Exception as e:
        reason = f"could not get a Runpod API key: {e}"
        typer.echo(f"⚠️  VLM judge: {reason}")
        return JudgeResult(None, reason)

    content = [{"type": "text", "text": _build_judge_prompt_text(task_description, len(frames))}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_frame_to_base64_jpeg(frame)}"},
            }
        )

    payload = {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "model": _RUNPOD_JUDGE_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 200,
            },
        }
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(
            f"{_RUNPOD_API_BASE}/{RUNPOD_JUDGE_ENDPOINT_ID}/runsync",
            json=payload,
            headers=headers,
            # Runpod holds a /runsync request open server-side for a while before
            # falling back to an IN_PROGRESS/IN_QUEUE response for the poll loop
            # below to pick up - 60s was shorter than that hold, so a real cold
            # start (measured ~190s) killed the connection locally before Runpod
            # ever got to respond, never reaching the poll loop at all.
            timeout=100,
        )
        resp.raise_for_status()
        result = resp.json()

        job_id = result.get("id")
        deadline = time.monotonic() + _RUNPOD_POLL_TIMEOUT_S
        while result.get("status") in ("IN_QUEUE", "IN_PROGRESS") and time.monotonic() < deadline:
            time.sleep(_RUNPOD_POLL_INTERVAL_S)
            status_resp = requests.get(
                f"{_RUNPOD_API_BASE}/{RUNPOD_JUDGE_ENDPOINT_ID}/status/{job_id}",
                headers=headers,
                timeout=30,
            )
            status_resp.raise_for_status()
            result = status_resp.json()

        if result.get("status") != "COMPLETED":
            reason = f"Runpod job did not complete in time (status={result.get('status')})"
            typer.echo(f"⚠️  VLM judge: {reason}.")
            return JudgeResult(None, reason)

        verdict_text = result["output"][0]["choices"][0]["message"]["content"]
        return _parse_verdict(verdict_text)

    except Exception as e:
        reason = f"Runpod call failed: {e}"
        typer.echo(f"⚠️  VLM judge: {reason}")
        return JudgeResult(None, reason)


def judge_episode(frames: list, task_description: str = DEFAULT_JUDGE_TASK_DESCRIPTION) -> JudgeResult:
    """
    Judge whether a sequence of camera frames (list of (H, W, 3) uint8 numpy
    arrays, in chronological order) shows a valid completion of
    `task_description`.

    Returns a JudgeResult: `.verdict` is True (valid), False (invalid), or
    None if the judge call itself failed (missing dependency, bad key,
    network error, endpoint cold-start exceeded the poll timeout, etc.);
    `.reason` is always a short human-readable explanation - the model's own
    stated reasoning on a real verdict, or what went wrong otherwise. This is
    a strict data-quality gate for perturbation-augmented training data:
    callers should fail CLOSED and DISCARD the episode when verdict is None,
    the same as an explicit False verdict - an episode that cannot be
    verified is treated as not trustworthy enough to keep, not defaulted to
    "probably fine."

    Backend is local Ollama by default (free, no cold start, runs on this
    machine); set VLM_JUDGE_BACKEND=runpod for the Qwen2.5-VL Runpod
    serverless path.
    """
    backend = os.environ.get("VLM_JUDGE_BACKEND", "ollama").strip().lower()
    if backend == "runpod":
        return _judge_episode_runpod(frames, task_description)
    return _judge_episode_ollama(frames, task_description)
