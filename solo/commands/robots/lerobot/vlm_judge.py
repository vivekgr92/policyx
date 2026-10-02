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
import re
import subprocess
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
# qwen3-vl:4b-instruct chosen as the final pick over qwen2.5vl:7b (same
# Qwen-VL lineage already proven on the Runpod backend, but 7B's ~5-9GB real
# runtime footprint leaves uncomfortably little headroom on this machine's
# real hardware - confirmed via sysctl/system_profiler: Mac mini, Apple M4,
# 16GB unified memory) and over llava:7b (older architecture, no longer the
# best fit now that Ollama has first-class Qwen3-VL support).
#
# Specifically the "-instruct" tag, NOT the bare "qwen3-vl:4b" tag used
# originally - real root cause found and verified empirically: the bare "4b"
# tag's Modelfile (`ollama show qwen3-vl:4b --modelfile`) sets
# `RENDERER/PARSER qwen3-vl-thinking`, meaning it ALWAYS generates a
# `<think>...</think>` reasoning block regardless of the `think: false`
# request option or a "/no_think" prompt directive - tested directly: even a
# trivial "say hello" prompt produced 193 real eval tokens of genuine
# reasoning in the `thinking` field with both suppression attempts in place.
# This isn't "probabilistic non-compliance" as initially suspected, it's
# architectural - Ollama's library ships separate "-thinking" and
# "-instruct" tags per size specifically because a request-time flag can't
# reliably override a model built around always reasoning first. Verified
# the "-instruct" tag has NO `thinking` field in its response at all (tested
# directly) - a real, deterministic fix instead of trying to suppress
# thinking after the fact. Override via OLLAMA_JUDGE_MODEL if a different
# local model fits better in practice.
_OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
_OLLAMA_JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "qwen3-vl:4b-instruct")

# Ollama's default context window (4096 tokens) is too small for a multi-frame
# judge request - must match EXACTLY between the preload call and every real
# judge call, since loading a model with one num_ctx then requesting a
# different one forces an actual reload (verified empirically: a preload
# without this option dropped the already-loaded 32768-context/8.0GB instance
# down to a 4096-context/3.5GB one, which the next real call would then have
# had to reload again from scratch - completely defeating the point of
# preloading). See _judge_episode_ollama()'s real error for why 32768
# specifically: 5 real frames alone measured at 5628 tokens.
_OLLAMA_NUM_CTX = 32768
# How long Ollama keeps the model loaded after the last request specifying
# this. Applied to the preload call AND every real judge call (sliding
# window), so a gap between episodes longer than Ollama's 5-minute default
# doesn't silently evict the model mid-session.
_OLLAMA_KEEP_ALIVE = "30m"

# Runpod serverless backend (opt-in) - see module docstring for cost/rationale.
RUNPOD_JUDGE_ENDPOINT_ID = os.environ.get("RUNPOD_JUDGE_ENDPOINT_ID", "lt8yd7ssvip3y9")
_RUNPOD_JUDGE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
_RUNPOD_API_BASE = "https://api.runpod.ai/v2"
_RUNPOD_POLL_INTERVAL_S = 3.0
_RUNPOD_POLL_TIMEOUT_S = 300.0  # real measured cold start: 189s delay + ~1s exec (see docstring below)

# Frame-selection strategy, redesigned around what the rubric actually needs
# to see rather than generic uniform coverage (see vlm_judge_rules.md): most
# criteria - gripper empty, object upright, no drop-and-abandon - are END-
# STATE checks that only need the final frame(s). "Task completion" itself
# needs a START frame too, to compare against (you can't tell an object
# "moved from A to B" by only looking at the end). The one thing that
# genuinely benefits from a middle frame is catching a transient collision or
# a perturbation-induced stumble that happened to recover by luck before the
# end - real but lower-priority, kept as a small safety net rather than the
# main sampling strategy. This both concentrates frames on what the rubric
# needs (likely improving accuracy) and reduces real total frame count below
# the previous 8-frame baseline-evenly-spaced-across-the-whole-episode
# approach, which directly shrinks prompt-processing time and therefore
# timeout exposure (see _OLLAMA_SECONDS_PER_FRAME_ESTIMATE below) as a side
# benefit, not just an accuracy change.
START_FRAME_COUNT = 1  # the episode's start - needed to judge "did it actually move"
END_FRAME_COUNT = 3  # the episode's last few frames - covers gripper/upright/drop-abandon checks
# with a little robustness against the single last frame happening to be a
# transient motion-blur moment right as the gripper releases
MIDDLE_PERTURBATION_FRAMES = 2  # highest-realized-perturbation-magnitude frames from the middle, as a collision/stumble safety net
MAX_TOTAL_JUDGE_FRAMES = 6  # hard cap - lower than before since this strategy needs fewer frames to cover the same rubric


def select_judge_frame_indices(
    num_frames: int,
    perturbation_magnitudes: Optional[list] = None,
    start_count: int = START_FRAME_COUNT,
    end_count: int = END_FRAME_COUNT,
    middle_top_k: int = MIDDLE_PERTURBATION_FRAMES,
    max_total: int = MAX_TOTAL_JUDGE_FRAMES,
) -> list:
    """Pick frame indices to send to the judge: `start_count` frames from the
    very start (for "did it actually move" comparison), `end_count` frames
    from the very end (covers the end-state rubric checks - this also
    guarantees the true final frame is always included, unlike an evenly-
    spaced baseline that could drop it when competing for a small budget),
    plus - if per-step realized perturbation magnitudes are known -
    `middle_top_k` highest-magnitude frames from the remaining middle of the
    episode, as a safety net for a transient mid-episode collision/stumble
    that recovered by luck before the end. Capped at `max_total` total.

    `perturbation_magnitudes`, when given, must have length `num_frames` - one
    scalar per frame, the realized per-step perturbation (not the constant
    target fraction, which is the same for a whole episode and so useless for
    localizing where in the episode perturbation actually hit hardest).
    """
    if num_frames <= 0:
        return []
    if num_frames <= start_count + end_count:
        return list(range(num_frames))

    selected = set(range(min(start_count, num_frames)))
    selected |= set(range(max(0, num_frames - end_count), num_frames))

    if perturbation_magnitudes and len(perturbation_magnitudes) == num_frames and middle_top_k > 0:
        middle_candidates = [i for i in range(num_frames) if i not in selected]
        ranked = sorted(middle_candidates, key=lambda i: perturbation_magnitudes[i], reverse=True)
        for idx in ranked[:middle_top_k]:
            if len(selected) >= max_total:
                break
            selected.add(idx)

    if len(selected) > max_total:
        # Over budget even before adding perturbation frames (tiny max_total) -
        # keep the TRUE final frame (num_frames - 1) above everything else,
        # since that's the one frame several rubric criteria absolutely need
        # and must never be the one trimmed away; then the rest of the end
        # frames closest-to-last first, then the start frame(s) last.
        last_index = num_frames - 1
        ordered = sorted(
            selected,
            key=lambda i: (0 if i == last_index else (1 if i >= num_frames - end_count else 2), -i),
        )
        selected = set(ordered[:max_total])

    return sorted(selected)


def _frame_to_base64_jpeg(frame) -> str:
    """Encode a (H, W, 3) uint8 RGB numpy array as a base64 JPEG string."""
    from PIL import Image

    image = Image.fromarray(frame.astype("uint8"))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# User-editable judging criteria, loaded fresh per call (not cached) so edits
# take effect on the next judge call with no restart needed. Path is next to
# this file, not under ~/.solo - this is judging logic shipped with the repo,
# not per-user runtime config, but still meant to be hand-edited in place.
_JUDGE_RULES_PATH = os.path.join(os.path.dirname(__file__), "vlm_judge_rules.md")


def _load_judge_rules() -> str:
    """Best-effort: missing/unreadable rules file degrades to task-description-
    only judging rather than failing the whole call."""
    try:
        with open(_JUDGE_RULES_PATH, "r") as f:
            return f.read().strip()
    except Exception:
        return ""


def _extract_rule_criterion_names(rules_text: str) -> list:
    """Pull criterion names straight out of the rules file's own bold
    (**Name**) headers, so editing vlm_judge_rules.md to add/remove/rename a
    criterion automatically updates both what the prompt asks the model to
    check and what the checklist parser below looks for - no separate
    hardcoded criteria list to drift out of sync with the user-editable file."""
    return re.findall(r"\*\*(.+?)\*\*", rules_text)


def _build_judge_prompt_text(task_description: str, num_frames: int) -> str:
    # NOTE on thinking suppression: this used to need a trailing "/no_think"
    # directive here, because the original "qwen3-vl:4b" tag ALWAYS reasons
    # first regardless of request-time flags (see _OLLAMA_JUDGE_MODEL's real
    # root-cause comment - it's a different model build via Ollama's
    # "-thinking" tag, not a flag that was being ignored). Switching to the
    # "qwen3-vl:4b-instruct" tag fixes this at the source - verified directly,
    # that tag's responses have no `thinking` field at all - so no prompt-side
    # workaround is needed here any more.
    rules_text = _load_judge_rules()
    rules_block = f"\n{rules_text}\n\n" if rules_text else "\n"
    criterion_names = _extract_rule_criterion_names(rules_text)

    # Asking for a second, structured section (after the required one-line
    # verdict) so a per-criterion checklist can be displayed in the terminal -
    # kept as a strict, exact-name-echo format rather than free-form JSON/etc.
    # since criteria names are pulled from the rules file itself, so this
    # never drifts out of sync with what's actually in it.
    checklist_instruction = ""
    if criterion_names:
        criteria_lines = "\n".join(f"- {name}" for name in criterion_names)
        checklist_instruction = (
            f"\nAfter that line, add a line \"CHECKLIST:\" followed by one line per "
            f"criterion below, each EXACTLY in the form \"<criterion name>: PASS\" "
            f"or \"<criterion name>: FAIL - <brief reason>\" - use the exact "
            f"criterion names given below, do not reword or abbreviate them:\n"
            f"{criteria_lines}\n"
        )

    return (
        f"You are judging a robot arm demonstration recorded for training data. "
        f"The task description is: \"{task_description}\".\n\n"
        f"The robot arm is set up with two zones on a table, A (left side) and B "
        f"(right side). The task involves picking an object and moving it between "
        f"these zones as described.\n"
        f"{rules_block}"
        f"Below are {num_frames} frames from the episode, in chronological order: "
        f"starting frame(s) first, then (if a perturbation-notable moment occurred "
        f"mid-episode) one from the middle, then the final frame(s) showing the "
        f"episode's end state last.\n\n"
        f"Look at how the scene changes from the first frame to the last. Judge "
        f"whether this episode shows a VALID, successful completion of the task "
        f"(the object was actually grasped, moved in the correct direction, and "
        f"released at the target, with none of the house rules above violated) "
        f"versus an INVALID one (dropped object, missed grasp, wrong direction, a "
        f"house-rule violation, or other failure - these can happen when the "
        f"recorded actions have been perturbed with random noise for data "
        f"augmentation).\n\n"
        f"Respond with EXACTLY one line starting with \"VALID\" or \"INVALID\", "
        f"followed by a dash and a ONE-SENTENCE reason - do not explain your "
        f"reasoning at length, a single short sentence is all that's needed. "
        f"Example:\n"
        f"VALID - the object moved from the left zone to the right zone and the "
        f"gripper is empty at the end.\n"
        f"{checklist_instruction}"
        f"Do not include anything else in your response."
    )


def _parse_checklist(verdict_text: str, criterion_names: list) -> Optional[list]:
    """Best-effort parse of the model's "CHECKLIST:" section into a list of
    (name, passed, reason) tuples, matched against the real criterion names
    from vlm_judge_rules.md (not a hardcoded list - see
    _extract_rule_criterion_names). Returns None (not a partial/garbled list)
    if there's no criteria to check, no CHECKLIST section in the response at
    all, or not one criterion name could be matched - callers should fall back
    to showing just the overall verdict+reason in that case, same as if this
    feature didn't exist, rather than render a confusing incomplete checklist."""
    if not criterion_names:
        return None
    marker_idx = verdict_text.upper().find("CHECKLIST")
    if marker_idx == -1:
        return None
    checklist_block = verdict_text[marker_idx:]

    results = []
    for name in criterion_names:
        match = re.search(
            re.escape(name) + r"\s*:\s*(PASS|FAIL)\b(?:\s*-\s*(.*))?",
            checklist_block,
            re.IGNORECASE,
        )
        if match:
            passed = match.group(1).upper() == "PASS"
            reason = (match.group(2) or "").strip()
            results.append((name, passed, reason))
    return results if results else None


def _render_checklist(checklist: list) -> None:
    typer.echo("📋 Rubric checklist:")
    for name, passed, reason in checklist:
        icon = "✅" if passed else "❌"
        suffix = f" - {reason}" if (not passed and reason) else ""
        typer.echo(f"   {icon} {name}{suffix}")


def _parse_verdict(verdict_text: str) -> JudgeResult:
    """Parse a model response of the form "VALID - <reason>" / "INVALID -
    <reason>" (optionally followed by a "CHECKLIST:" section - see
    _parse_checklist) into a JudgeResult, preserving the model's actual stated
    reason rather than discarding it. Only the first line is echoed/parsed as
    the verdict+reason - the checklist section, when present, is rendered
    separately by _render_checklist() so the main verdict line stays exactly
    as short as before this feature existed."""
    stripped = verdict_text.strip()
    first_line = stripped.splitlines()[0] if stripped else stripped
    typer.echo(f"🧑‍⚖️  VLM judge: {first_line}")

    upper = first_line.upper()
    reason = first_line
    if "-" in first_line:
        _, _, after_dash = first_line.partition("-")
        after_dash = after_dash.strip()
        if after_dash:
            reason = after_dash

    criterion_names = _extract_rule_criterion_names(_load_judge_rules())
    checklist = _parse_checklist(stripped, criterion_names)
    if checklist:
        _render_checklist(checklist)

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

# The real bug this fixes: Ollama's public streaming API emits ZERO bytes
# during prompt processing (image encoding + context building) - confirmed
# empirically, twice, with fresh/uncached frames: a real 15-frame request
# streamed NOTHING for 100.37s and 106.65s respectively before its first byte.
# A flat timeout short enough to catch a genuine mid-generation stall (90s)
# is too short to survive this silent phase once enough frames are involved -
# exactly what happened in the live bug report (a 15-frame call hit a 180s-era
# flat timeout while still only 85% through prompt eval). MAX_TOTAL_JUDGE_FRAMES
# was independently reduced to 8 specifically to shrink this phase, but this
# scales explicitly rather than assuming frame count always stays under
# whatever that cap happens to be set to.
#
# Deliberately a SIMPLE time-per-frame estimate, not a token-based one: a
# token-per-frame model was tried first and found unreliable - real testing
# showed 8 real frames from one actual recorded episode need ~16,600 prompt
# tokens (not the ~9,000 a flat "~1,100 tokens/frame" estimate predicted from
# an earlier, different episode's frames), because token count depends on
# real image content/complexity, not just frame count. A generous flat
# per-frame time allowance, calibrated against the worst real observation so
# far (8 frames still only 86% through prompt eval at 98.4s, projecting
# ~115s+ just for prompt eval to finish), is simpler and more robust against
# this content-dependent variance than trying to precisely predict token
# counts. `requests` applies a single read timeout uniformly across an
# entire streamed response (no public way to use a shorter value once
# generation starts and a longer one during prompt eval), so this same
# number also becomes the inter-chunk stall-detection window during
# generation - looser stall detection there in exchange for not prematurely
# killing a call that's still legitimately encoding frames, the same
# fail-closed-favors-correctness tradeoff already made elsewhere in this file.
_OLLAMA_SECONDS_PER_FRAME_ESTIMATE = 35.0
_OLLAMA_MIN_REQUEST_TIMEOUT_S = 250.0  # floor for even a single frame, real margin over observed generation-time variance

# Ollama's own server log (real install-default path for the macOS app -
# confirmed present and actively written on this machine; NOT a stable public
# API, path/format could differ across installs/versions). Used only for a
# best-effort live progress display during the silent prompt-processing
# phase described above - any failure to find/read/parse it is swallowed and
# never affects the real judge call.
_OLLAMA_LOG_PATH = os.path.expanduser(os.environ.get("OLLAMA_LOG_PATH", "~/.ollama/logs/server.log"))


def _estimate_ollama_request_timeout_s(num_frames: int) -> float:
    """A generous read-timeout covering the worst-case prompt-processing
    duration for `num_frames` frames - see the constants above this function
    for the real measured basis and why a single value has to cover both the
    prompt-eval and generation phases."""
    return max(_OLLAMA_MIN_REQUEST_TIMEOUT_S, num_frames * _OLLAMA_SECONDS_PER_FRAME_ESTIMATE)


def _start_prompt_progress_tail():
    """Best-effort live display of Ollama's own real prompt-processing
    progress (n_tokens / percent / tokens-per-sec) during the phase where the
    public streaming API itself emits nothing at all (see the timeout
    constants above - confirmed empirically). Tails `_OLLAMA_LOG_PATH` and
    lets matching lines print straight to this terminal.

    Deliberately a genuine OS subprocess, not a Python thread: a
    threading.Thread doing time.sleep()-based polling alongside the main
    thread's blocking HTTP read was tried first and found UNRELIABLE on this
    machine - tested directly: the watchdog thread's own sleep() calls were
    observed to stall for 100+ seconds right alongside the main thread's
    blocked socket read, rather than firing on schedule, even though
    CPython's socket module is documented to release the GIL during blocking
    reads. A real subprocess shares no interpreter/GIL with the request at
    all, so its output streams to the terminal completely independently of
    whatever the main thread is doing - verified empirically to actually work
    where the threading approach did not.

    Returns None (silently) if the log doesn't exist or the subprocess can't
    be started - this is a display enhancement only, never load-bearing."""
    if not os.path.exists(_OLLAMA_LOG_PATH):
        return None
    try:
        # start_new_session=True puts `bash` (and the `tail`/`grep` it forks
        # for the pipeline) in their own process group - required for
        # _stop_prompt_progress_tail() to actually kill the whole pipeline.
        # Verified empirically this was necessary: a plain proc.terminate()
        # only signals the top-level `bash` process, leaving the piped
        # `tail`/`grep` children orphaned and still running indefinitely.
        return subprocess.Popen(
            [
                "bash", "-c",
                f'tail -f -n0 "{_OLLAMA_LOG_PATH}" | grep --line-buffered "prompt processing"',
            ],
            stdout=None,  # inherited - writes straight to this terminal, no Python-side reading needed
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return None


def _stop_prompt_progress_tail(proc) -> None:
    if proc is None:
        return
    import os as _os
    import signal as _signal

    try:
        _os.killpg(_os.getpgid(proc.pid), _signal.SIGTERM)
        proc.wait(timeout=2)
    except Exception:
        try:
            _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
        except Exception:
            pass


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
        "think": False,  # harmless no-op on the "-instruct" tag (it has no thinking
        # mode to disable); kept in case OLLAMA_JUDGE_MODEL is overridden to a
        # "-thinking" tag, where this request option genuinely has no effect either
        # (see _OLLAMA_JUDGE_MODEL's comment) - the real fix is the model tag itself.
        "options": {"num_ctx": _OLLAMA_NUM_CTX},
        "keep_alive": _OLLAMA_KEEP_ALIVE,
    }

    request_timeout = _estimate_ollama_request_timeout_s(len(frames))
    tail_proc = _start_prompt_progress_tail()
    if tail_proc is not None:
        typer.echo("🔎 Prompt processing (live Ollama progress below)...")
    try:
        resp = requests.post(
            f"{_OLLAMA_API_BASE}/api/chat",
            json=payload,
            timeout=request_timeout,
            stream=True,
        )
        resp.raise_for_status()

        content_parts = []
        final_chunk = None
        # Deliberately NOT echoing msg["thinking"] live (an earlier version did)
        # - the user found a full live thinking-stream too noisy and asked for
        # a single final summary instead. The progress bar alone remains as the
        # "is it still working" signal during generation; the real verdict+
        # reason prints once, after the call completes, via _parse_verdict().
        with tqdm(total=None, desc="Generating", unit="chunk") as bar:
            for line in resp.iter_lines():
                if not line:
                    continue
                if tail_proc is not None:
                    # First real byte means generation has started - the
                    # silent prompt-processing phase this was watching is over.
                    _stop_prompt_progress_tail(tail_proc)
                    tail_proc = None
                chunk = json.loads(line)
                bar.update(1)
                content_delta = chunk.get("message", {}).get("content") or ""
                if content_delta:
                    content_parts.append(content_delta)
                if chunk.get("done"):
                    final_chunk = chunk
                    break

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
            f"Ollama produced no streamed output for {request_timeout:.0f}s "
            f"(scaled for {len(frames)} frames) - likely genuinely stuck, not "
            f"just slow prompt processing under real system load"
        )
        typer.echo(f"⚠️  VLM judge: {reason}")
        return JudgeResult(None, reason)
    except Exception as e:
        reason = f"Ollama call failed: {e}"
        typer.echo(f"⚠️  VLM judge: {reason}")
        return JudgeResult(None, reason)
    finally:
        _stop_prompt_progress_tail(tail_proc)


def preload_ollama_judge_model() -> bool:
    """Proactively load the Ollama judge model into memory once, at the start
    of a replay session, instead of paying the cold-load cost on whichever
    episode happens to be judged first. Uses Ollama's documented preload
    pattern - a chat request with an empty `messages` list triggers a model
    load with no real generation (confirmed live: real response carries
    `"done_reason": "load"`) - with the EXACT SAME `options`/`keep_alive` the
    real judge calls use, since a mismatch forces a real reload on the first
    real call anyway (see _OLLAMA_NUM_CTX's comment - verified empirically,
    not assumed).

    Non-fatal if it fails: returns False and lets the first real judge call
    pay the cold-load cost itself, same as before this existed. Call this at
    most once per `solo robo --replay` process invocation, before the
    per-episode loop starts - not per-episode."""
    typer.echo(f"🔥 Preloading {_OLLAMA_JUDGE_MODEL} into memory...")
    try:
        resp = requests.post(
            f"{_OLLAMA_API_BASE}/api/chat",
            json={
                "model": _OLLAMA_JUDGE_MODEL,
                "messages": [],
                "options": {"num_ctx": _OLLAMA_NUM_CTX},
                "keep_alive": _OLLAMA_KEEP_ALIVE,
            },
            timeout=300,  # a genuine cold load of a multi-GB model can take a while
        )
        resp.raise_for_status()
        typer.echo(f"✅ {_OLLAMA_JUDGE_MODEL} loaded and will stay warm for {_OLLAMA_KEEP_ALIVE}.")
        return True
    except Exception as e:
        typer.echo(
            f"⚠️  Could not preload {_OLLAMA_JUDGE_MODEL} ({e}) - the first judged "
            f"episode will pay the load cost instead."
        )
        return False


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
                "max_tokens": 300,  # bumped from 200 to leave room for the
                # per-criterion CHECKLIST section now also requested
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
