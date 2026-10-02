"""
VLM-based episode quality judge for replay-generated training data.

Given a handful of frames sampled from a replayed episode plus a task
description, asks a vision-capable Claude model whether the episode shows a
valid completion of that task - used to filter out perturbation-induced
failures (dropped object, missed grasp, wrong direction) before they're saved
into a training dataset.
"""

import base64
import json
import os
from io import BytesIO
from typing import Optional

import typer
from rich.prompt import Prompt

from solo.config import CONFIG_PATH

DEFAULT_JUDGE_TASK_DESCRIPTION = "Pick cup and place"
_JUDGE_MODEL = "claude-sonnet-5"

# Frame-selection tuning. No real failure-case data has validated these numbers
# yet (the zero-shot prototype only had clean-success episodes to test against)
# - adjust once real perturbation-induced failures have been judged and checked
# against what a human would've caught.
BASELINE_FRAME_COUNT = 10  # evenly-spaced frames across the whole episode, for general coverage
TOP_PERTURBATION_FRAMES = 5  # highest-realized-perturbation-magnitude frames to zoom in on
PERTURBATION_CONTEXT_WINDOW = 1  # also include this many frames before/after each top-perturbation frame
MAX_TOTAL_JUDGE_FRAMES = 25  # hard cap on total frames sent to the judge per episode (cost/latency)


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


def get_anthropic_api_key() -> str:
    """Get the Anthropic API key from env, saved config, or prompt for it once."""
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    config = _load_config()
    key = config.get("anthropic", {}).get("api_key")
    if key:
        return key

    typer.echo("\n🔑 An Anthropic API key is required for the VLM judge.")
    typer.echo("   Create one at https://console.anthropic.com/settings/keys")
    key = Prompt.ask("Enter your Anthropic API key")
    config.setdefault("anthropic", {})["api_key"] = key
    _save_config(config)
    return key


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


def judge_episode(frames: list, task_description: str = DEFAULT_JUDGE_TASK_DESCRIPTION) -> Optional[bool]:
    """
    Judge whether a sequence of camera frames (list of (H, W, 3) uint8 numpy
    arrays, in chronological order) shows a valid completion of
    `task_description`.

    Returns True (valid), False (invalid), or None if the judge call itself
    failed (missing dependency, bad key, network error, etc.) - callers
    should default to KEEPING the episode on None rather than discarding
    potentially-good data over a transient failure.
    """
    try:
        import anthropic
    except ImportError:
        typer.echo("⚠️  VLM judge: the 'anthropic' package is not installed (pip install anthropic). Keeping episode.")
        return None

    if not frames:
        typer.echo("⚠️  VLM judge: no frames to judge. Keeping episode.")
        return None

    try:
        api_key = get_anthropic_api_key()
    except Exception as e:
        typer.echo(f"⚠️  VLM judge: could not get an Anthropic API key ({e}). Keeping episode.")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)

        content = [
            {
                "type": "text",
                "text": (
                    f"You are judging a robot arm demonstration recorded for training data. "
                    f"The task description is: \"{task_description}\".\n\n"
                    f"The robot arm is set up with two zones on a table, A (left side) and B "
                    f"(right side). The task involves picking an object and moving it between "
                    f"these zones as described.\n\n"
                    f"Below are {len(frames)} frames sampled evenly across the episode, in "
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
                ),
            }
        ]
        for frame in frames:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _frame_to_base64_jpeg(frame),
                    },
                }
            )

        response = client.messages.create(
            model=_JUDGE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": content}],
        )
        verdict_text = response.content[0].text.strip()
        typer.echo(f"🧑‍⚖️  VLM judge: {verdict_text}")

        upper = verdict_text.upper()
        if upper.startswith("VALID"):
            return True
        elif upper.startswith("INVALID"):
            return False
        else:
            typer.echo("⚠️  VLM judge: unexpected response format, keeping episode.")
            return None

    except Exception as e:
        typer.echo(f"⚠️  VLM judge: API call failed ({e}). Keeping episode.")
        return None
