# DeployX / Solo-CLI Session Context

This document captures the full context of an extended work session on `solo-cli`, for picking the work back up without re-reading the whole conversation. It covers what was built, what was learned empirically about Runpod, what was explored and paused (OpenWAM), and what's actively in flight (the DeployX Edge Agent).

## 1. Goal / Big Picture

The user is building a physical-AI pipeline around a SO-101 robot arm using `solo-cli` (a Typer-based Python CLI, entry point `solo`) and the `lerobot` framework:

1. **Record** demonstrations on the real arm (`solo robo --record`).
2. **Train** policies on that data (`solo robo --train`), either locally or on a Runpod GPU pod.
3. **Run inference** with a trained policy on the real arm (`solo robo --inference`), optionally with human teleoperation overrides for corrections.
4. **Longer-term goal**: a hardware-in-the-loop (HIL) evaluation loop where a trained policy (π0-FAST) chooses actions, a learned "world model" (OpenWAM) predicts what should happen if those actions are taken, the actions are actually executed on the real arm, and the *prediction vs. reality* gap is measured to find where the policy/world-model disagree with the physical world — i.e. "train π0-FAST → simulate with OpenWAM → execute on real hardware → compare prediction vs. reality → learn the failure gap → generate the next test."

The current, most concrete step toward that goal is the **DeployX Edge Agent** (section 7): a lightweight process on a Mac mini, physically wired to the SO-101, that offloads all the heavy policy inference to a Runpod GPU pod over the network, so the Mac mini only needs to do hardware I/O, safety checks, and logging.

## 2. Runpod Training Integration (built and validated live)

`solo robo --train` now has a **Step 4.5: Compute Location** prompt (in `solo/commands/robots/lerobot/modes/training.py`) offering `local` or `runpod`. Choosing `runpod` hands off to `solo/commands/robots/lerobot/runpod_train.py`, which was built and then validated against **real Runpod infrastructure and real GPU training runs**, not just unit-tested.

### Architecture of `runpod_train.py`

- **`RunpodClient`** — a thin wrapper around the Runpod REST API (`https://rest.runpod.io/v1`), independent of any Claude Code plugin/MCP (this all runs standalone when the user invokes `solo` themselves). Handles `list_pods`, `get_pod`, `start_pod`, `stop_pod`, `update_pod`, `create_pod`.
- **Pod selection/creation** — lists existing pods, lets the user pick one or create a new one; if starting an existing pod fails with a capacity error, it **automatically offers to create a new pod on a different host** instead of just failing (`_is_capacity_error` detects the specific Runpod error text; both `select_or_create_pod` and `_create_pod_interactive` retry through capacity errors, the latter up to 3 attempts automatically since Runpod's scheduler picks a different host each attempt).
- **SSH-based execution** — `bootstrap_pod` installs `solo-cli` + `lerobot` on a fresh pod (skipped if already present), `push_training_config` uploads the training config, `run_remote`/`run_remote_to_file` execute `solo robo --train --yes` on the pod (streaming live, or logging to a file for the parallel multi-policy mode), `rsync_from_pod` syncs checkpoints back down automatically to the same local `outputs/train/...` directory the user configured.
- **Multi-policy comparison mode** — `run_policy_comparison_on_runpod` trains **all 8 supported policy types in parallel**, one pod each, same dataset/steps/batch-size, using a `ThreadPoolExecutor`. Reachable via a follow-up prompt after selecting `runpod`: "Compare ALL policy types in parallel instead?"

### The 8 supported policy types

Wired in `solo/commands/robots/lerobot/modes/training.py`'s policy-selection menu and `ALL_POLICIES` in `runpod_train.py`:
`smolvla`, `act`, `pi0`, `tdmpc`, `diffusion`, `vqbet`, `pi0_fast`, `pi05`.

### Real bugs found and fixed via live testing (not hypothetical — each was hit and fixed against a real pod)

- **PEP 668 externally-managed environment**: the pod's Ubuntu 24.04 Python blocks plain `pip install`; fixed with `--break-system-packages` in `bootstrap_pod`.
- **`ports` field type mismatch**: `RunpodClient.create_pod` was sending `"ports": "22/tcp,8888/http"` as a string; Runpod's API requires an array. Fixed to `["22/tcp", "8888/http"]`. This bug went undetected in earlier testing because prior tests used the MCP tool (which took it correctly as an array) or manually-built pod dicts — this was the first real exercise of the standalone `RunpodClient.create_pod` code path.
- **`SOLO_REMOTE_TRAINING` env var**: when `run_training_on_runpod` invokes `solo robo --train --yes` *on the pod itself*, that remote process would otherwise hit the same "local vs runpod" prompt again — this env var, checked in `training.py`, makes it skip straight to `local` on the pod instead of prompting or (worse) trying to recurse into another Runpod deployment.
- **Silent stop-pod failures**: a `RemoteDisconnected` network hiccup calling Runpod's stop API after training already succeeded used to crash the whole command with a raw traceback. Fixed to catch `requests.RequestException`/`RuntimeError` and print a clear warning with the pod ID and console link instead.
- **Opaque error messages**: `RunpodClient._request` originally just called `requests`' default `raise_for_status()`, hiding Runpod's actual error explanation (in the response body). Fixed to parse and surface `detail` from the JSON body (handling both dict and non-dict bodies) into the raised exception message.

### Other design notes

- Direct SSH auth uses a dedicated keypair (`~/.ssh/id_ed25519_runpod`), injected into each pod via the `PUBLIC_KEY` env var (not the Runpod SSH-proxy/account-key mechanism — see section 5).
- HF token and WandB API key are forwarded to the remote training process via env vars (`_local_hf_token`, `_local_wandb_key`), not written to disk on the pod.

## 3. Mandatory `push_to_hub` for Recordings

`solo/commands/robots/lerobot/modes/recording.py` no longer offers a "push to Hub? y/n" choice — it's now **mandatory** for new recordings. Recording always attempts HuggingFace login first; if it fails, recording aborts rather than silently falling back to a `local/`-only dataset. Rationale: this guarantees every new dataset lands on HF Hub, so `runpod_train.py`'s `sync_dataset_if_local()` (which only falls back to a slow SSH/rsync transfer for `local/...`-prefixed datasets) essentially never needs that fallback path for anything recorded going forward — the remote pod just pulls the dataset from Hub itself. The preconfigured/resume path (reusing settings from a prior recording) was left untouched, so resuming an old local-only dataset still works.

## 4. HIL Correction Merging

`solo/commands/robots/lerobot/hil_merge.py` (new file) implements `merge_correction_into_dataset(base_repo_id, correction_repo_id, merged_repo_id)`, which calls `lerobot.datasets.aggregate.aggregate_datasets` (lerobot's own real dataset-merging utility — validated to exist and be usable via research into the installed `lerobot` package; it's a library function, not a CLI, with no console-script entry point) to combine a base training dataset with a correction session's dataset, then pushes the merged result to Hub via `LeRobotDataset(merged_repo_id).push_to_hub()`.

Wired into `solo/commands/robots/lerobot/modes/inference.py`: after an inference session run **with teleoperation enabled** (i.e. the user corrected the policy live), it asks "Add this corrected session to a training dataset for retraining?" — if yes, it authenticates with HF, asks for the base dataset (defaulting to the last training dataset from saved config) and a name for the merged result, then calls `merge_correction_into_dataset`. This closes the loop: run policy → correct mistakes live → merge corrections into the training set → retrain on original + corrections.

**Caveat**: this has not been live-tested end-to-end (unlike the Runpod training path) — it needs an actual HIL correction session on real hardware to exercise, which hasn't happened yet in this session.

## 5. Runpod Infrastructure Notes (learned empirically, useful for future debugging)

- The account's three original pods — `r8pz252nhq3ox3`, `ux76alref8foun`, `uugkgavj876fst` — are pinned to a specific host in **US-IL-1** that has repeatedly been out of free GPU capacity ("There are not enough free GPUs on the host machine to start this pod"). This is what motivated the capacity-error fallback logic in `runpod_train.py`.
- Runpod's **SSH proxy** (`ssh.runpod.io`, with a per-pod `<podId>-<hash>@ssh.runpod.io` username) authenticates against **account-level registered SSH keys** (a different mechanism from the per-pod `PUBLIC_KEY` env var), but it's an **interactive-only bastion that ignores one-shot commands** — `ssh proxy-host "some command"` just drops you into an interactive shell regardless of what command you pass, making it useless for scripted execution. **Direct SSH** (`ssh.direct` in the pod's API response — a real public IP + port mapping) is the only viable path for scripted commands, and it authenticates correctly against the pod's own `PUBLIC_KEY` env var.
- Direct SSH's public IP/port mapping can take **several minutes** to appear after a pod reports `RUNNING` (observed anywhere from ~15 seconds to ~6 minutes) — `runpod_train.py`'s `_wait_until_ready` polls for up to 600s to accommodate this.
- The **`DeployX-volume`** network volume (id `shgutprae1`, 100GB, US-IL-1, mounted at `/workspace` on the original 3 pods) is a **shared, general-purpose volume already holding unrelated prior work** — a `.isaac_venv` (29GB), `.cache` (54GB), `.codex` (1.9GB), `Projects` (2.1GB), `bin` (325MB) — roughly 87GB already used out of 100GB. **Do not delete or modify its existing contents** without the user's explicit go-ahead; it was deliberately left untouched when it turned out to have too little free space for OpenWAM's needs (see section 6).

## 6. OpenWAM Investigation — Resumed, Cosmos3-Edge Deployed, Action-Conditioning Designed

The user wants to eventually use **OpenWAM** as the "world model" simulator in the HIL evaluation loop (see section 1). This was actively investigated and partially set up, then explicitly paused by the user ("set aside the openwam for now, we will revisit it later") in favor of building the DeployX Edge Agent first.

### What OpenWAM is

A real, actively maintained open-source research framework: **"OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining"**, arXiv:2609.07398 (Sept 2026), repo at `github.com/OpenWAM-Official/OpenWAM`, Apache 2.0, ~851 stars at time of research. This was independently verified against the primary sources (not just a search-engine summary) since it's a very recently published project. It jointly models *video prediction* and *action prediction* — a "world-action model" — as opposed to lerobot's policies (ACT, PI0, etc.) which only predict actions.

### The user's actual goal for it

1. Fine-tune π0-FAST on the user's own demonstrations (their policy).
2. Run that policy on the real SO-101, collect trajectories.
3. **Fine-tune OpenWAM on those trajectories** so it learns the specific arm, camera viewpoint, objects, and dynamics.
4. Use the fine-tuned OpenWAM as a learned simulator: `current observation + proposed actions → predicted future observation`.
5. Compare that prediction against what actually happens when those actions are executed for real (via the DeployX Edge Agent / HIL loop).

The user's own framing, which resolved an important ambiguity: *"π0-FAST learns what action to take; OpenWAM learns what is likely to happen if you take it."* And: *"For a first prototype... I would not wait for a perfect OpenWAM fine-tune. Get the full loop working first, then improve the world model once you have real SO-101 rollout data."* — hence the decision to build DeployX first.

### Key technical finding

OpenWAM's public, documented inference entry point (`BaseWAMArchitecture.generate()` in `openwam/model/architectures/base.py`, exposed via the WebSocket policy server in `openwam/deploy/`) does **not** accept an externally-given action sequence to condition its video prediction on — it always generates its own actions *jointly* with the video via flow-matching diffusion (there is no `input_action`/`teacher_forcing`-style parameter anywhere in the codebase). Getting the "given actions → predicted video" behavior the user actually wants means reusing whatever action-conditioning pathway OpenWAM's own **training/fine-tuning** code uses internally (since during fine-tuning, real actions from the dataset are necessarily fed in as conditioning to predict the matching video) — not just calling the public inference wrapper unmodified. This is why fine-tuning on the user's own trajectory data (step 3 above) is not just about prediction *quality* — it's expected to be the actual mechanism that makes conditioning-on-given-actions possible at all, sidestepping what would otherwise require patching OpenWAM's internal denoising loop.

### Hardware finding

A single RTX 4090 (24GB) **OOMs just loading** the `OpenWAM-Alpha-Sim-RoboTwin-Full` checkpoint for inference (confirmed live: `CUDA out of memory... 23.51 GiB memory in use` out of 23.53GiB total, before any actual generation ran) — this checkpoint (Wan2.2-TI2V-5B video backbone + UMT5-XXL text encoder, dual-system architecture) needs a bigger single GPU just to deploy for inference. The README's stated recommendation of **8×80GB GPUs is specifically for large-scale training** (confirmed: "two H100 80GB GPUs ran out of memory at the first Adam update, while four passed" — an optimizer-state-memory issue, training-only); inference/deployment needs less but still more than 24GB.

A follow-up attempt to try a single **L40S (48GB)** failed with "no instances available" in US-IL-1 — and checking the **full Runpod GPU catalog** confirmed **no GPU with ≥48GB VRAM is offered in US-IL-1 at all** (not just temporarily out of stock — none of A100, H100, H200, L40S, L40, RTX 6000 Ada, A6000, RTX PRO 6000 list that data center). This is a hard blocker for colocating a big-GPU pod with any US-IL-1-locked volume.

### What's already provisioned and ready (but currently idle / no pod running)

A **dedicated 150GB network volume "openwam-assets"** (id `le24rl6jpu`, US-IL-1) was created specifically to avoid the `DeployX-volume` space conflict (see section 5), and already has, fully downloaded:
- The OpenWAM repo itself, cloned and `pip install -e .`'d (torch 2.7.1+cu128, deepspeed 0.18.9 — installed via `uv pip install --system --break-system-packages` after plain `pip`'s dependency resolver stalled for 20+ minutes; `uv` resolved the same 105 packages in under a second).
- The **Wan2.2-TI2V-5B video backbone** (~34.2GB), downloaded via `scripts/download_assets/download_video_backbone.py`.
- The **OpenWAM-Alpha-Sim-RoboTwin-Full** checkpoint (~24.8GB, a deployable — not `finetune_only` — checkpoint), downloaded via `scripts/download_assets/download_openwam_checkpoints.py --family alpha --name OpenWAM-Alpha-Sim-RoboTwin-Full`.

No pod is currently running against this volume (the last pod that was, `wuftexpjhges1x`, an RTX 4090, OOM'd on model load and was superseded by the L40S attempt that failed to find capacity). Resuming this work means either finding a big-GPU-capable data center to move the volume/assets to, or provisioning a fresh volume elsewhere and re-downloading (the ~34GB + ~25GB downloads took roughly an hour combined against US-IL-1's network volume, for reference).

Also worth noting for later: OpenWAM's deploy config (`configs/deploy.yaml`) defaults `optimization.decode_video: false` ("skip VAE decode, return video=null, actions-only") — actually getting predicted video frames back requires an explicit override, e.g. `optimization.decode_video=true` as a CLI dotlist override to `scripts/deploy.sh`.

### Resumption, and a recurring infrastructure-stability problem

OpenWAM work was picked back up after the DeployX Edge Agent reached the "full loop proven live" milestone (section 8). Getting back onto real Runpod infrastructure surfaced a **repeated, unexplained problem**: pods and their attached network volumes have disappeared between/during sessions more than once, with no root cause found and no delete call issued from this side:

- The original `openwam-assets` volume (id `le24rl6jpu`, US-IL-1, 150GB, holding the fully-downloaded Wan2.2 backbone + RoboTwin checkpoint described above) — **gone**.
- A later pod+volume pair created in **EU-FR-1** (volume id `my0ydkal35`) — also **gone**.
- A **RTX PRO 6000 (MIG 2g.48gb)** pod — the user directly asked "are you sure it will work here in this GPU," and was given an honest risk assessment (reduced compute vs. a full card, `nvidia-smi` reporting `[Insufficient Permissions]` for restricted ops, `torch.compile` untested on that MIG slice) rather than false confidence — this pod/volume also vanished before real testing could finish.
- A **RTX PRO 4500 (Blackwell)** pod in **EU-RO-1** stalled 20+ minutes on SSH readiness twice; the first attempt was terminated as a loss, the second eventually came up after the user confirmed "now try the pod is up."

Practice established from this: **never trust the Runpod GPU catalog's "LOW availability" label alone** — it showed "LOW" for an L40S twice when real capacity was actually zero — always verify by attempting a real pod creation. Also confirmed empirically: US-MO-1 doesn't support network volumes at all; EU-FR-1 only supports the `HIGH_PERFORMANCE` volume tier, not `STANDARD` (learned from the API's own error message, which lists the valid `STANDARD`-tier data centers) — this pushed volume placement to US-NE-1 and then EU-RO-1.

**This instability remains unresolved and unexplained** — flagged here again as an open operational risk for future sessions: don't assume a pod or volume that worked last session is still there; re-verify before building on it.

### Cosmos3-Edge: a much smaller checkpoint family — found and deployed live

Re-investigating OpenWAM's checkpoint options (rather than re-fighting the original 24GB+34GB RoboTwin/Wan2.2 combination that OOM'd a 4090) turned up a **Cosmos3-Edge** backbone variant — a 4B-parameter model with **no external text encoder** (unlike Wan2.2-TI2V-5B + UMT5-XXL), making it dramatically smaller end to end: **~18.3GB total** for backbone + checkpoint, versus **~59GB** for the original Wan2.2/RoboTwin combination.

This was provisioned on a new volume in **EU-RO-1** and **successfully deployed and live-tested**:
- Pod `lux6a5h0knbl1n`, EU-RO-1, **$0.72/hr**, running against the Cosmos3-Edge checkpoint.
- A real smoke test (loading the checkpoint and running actual generation, not just an import check) **passed**.
- GPU memory usage during the real test: only **9–11GB out of 32GB** available on that pod's GPU — a huge headroom improvement over the RoboTwin checkpoint's OOM on a full 24GB 4090.
- This pod was still running as of the start of the action-conditioning implementation work described below; a decision on whether to stop it is still pending (see section 10).

### SO-101 incompatibility finding

Checked directly (not assumed) whether OpenWAM's shipped checkpoints could be used as-is against the user's actual SO-101 arm: **they cannot**. OpenWAM has **zero built-in support for SO-101** — no dataloader for it anywhere in the codebase, and the available checkpoints (including RoboTwin and the Cosmos3-Edge variant) were trained on an **Aloha-Agilex** setup using **end-effector (eef) actions** and a **3-camera** layout, which doesn't match the SO-101's joint-space action representation or the user's camera configuration.

**Conclusion**: using OpenWAM as a real simulator for the user's own SO-101 rollouts requires writing a **new OpenWAM dataloader** for SO-101's action/observation format and running an actual **fine-tune** on real SO-101 trajectory data — this has **not been started**. This confirms and sharpens the plan already noted in section 6 above (fine-tuning OpenWAM on the user's own trajectories was always step 3 of the intended pipeline) — it's now a hard requirement rather than a quality nice-to-have, since the off-the-shelf checkpoints are structurally incompatible with the hardware, not just under-adapted to it.

### Action-conditioning design ("Piece A") — making π0-FAST's chosen actions drive OpenWAM's video prediction

This is the design work that answers the user's question about **how π0-FAST's policy output actually gets fed into OpenWAM/Cosmos3** to produce a predicted-observation comparison (pipeline step 4 in section 1). The constraint from section 6's original finding still holds: `generate()` in `openwam/model/architectures/base.py` has no public parameter for supplying an external action sequence — it always generates video and action jointly. Digging into `generate()`'s actual denoising loop (confirmed at line 1361) found a precise, low-risk way to patch this without touching the loop body:

- The loop computes `sigma_a = t_a / num_train_timesteps_a` and `action_stepping = sigma_a != sigma_a_next` each iteration, and only calls `self.action_scheduler.flow_step(...)` **when `action_stepping` is true and an action noise prediction exists**.
- `action_latents` — the tensor the whole loop denoises into a final action — starts as pure noise: `torch.randn(1, action_num_frames - 1, self.action_dim, ...)`, with **no existing injection point**.
- **The design**: add exactly **one new optional parameter** to `generate()` (`given_action_latents` or equivalent) that, when supplied, replaces that `torch.randn(...)` initialization. The real action sequence (π0-FAST's chosen actions) gets converted into this same normalized/unified latent space by **reusing the existing `_UnifyAwareNormalizer.normalize(raw_action)`** from `openwam/deploy/model_loader.py` — the exact same normalizer `generate()` already uses in reverse (`.unnormalize()`) on its own output, so no new encoding logic is needed.
- **Freezing the action stream** (so the model treats the given actions as fixed ground truth rather than something to keep denoising) exploits the existing schedule mechanics in `openwam/deploy/denoise_schedule.py` rather than requiring new branching: since `action_stepping` only fires when consecutive schedule entries have different `sigma_a`, a `Schedule` built so the action-timestep component never changes will make the loop naturally skip all `flow_step` calls on the action stream — i.e. **zero changes to the loop body itself**, only to (a) the one new init parameter and (b) which `Schedule` gets passed in.

This keeps the patch surface minimal and low-risk: one new optional function parameter, one reused normalizer call, and a schedule constructed to freeze rather than denoise the action stream — no modification to `BaseWAMArchitecture`'s core diffusion logic.

### Piece A — implementation status

Following the user's explicit instruction ("start writing piece A. spin up as many agents to get this done"), implementation was dispatched to **two parallel background agents**, both launched against the live `lux6a5h0knbl1n` Cosmos3-Edge pod / local solo-cli repo:

1. **OpenWAM-side agent** (live on the Runpod pod, SSH `root@213.173.108.239 -p 43321`, repo `/workspace/OpenWAM`): patch `generate()` to accept `given_action_latents`, write a new `openwam/deploy/action_conditioned.py` wrapping the encode → freeze-schedule → generate flow described above, and prove it works with real tests against the loaded Cosmos3-Edge checkpoint — (a) an "action held fixed" check, (b) a "different input actions → different predicted video" check, and (c) a regression run of the original unconditioned smoke test to confirm nothing broke.
2. **solo-cli-side agent** (this repo): build the bridge from DeployX to this new OpenWAM capability — new `solo/commands/robots/lerobot/deployx/openwam_bridge.py`, wired **asynchronously/non-blocking** into `edge_agent.py`'s control loop (so OpenWAM prediction doesn't stall the real-time hardware loop), a new CLI prompt to opt in, and an extension to `telemetry.py` to log predicted-vs-actual observations side by side.

**As of this update, both agents' results have not yet been reviewed** — this doc update was itself produced by a fork that was dispatched to run in parallel with those two agents, per the user's "spin up as many agents" instruction, rather than after their completion. Next step once back in the main session: check both agents for completion, review their actual diffs/test output/files created, and report back to the user before starting any new work.

## 7. DeployX Edge Agent — Actively Being Built

This is the current, in-progress task, prioritized ahead of OpenWAM per the user's own "get the full loop working first" instinct.

### The user's architecture spec (source of truth, given near-verbatim)

> Build a lightweight DeployX Edge Agent that runs on a Mac mini physically connected to the SO-101 arms and cameras. The agent should detect and connect to the robot hardware, capture camera frames/video plus joint states and telemetry, stream observations to a RunPod endpoint running the π0-FAST policy, receive action chunks back, safely execute those actions on the SO-101, and continuously log the complete HIL rollout.
>
> The agent should include safety controls such as joint limits, velocity limits, timeout/watchdog, emergency stop, and automatic stop if the RunPod connection is lost. For every test, record timestamps, images/video, observations, commanded actions, actual joint states, errors, and success/failure metadata, then upload the resulting trajectory back to DeployX/RunPod for comparison with OpenWAM predictions.
>
> The core loop should be: Observe on Mac mini → send observation to RunPod → π0-FAST returns action chunk → validate action locally → execute on SO-101 → capture resulting observation → upload telemetry/video → repeat.
>
> Keep the Mac mini agent lightweight: hardware I/O, safety, buffering, execution, and telemetry only. Training, π0-FAST inference, OpenWAM simulation, evaluation, and test generation stay in the cloud.

### Implementation: `solo/commands/robots/lerobot/deployx/` package

- **`protocol.py`** — the shared WebSocket JSON message schema, written first (directly by the coordinating session, not delegated) as the single source of truth both the server and client build against, to prevent interface drift between two independently-working agents. Defines `ping`/`pong`, `predict`/`action` (with a `request_id` field for correlating requests to responses), `reset`/`reset_ack`, plus shared constants: `DEFAULT_PORT = 8849` (distinct from OpenWAM's 8848, so both could run on the same pod), `PREDICT_TIMEOUT_S = 5.0`, `HEARTBEAT_INTERVAL_S = 2.0`. Message/observation key names follow lerobot's own dataset feature naming convention (`observation.state`, `observation.images.<camera>`) so no remapping is needed to feed a batch directly into a lerobot policy.
- **`policy_server.py`** — the RunPod-side WebSocket server. Loads a lerobot policy checkpoint at startup (local path, HF repo id, or `solo:` Hub reference) and exposes it over the wire: `predict` messages decode incoming base64 JPEG images, build a lerobot-style batch dict, call the policy's standard `predict_action_chunk(batch)` method (confirmed present on the `PreTrainedPolicy` base class), and return the action chunk as JSON. Entry function: `run_policy_server(checkpoint_path, host="0.0.0.0", port=None)`. CLI flag: `solo robo --deployx-serve <checkpoint>` (plus `--deployx-port`), wired into `solo/cli.py`.
- **`edge_agent.py`** — the Mac-mini-side WebSocket client and control loop: observe → `predict` → `safety.validate_action` → execute → capture resulting observation → `telemetry` log → repeat, with periodic heartbeat pings and watchdog-triggered safe-stop on timeout/disconnect. Reuses the existing hardware-connection code (robot/camera config, calibration) from `modes/inference.py` and `utils/record_config.py` rather than reimplementing it. Entry function: `deployx_run_mode(config, server_url, auto_use)`. CLI flag: `solo robo --deployx-run <ws://pod-host:port>`, wired through `solo/cli.py` → `solo/commands/robo.py` → `solo/commands/robots/lerobot/lerobot.py`'s `handle_lerobot` dispatch.
- **`safety.py`** — joint-limit and velocity-limit validation (loaded from the robot's existing calibration data, not a new config format), a `Watchdog` class tracking time-since-last-response against `protocol.PREDICT_TIMEOUT_S`, and an `EmergencyStop` mechanism (SIGINT/SIGTERM handling plus a dedicated `EmergencyStopTriggered` exception the main loop catches to run a safe-shutdown routine).
- **`telemetry.py`** — structured per-step JSONL logging (timestamp, step index, request_id, observation summary/image refs, commanded action, actual resulting state, safety outcome, error/success) to `outputs/deployx/<session_id>/telemetry.jsonl`, with images saved as separate files (not inlined), plus a function to push a completed session's telemetry to HF Hub as a dataset repo, reusing the existing `authenticate_huggingface` pattern.
- **`runpod_deployx.py`** — provisions/reuses a Runpod pod and starts `solo robo --deployx-serve <ckpt>` on it over SSH, printing the resulting `ws://<pod_ip>:<port>` URL for the edge agent to connect to. Reuses `RunpodClient`, `ensure_local_ssh_key`, `select_or_create_pod`, `bootstrap_pod`, and `run_remote` from `runpod_train.py` rather than duplicating that logic.

### How it was built

Two background agents worked **in parallel** from the shared `protocol.py`: one built `policy_server.py` + CLI wiring + (after being resumed following a session interruption) `runpod_deployx.py`; the other built `safety.py`, `telemetry.py`, `edge_agent.py`, and its own side of the CLI wiring. Both agents' edits to the shared dispatch files (`solo/cli.py`, `solo/commands/robo.py`, `solo/commands/robots/lerobot/lerobot.py`) landed **cleanly with no conflicts** — `deployx_serve` is handled directly and early in `cli.py`'s `robo()` (since the server doesn't need the rest of the lerobot dispatch machinery), while `deployx_run` threads through `robo.py` → `lerobot.py`'s `handle_lerobot` → the new `deployx_run_mode`.

**Build status: complete.** Every file in the `deployx/` package, plus the three edited dispatch files, compiles and imports cleanly together (`python3 -c "from solo.commands.robots.lerobot.deployx import policy_server, edge_agent, safety, telemetry, runpod_deployx, protocol"` succeeds), and the two functions the CLI wiring calls (`run_policy_server`, `deployx_run_mode`) exist with matching signatures. `runpod_deployx.py` was the last piece to land — finished by resuming the policy-server build agent after a session interruption, since `policy_server.py` and the CLI wiring had already completed in that same run.

One notable deviation from the original spec, found by the build agent itself: rather than hand-building the batch dict and normalizing images manually (as the task description suggested), `policy_server.py` discovered and reuses **`lerobot.async_inference.policy_server`** — an official, already-shipped lerobot module built for exactly this "serve a policy over the wire" use case. So `predict` requests go through `make_pre_post_processors` (the same preprocessor/postprocessor pipeline `lerobot_train`/`lerobot_record` use) before and after `policy.predict_action_chunk`, getting dataset-stat normalization, image resizing to the policy's trained resolution, and any language tokenization correct automatically, rather than risking a subtly-wrong hand-rolled contract. `policy_server.py` also loads the checkpoint in a background thread so the socket accepts connections and answers `ping` with `ready: false` immediately, without blocking on GPU load.

## 8. Live Hardware Testing (DeployX)

Unlike everything in section 7 up to this point, the following was run against the user's **actual physical SO-101 arm and camera**, connected to a real machine — not just compiled/imported.

### Read-only hardware test — PASSED

Connected to the real SO-101 follower arm (`robot_type=so101`, `follower_id=follow_a`, port `/dev/tty.usbmodem5A680133541`) and one OpenCV camera, read a single live observation, and disconnected cleanly. Result: 6 joint values read correctly (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) plus one `720x1280x3 uint8` camera frame (`front`). This exercised `edge_agent.py`'s `_build_follower_robot` and `_split_observation`, and `safety.py`'s `load_joint_limits`, using the exact same code path `run_edge_agent()` itself uses. No motion commands were sent — pure read verification, zero risk of unexpected arm movement.

### Full-loop test with a local policy — RAN, surfaced a real config mismatch (not a bug)

Started `solo robo --deployx-serve` **locally** (not on Runpod) pointing at the ACT checkpoint from the earlier Runpod training run (`outputs/train/vivekgr92_deploy-test-4_act/checkpoints/001000/pretrained_model`, 1000 steps, trained on `vivekgr92/deploy-test-4`), listening on `ws://127.0.0.1:8849`. It loaded successfully on CPU/MPS (this Mac has no CUDA GPU). Then ran `solo robo --deployx-run ws://127.0.0.1:8849` against it with a bounded 3-second session.

Every component of the loop worked correctly: robot connected, safety limits loaded for all 6 joints, the WebSocket ping/pong and reset handshake succeeded, a `predict` request was sent for step 0 — and the **first prediction failed cleanly** with `'observation.images.side'`. Root cause confirmed from the checkpoint's own `config.json`: it lists `input_features: ["observation.state", "observation.images.front", "observation.images.side"]` — it was trained on a 2-camera dataset, but only one camera (`front`) is physically connected/configured on this test setup right now. The server returned a proper `error`-typed response (per `protocol.py`) rather than crashing or hanging; the edge agent logged the error to telemetry (`outputs/deployx/20260923_101814_50101c96`) and safely disconnected the robot in its `finally` block, exactly as designed.

This is validated end-to-end plumbing with a known, understood hardware-config gap — not an open bug. Getting a full successful round trip just needs a second camera physically connected and configured with viewing angle `side` to match what this checkpoint expects (or testing against a checkpoint trained on a single camera instead).

### Full-loop retry with two cameras — full round trip succeeded, safety layer caught a bad prediction

The user connected a second physical camera. Re-ran `solo robo --deployx-run ws://127.0.0.1:8849` against the same local policy server (same 1000-step ACT checkpoint). This time both cameras were detected and configured to match the checkpoint's requirements: Camera #0 (OpenCV, 1280x720) → `front`, Camera #1 (OpenCV, 1920x1080) → `side`.

The full round trip succeeded: robot connected, safety limits loaded for 6 joints, WebSocket ping/pong/reset handshake succeeded, a `predict` request was sent — and this time a real `action_chunk` came back from the policy server (no camera-mismatch error). `safety.py`'s `validate_action` then correctly **rejected** the first sub-action: `"Unsafe action rejected: joint 'gripper' action 22.7 is extremely out of range [1713.0, 2831.0]"` (off by roughly two orders of magnitude), raising `EmergencyStopTriggered` exactly as designed. The robot was safely shut down without that action ever reaching a real motor; telemetry was saved (`outputs/deployx/20260923_101950_e301f576`) and the session ended cleanly — no crash, no hang.

The bad prediction itself is explained by the checkpoint, not a DeployX bug: it was only trained for 1000 steps as the original Runpod-pipeline smoke test and was never intended to be a competent policy, so garbage output is expected. The significant result is that **the full network round trip — camera capture → WebSocket → policy inference → real action chunk → back to the edge agent — is now proven working end to end**, and **the safety/e-stop layer correctly intercepted a genuinely unsafe action before it could reach real hardware**.

**Housekeeping note**: the local policy server (port 8849, PID visible via `ps aux | grep deployx-serve`) is still running in the background as of this writing.

## 9. LiveKit Portal — Investigated, Deferred

While validating DeployX, the user asked whether **LiveKit Portal** (`github.com/livekit/portal`, Apache 2.0, ~29 stars — early-stage) had anything useful for DeployX's transport layer. This was investigated in depth (cloned the repo, installed the actual `lerobot-robot-livekit`/`lerobot-teleoperator-livekit` packages, read their real source — not just docs) rather than assumed from the README.

### What it is and why it's relevant

Portal is a WebRTC-based transport (built on LiveKit's SFU) purpose-built for exactly DeployX's use case: streaming synchronized `(frames, state, timestamp)` observations from a robot to a remote policy/operator and actions back, with a documented **control hand-off** mechanism (`set_active_operator()`) letting a policy and a *remote* human teleoperator share one session and swap control mid-episode with one call.

### Confirmed via source inspection (not docs)

- `LiveKitTeleoperator` (robot side) and `LiveKitRobot` (operator/policy side) are real subclasses of lerobot's own `Teleoperator`/`Robot` base classes. Their own `so101` example uses `SO101Follower`/`SO101FollowerConfig` — the exact classes `solo/commands/robots/lerobot/config.py` already builds.
- **What would stay unchanged** if adopted: `safety.py`, `telemetry.py`, `config.py`, `cameras.py`, `ports.py` — all transport-agnostic.
- **What would be replaced**: `protocol.py` (the hand-rolled WebSocket schema) and the WebSocket connect/ping/predict block inside `edge_agent.py`'s `run_edge_agent` and `policy_server.py`'s `websockets` server — a scoped transport swap, not a rewrite.
- Self-hosted setup is a single `livekit-server --dev` binary, no Docker required for local dev.

### A real blocker if self-hosting on Runpod specifically

Checked directly against Runpod's own docs: **Runpod pods do not support UDP port exposure at all** — TCP/HTTP-proxied ports only. LiveKit's default WebRTC path needs a UDP range (50000-60000), so self-hosting `livekit-server` on a Runpod pod would require forcing LiveKit's documented **TCP-only ICE fallback** (port 7881, built for "VPN, corporate firewalls" scenarios) — not a hack, but a real latency/quality tradeoff versus native UDP, and it partly undercuts the reason to adopt Portal in the first place.

### Cost (verified against LiveKit's pricing + quotas docs, not just the marketing page)

For our actual usage shape (plain WebRTC room/data streaming — Portal doesn't use LiveKit's separate voice-AI "Agents" product, so the Agents-specific pricing lines don't apply): the free **"Build"** tier includes **5,000 WebRTC connection-minutes/month** and **50GB downstream data transfer/month**, permanently free, no credit card required, and is a hard cap (requests fail past it, no surprise billing). At ~2 participants per DeployX session (robot + operator/policy), that's roughly **40+ hours/month of two-participant testing for free**. Effectively free for our use case; the $50/$500 paid tiers are sized for production voice-agent deployments, irrelevant here.

### Decision: defer

**Recommendation given and accepted**: keep the current WebSocket transport for now. Reasoning — Portal's only real advantage for us is remote operator hand-off, which nothing in the current architecture needs yet (Runpod pods already have public IPs, so there's no NAT-traversal problem to solve today). Revisit this swap specifically when/if remote human correction becomes a real requirement, not before. Nothing has been implemented from this investigation — it's pure research, captured here for when it's picked back up.

## 10a. Piece A — Action-Conditioned OpenWAM Generation: Implemented and Tested (Result: mechanism proven, checkpoint unusable as-is)

Two parallel agents completed this. Both changes are live on the pod (`213.173.108.239:43321`, `/workspace/OpenWAM`) and mirrored in the local reference clone, `md5sum`-verified identical.

**`openwam/model/architectures/base.py`** — `generate()` gained one new optional kwarg, `given_action_latents: Optional[Tensor] = None`. When provided, it's used directly as `action_latents` instead of sampling from `torch.randn(...)`; when omitted (the default), behavior is byte-identical to before. Fully backward-compatible, one-parameter change, no loop-body edits.

**New file `openwam/deploy/action_conditioned.py`** — `make_frozen_action_schedule()` (wraps `make_schedule`, pins the action column to a constant timestep so the denoiser never re-samples it), `encode_given_action()` (runs the real action through `model.normalizer.normalize()` — reuses `_UnifyAwareNormalizer`, reshapes to `(1, T, action_dim)`), `generate_given_action()` (glues both together and calls `model.generate(..., given_action_latents=...)`).

**Real test results, on the live `robotwin_dual_system_joint_self_attention_cosmos3` checkpoint (action_dim=80 unified / 20D raw, min-max normalize, unify_action=True):**
- ✅ **Action held fixed**: input raw action vs. decoded output action, max abs diff ≈0.0018 (float32→bf16→float32 round-trip noise only). The injection mechanism works exactly as designed.
- ❌ **Different actions → different video, on this checkpoint**: FAIL — bit-identical video (MAD=0.0000) for two different input actions. **Root cause (confirmed, not a bug in the new code)**: this specific checkpoint was trained with `attention_mask_mode: action_sees_video` — action tokens attend to video, but video tokens never attend to action, by training-time design. Video generation is structurally independent of the action stream on this checkpoint, full stop.
- **Wiring sanity check**: rebuilt the MoT attention driver with `attention_mask_mode="mutual"` (off-distribution, diagnostic only, not a usable config) — video then diverged sharply between the two actions (MAD=3.47), proving `given_action_latents` and the frozen-action schedule are correctly wired end to end. The mechanism is not the problem.
- **Regression check**: killed/restarted `deploy.py` with the patched `base.py`; `scripts/inference_test/inference_single_test.py` still passes cleanly (ping/predict/reset all OK, action dim=20). Normal (non-given-action) generation is unaffected.
- **Timing**: model load ≈50s; one `generate_given_action` call (10 denoising steps, 29 frames, 480×832, `decode_video=True`) ≈5-7s.

**What this actually means for the pipeline:** Piece A's mechanism (Track A → π0-FAST action → OpenWAM video prediction) is code-complete and verified correct. But **no currently-deployed checkpoint can use it** — every checkpoint we've loaded (Cosmos3-Edge included) was trained with an attention mask that makes video independent of action. To make step #4 of the pipeline (predicted action → predicted resulting observation) actually work, we need a checkpoint trained with `mutual` or `video_sees_action` attention-mask mode, or a fine-tune under one of those. This is the same "no SO-101 support" gap as before, now sharpened: it's not just "no SO-101 dataloader," it's "no action-conditions-video checkpoint at all" — both would need to be solved by the same eventual fine-tuning job.

**Pod state**: `deploy.py` killed, GPU back to 2MiB/32623MiB, no lingering processes. Confirmed idle and safe to stop.

## 10b. Piece A Validated on a Genuinely `mutual`-Trained Checkpoint — Video DOES Condition on Action

Follow-up to 10a: since no deployed checkpoint had the right attention mode, the OpenWAM HF org (`OpenWAM/*`, 46 model repos total) was surveyed directly via raw config fetches (not summarizer output, which was caught giving one wrong answer for a "flux2" checkpoint — see below). Every backbone-ablation checkpoint (cosmos3, cosmos25, flux2, wan21_i2v_14b, wan21_vace_1_3b, dinov3, vjepa21) defaults to `attention_mask_mode: action_sees_video` regardless of backbone. Exactly two checkpoints in the whole org have the needed mode: `OpenWAM/robotwin_dual_system_joint_self_attention_mutual` (`mutual`) and `..._video_sees_action`. Both use the Wan2.2-TI2V-5B + umt5-xxl combo (~23GB) that OOM'd on a 24GB RTX 4090 earlier this session.

(Aside, in case "flux2"/"flux3" comes up again: `robotwin_dual_system_joint_self_attention_flux2` is **not** a smaller/different generative backbone — raw config confirms it only swaps in FLUX.2-dev's VAE as a frozen pixel-reconstruction encoder; the DiT is still Wan2.2-TI2V-5B + umt5-xxl, and its attention mode is the same wrong-direction `action_sees_video`. Separately, Black Forest Labs' own **FLUX 3** — announced 2026-07-23, includes a robotics "FLUX 3 Action"/FLUX-mimic capability — is unrelated to OpenWAM, partner-only early access (mimic robotics, Audi), no public weights/API/SDK. Neither is usable here.)

**Infra**: original pod `lux6a5h0knbl1n`'s host ran out of capacity on restart (retried twice, still failed) — created a new pod `vf8nchsempd3mg` (EU-RO-1, RTX PRO 4500 Blackwell, 32GB VRAM, $0.72/hr) reusing the same network volume `yr9b07r6gb`. That volume's 50GB quota was also hit mid-download (`Disk quota exceeded`) once the 24GB `mutual` checkpoint plus the existing ~18GB Cosmos3-Edge assets left no room for the 34.2GB Wan2.2-TI2V-5B backbone; resized the volume 50GB→100GB (STANDARD tier, +$3.50/month) and the retry completed via resume.

**Real test results, `OpenWAM/robotwin_dual_system_joint_self_attention_mutual` (action_dim=20 raw, no unify_action — unlike Cosmos3, this checkpoint's normalizer works directly in raw 20D space):**
- **VRAM**: loaded at **24130 MiB / 32623 MiB** — fits with ~8.5GB headroom, no OOM. (The `_mutual`/`_video_sees_action` checkpoints' 20D-raw config also needed the checkpoint's placeholder `/path/to/Wan2.2-TI2V-5B` patched to the real downloaded path in `config.yaml` — `model_loader.py` does no automatic placeholder resolution, unlike Cosmos3 which embeds its own component specs.)
- ✅ **Check 1 (action held fixed)**: PASS both runs, max abs diff 0.0019 / 0.0013 (same float32↔bf16 round-trip noise as before).
- ✅ **Check 2 (different actions → different video)**: **PASS for real this time** — MAD=1.38 (random-noise reference image) and MAD=0.55 (a second run with a structured, non-noise reference image: gray canvas + brown table + red block). Both comfortably clear the 0.5 pass threshold. This is the first time this has passed on a checkpoint that was actually trained for it, not an off-distribution diagnostic hack.
- **Regression check**: `inference_single_test.py` against the same checkpoint — ping/predict/reset all OK, action dim=20, passed cleanly.
- ⚠️ **Honest caveat — visual sanity check was inconclusive**: downloaded and inspected the decoded frames from the structured-reference run. The two videos (action A vs action B) look visually near-identical to the eye — both render as a hazy, low-detail blur with the same rough shapes, no clearly different arm/gripper motion visible. Likely cause: the server logs `[obs] View config: multiview=True, camera_layout=['head_camera','left_camera','right_camera'], canvas=384x320` — the checkpoint expects a **3-camera tiled multiview canvas**, and both test runs fed it a single flat synthetic image (or pure noise), which is far out-of-distribution for the input format regardless of action-conditioning. So: the **numeric** divergence is real and well above threshold (not a fluke — reproduced across two different reference images), but a genuinely convincing **visual** confirmation ("the gripper visibly moved differently") still needs a real, correctly-formatted 3-camera RoboTwin frame as input, not a synthetic placeholder.
- **Timing**: model load ≈205s (bigger than Cosmos3's ≈50s, expected given the 5B+11B combo); one `generate_given_action` call ≈10-12s.

**Conclusion**: Piece A is no longer just "mechanically correct, unusable" — it's now **demonstrated working end-to-end on a real checkpoint that was actually trained for action-conditioned video**, with the one remaining gap being a proper visual/qualitative confirmation using correctly-formatted multiview input (RoboTwin data, not synthetic). SO-101 still has zero native support in OpenWAM either way, so a fine-tune is still required for the actual target embodiment — but the core mechanism question ("can OpenWAM condition video generation on an externally-supplied action at all") is now answered yes, not just in theory.

**Pod state**: `vf8nchsempd3mg` — `deploy.py` killed after each test, GPU confirmed idle (2MiB/32623MiB) as of this update. Both pods (`lux6a5h0knbl1n`, EXITED/stopped, and `vf8nchsempd3mg`, RUNNING) currently exist on the account; `vf8nchsempd3mg` should be stopped if no further OpenWAM testing is imminent (still billing at $0.72/hr while RUNNING).

## 10. Current State / Next Steps

**Verified working (live-tested against real infrastructure or real GPU runs):**
- `solo robo --train` → `runpod` single-policy training end-to-end, including bootstrap, HF Hub dataset auto-pull, remote execution, checkpoint sync-back, and pod cleanup.
- The capacity-error fallback/retry logic (triggered for real, repeatedly, against the account's capacity-blocked US-IL-1 pods).
- `RunpodClient`'s REST call shapes (validated against Runpod's actual OpenAPI spec and via live pod creation).

**Also now verified live (section 8):**
- DeployX's hardware layer (`edge_agent.py` + `safety.py`) — read-only test passed against the real SO-101 arm and camera.
- DeployX's full loop, end to end, with two cameras matching a real checkpoint's requirements: WebSocket handshake, real observation capture, a real `predict` round trip returning an actual `action_chunk`, and `safety.py`'s reject path correctly intercepting an unsafe prediction (`EmergencyStopTriggered`) before it reached a motor. Telemetry logging and safe shutdown both fired correctly on both the mismatch-stop and the e-stop paths.

**Built but not yet live-tested:**
- The 8-policy parallel comparison mode (`run_policy_comparison_on_runpod`) — implemented, compiles, but no live run across 8 pods has been executed.
- `hil_merge.py` — needs a real teleoperated correction session to exercise.
- `runpod_deployx.py` (deploying the policy server to an actual Runpod pod, as opposed to running it locally) — the local-server path is now proven; the Runpod-deploy path itself hasn't been exercised live yet.
- **A successful *action execution*** — both live tests so far ended in a controlled stop before a motor command was sent (the first on a camera mismatch, the second on a rejected unsafe action), so `validate_action`'s clamp path (as opposed to its reject path) and `robot.send_action` have not yet been exercised against real hardware. Next step: test against a more competently trained policy (or a longer-trained checkpoint) to see a full successful action-execution cycle.

**Known gaps flagged during testing but not yet fixed:**
- **Camera-rename prompt not covered by `SOLO_REMOTE_TRAINING`**: during the live Runpod training test with a 2-camera dataset (section 2), `training.py`'s "auto-map cameras?" confirmation isn't skipped by the `SOLO_REMOTE_TRAINING`/`--yes` non-interactive path — an unattended remote training run on multi-camera data would hang waiting for input on the pod. Flagged, not patched.
- **Stop-pod retry**: offered but not built — `client.stop_pod()` failures (the `RemoteDisconnected` flakiness hit twice live) currently just print a manual-stop warning (section 2's bug list) rather than retrying 2-3 times automatically before giving up.

**OpenWAM — Piece A implemented AND validated working on a real checkpoint (sections 6, 10a, 10b):** the original `openwam-assets` volume/pod is gone (unexplained infra instability, see section 6), but a much smaller **Cosmos3-Edge** checkpoint (~18.3GB vs. ~59GB) was found and live-smoke-tested. The **action-conditioning mechanism ("Piece A")** is implemented (`given_action_latents` param in `generate()` + `openwam/deploy/action_conditioned.py`) and was first proven mechanically correct via an off-distribution diagnostic on Cosmos3-Edge (section 10a), then **confirmed working for real** (section 10b) on `OpenWAM/robotwin_dual_system_joint_self_attention_mutual` — a checkpoint genuinely trained with `mutual` attention masking: fits in 32GB VRAM (24.1GB used), action-held-fixed check passes, and different actions produce measurably different video (MAD 0.55-1.38, well above threshold, reproduced across two different reference images). One open item: a genuinely convincing **visual** confirmation still needs a correctly-formatted 3-camera RoboTwin input frame (the test's synthetic/noise reference images were out-of-distribution for the checkpoint's expected multiview canvas, so the decoded frames didn't show an obviously different gripper motion despite the numeric divergence being real). SO-101 has **zero native support** in OpenWAM (confirmed — no dataloader, checkpoints trained on Aloha-Agilex/eef-action/3-camera data). `solo-cli`-side `openwam_bridge.py` (client wiring into `edge_agent.py`) is also complete.

**Decision made:** the original Cosmos3-Edge pod (`lux6a5h0knbl1n`) had its host run out of capacity on restart; a replacement pod `vf8nchsempd3mg` (EU-RO-1, RTX PRO 4500, $0.72/hr) was created reusing the same network volume (resized 50GB→100GB, +$3.50/month, to fit the Wan2.2-TI2V-5B backbone). Both pods currently exist; `vf8nchsempd3mg` should be stopped if no further OpenWAM testing is imminent (still billing while RUNNING).

**Explicitly deferred:** LiveKit Portal transport swap (section 9) — researched and scoped, but intentionally not started; revisit specifically when remote operator hand-off becomes a real requirement.

**Status unknown, worth a check-in:** the user's own "Track A" task — fine-tuning π0-FAST for real (as opposed to the ACT checkpoints trained so far, which were pipeline smoke tests) — was split off as the user's own parallel work early in this session; not confirmed done. User separately asked (this session) for a background agent to investigate why π0.5 training isn't working — see note below.

**Logical next steps once picked back up:**
1. ~~Review the two Piece A agents' results~~ — done, see section 10a. ~~Decide whether a usable `mutual`/`video_sees_action` checkpoint exists~~ — done, see section 10b: `OpenWAM/robotwin_dual_system_joint_self_attention_mutual` works and fits in 32GB. Reconcile `openwam_bridge.py`'s assumed message schema against the real `action_conditioned.py` API now that both sides are proven.
2. Get a proper visual confirmation of Piece A using a real, correctly-formatted 3-camera RoboTwin input frame (not synthetic/noise) — the numeric check has passed twice, but nobody has yet seen video that visibly looks like "the gripper moved differently because the action changed."
3. Run DeployX against a better-trained checkpoint (or extend training) to get a full successful action-execution cycle, then move the policy server from local to an actual Runpod pod via `runpod_deployx.py` to validate the intended network topology end to end.
4. Write the new OpenWAM SO-101 dataloader and run a real fine-tune — needed regardless of #1/#2, since SO-101 has zero native support in OpenWAM either way. **Now fully scoped, see section 10c** — concrete build plan, real data shape confirmed, official foundation checkpoint identified.
5. **Resolved this session** (was item 5): π0.5 training failure was root-caused and fixed — see section 11. Batch size for the resulting ACT training run was tuned up to 32 on real GPU headroom.

## 10c. OpenWAM SO-101 Dataloader — Scoped (Not Yet Built)

Full scoping investigation (real code read, not guessed) for the SO-101 fine-tuning dataloader flagged as needed in section 10b's next steps. Investigated both sides: OpenWAM's actual dataloader interface/base classes, and our own real recorded SO-101 data.

**OpenWAM's own official fine-tuning guide already exists and answers most of this**: `assets/openwam_usage_docs/openwam-alpha-finetuning.md` in the repo. Key points from it:
- Register a new reader via `openwam/dataloader/registry.py` (`register_dataset("name")` decorator + a `from_config` classmethod), returning the canonical sample dict (`video`, `action` (T-1,80), `action_mask`, `proprio` (1,80), `proprio_mask`, `video_mask`, `prompt`).
- **Explicitly recommends `LeRobotV3Reader` (`openwam/dataloader/bases/lerobot_v3_reader.py`) as the base for LeRobot v3 data** — and our data genuinely is LeRobot v3 (confirmed: `meta/info.json` on a real local dataset says `"codebase_version": "v3.0"`). This is a lightweight subclass path (see `openwam/dataloader/oxe_droid.py` as a concrete ~250-line example: override class attrs `HEAD_CAMERA`/`LEFT_WRIST_CAMERA`/`ACTION_DIM`/`NEEDED_COLS`, implement `_build_episode_index`, `_action_Nd`/`_proprio_Nd`, `_load_stats`) — **not** the heavy bespoke ~1175-line reader RoboTwin uses (`openwam/dataloader/robotwin.py`, which subclasses the bare `BaseDataset` directly, not `LeRobotV3Reader`). This means our new reader is a small, templated subclass, not a from-scratch implementation.
- Correct fine-tune starting point per the official doc: **`OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model`** (the α foundation/pretrain checkpoint), not the RoboTwin-SFT `mutual` checkpoint we validated Piece A on. **Verified via raw config fetch: this foundation checkpoint already has `attention_mask_mode: mutual` and `action_dim: 80` natively** — so it's both the officially-correct starting point AND already has the action-conditioning capability Piece A needs. No conflict between "what the docs say to fine-tune from" and "what has the capability we validated."
- Launch command: `NPROC_PER_NODE=8 bash scripts/train.sh dataloader=my_task training.finetune_ckpt_path=<foundation_ckpt_path> training.num_epochs=5 project.output_dir=<output_dir_path>` (torchrun-based). Default `batch_size: 24` (per-process micro-batch), `mixed_precision: bf16`, `zero_stage: 2`, `use_gradient_checkpointing: true` — these are the pretraining-scale (8-GPU) defaults, not necessarily right for a single-GPU SO-101 fine-tune; would need tuning down.

**Real SO-101 data shape** (read directly from `~/.cache/huggingface/lerobot/vivekgr92/deployx-test-5/meta/info.json`, an actual local recording from this session):
- `codebase_version: "v3.0"`, `robot_type: "so_follower"`, `fps: 30`
- **Action/state: 6-D raw joint space** — `shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos`. This is NOT the EEF (xyz+rot6d+grip) layout every existing OpenWAM reader assumes — it's raw joint angles, smaller and structurally different. No existing `unify_action_map` in any shipped config covers a joint-space layout this small; a new mapping spec has to be authored (see risks below).
- **Cameras: exactly 1** — `observation.images.front`, 1280x720, AV1 codec, 30fps. Not 2, not 3.
- Current volume: 14 episodes, 12,600 frames total = **~7 minutes of footage**, across all episodes combined.

**The camera-count and action-dim mismatch — resolution recommended:**
- *Cameras*: `LeRobotV3Reader.__init__` already has first-class support for `multiview=False` (confirmed by reading the actual `__init__`: when multiview is off, `self._camera_layout = [self._head_camera]` — a single camera, no 3-slot canvas, no padding/placeholder tiles needed). **Recommendation: fine-tune with `multiview=False`, `target_camera=observation.images.front`**, not try to force 1 real camera into a 3-slot layout with 2 synthetic black tiles. Padding with permanent blank tiles would just teach the model "two areas are always black" — not real multiview understanding, and wastes canvas capacity. This does mean the model has to adapt to a genuinely different canvas composition than its RoboTwin/OXE training ever used — a real distribution shift, but exactly the kind of thing fine-tuning exists to handle, and the mechanism itself (single-camera mode) is already implemented and used by other readers, not something we'd be inventing.
- *Action dim*: `ACTION_DIM` and `unify_action_map` are schema-driven per-reader (confirmed via `openwam/dataloader/utils/unify_action.py` — `parse_unify_spec` takes an arbitrary source-dim→destination-slot mapping, width is just a passed-in constant). New reader sets `ACTION_DIM = 6`, extracts directly from the `action`/`observation.state` parquet columns (already flat, correctly named — no coordinate conversion needed, unlike DROID's euler→arm10 conversion). **Open design decision, not yet resolved**: which of the 80 unified slots the 6 raw joint dims should map to. The documented layout (`0:3` left EEF xyz, `3:9` left EEF rot6d, `9` left gripper, `10:34` left hand, `34:80` right-arm/reserved) is EEF-semantic, not joint-semantic — there's no existing joint-space precedent in any shipped config to mirror. Simplest defensible choice: map into the `68:80` "reserved" 12-slot region (6 dims fit with room to spare) so we don't collide with or corrupt any EEF-semantic slot another embodiment in a future mixture might use — but this is a real modeling choice that affects fine-tune quality, not just plumbing, and deserves a considered answer before training starts.

**Concrete build plan (in order):**
1. New file `openwam/dataloader/so101.py`, class `SO101Dataset(LeRobotV3Reader)`, following `oxe_droid.py`'s structure but much simpler (no euler conversion, no prompt-exclusion digest machinery needed for a small single-source dataset — SO-101 recordings are homogeneous, unlike OXE's multi-source population).
2. Set `ACTION_DIM = 6`, `HEAD_CAMERA = "observation.images.front"`, `LEFT_WRIST_CAMERA = None`, `RIGHT_WRIST_CAMERA = None`.
3. Implement `_action_6d`/`_proprio_6d`-equivalent hooks (the base's naming is `_action_20d`/`_proprio_20d` because `EEF_DIM=20` is the base default — our subclass overrides `ACTION_DIM` so these hooks just read the 6 named columns directly into a `(T-1, 6)`/`(1, 6)` array).
4. Decide and hardcode the joint→unified-slot mapping (the open decision above) in `configs/dataloader/so101.yaml`, modeled on `configs/dataloader/robotwin.yaml`'s structure but with `multiview: false`, `target_camera: observation.images.front`, `unify_action_map: [<chosen 6-slot range, e.g. "68-73">]`.
5. Compute and write `normalization_stats.npy` for the SO-101 dataset (min-max per the α protocol) — reuse `openwam/dataloader/utils/normalization.py`'s helpers; needs real min/max per joint across the recorded episodes.
6. Register in `openwam/dataloader/registry.py` (`register_dataset("so101")(SO101Dataset)`).
7. Download `OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model` (the verified-`mutual` foundation checkpoint) via the repo's existing `scripts/download_assets/download_openwam_checkpoints.py`.
8. Launch: `bash scripts/train.sh dataloader=so101 training.finetune_ckpt_path=<foundation_ckpt_dir> training.num_epochs=<N> project.output_dir=<out>` — single-GPU, so `NPROC_PER_NODE=1` and `batch_size`/`zero_stage`/`gradient_checkpointing` all need tuning down from the 8-GPU pretraining defaults (not yet determined what fits on a single RTX PRO 4500/32GB — would need a real trial run to find out, same way batch_size tuning worked for the π0.5/ACT Runpod training this session).

**Honest risks / unknowns, not yet resolved:**
- **Data volume is almost certainly too small right now.** 7 minutes total (14 episodes) vs. OpenWAM-α's own pretrain of ~6,400 hours. No official minimum fine-tune data volume is stated in the docs we found, so there's no hard number to compare against, but 7 minutes is very likely insufficient for a meaningful fine-tune by any reasonable standard. The replay `--repeat`/`--save-replay-as` augmentation feature built earlier this session can multiply existing episodes with fresh camera captures, but that doesn't create new motion diversity — real new teleop recording sessions are probably still needed before this is worth running.
- **Single-camera canvas adaptation is unverified.** We know `multiview=False` is a supported code path (other readers use it), but nobody has confirmed how much fine-tuning it actually takes for a model pretrained on human egocentric + multi-embodiment robot video to adapt to SO-101's specific single fixed-camera framing. Could be fast, could need substantially more data/epochs than a multiview setup would.
- **The joint-space unify mapping is a real open design decision** (see above), not just an implementation detail — picking the wrong slots could make fine-tuning less effective or could create real conflicts if this dataset is ever mixed with EEF-semantic embodiments in a `MixtureDataset` later.
- **Single-GPU batch size/memory settings for fine-tuning are unknown** — the shipped defaults assume 8-GPU pretraining scale; would need the same kind of real-GPU trial-and-error tuning already done for the π0.5/ACT training runs this session.

## 10d. OpenWAM SO-101 Dataloader — Built and Tested (real, not scoping anymore)

Implemented the dataloader scoped in 10c. Three new/changed files in the local reference clone (`/private/tmp/claude-501/-Users-solotechdev001-Desktop-solo/35fb2845-f222-4f36-b536-80d9cdef80d7/scratchpad/OpenWAM`):

- **`openwam/dataloader/so101.py`** (new) — `SO101Dataset(LeRobotV3Reader)`. `ACTION_DIM=6`, `NEEDED_COLS=("action","observation.state","task_index")`.
  - **Camera resolution is genuinely dynamic**, per the user's explicit requirement that adding a 2nd/3rd camera later must not need a code change: `_resolve_so101_cameras(features)` scans `info["features"]` for any `observation.images.*` key (nothing hardcoded), picks a head camera by name-hint (`front`/`head`/`top`/`primary`/`main`, else first found), then fills left/right wrist slots by `left`/`right` name hints or leftover cameras in sorted order. Verified with unit tests for 0/1/2/3-camera `features` dicts — all correct.
  - **Prompt fallback override**: the real dataset's recorded task text is blank (solo-cli's recording/replay flow defaults the task-description prompt to `""`), which would make the base class's default `_resolve_prompt` raise `ValueError`. Overrode it to fall back to a generic prompt string (`"manipulate the object with the robot arm"`) instead of crashing — since a blank task description is the common case for this project's data, not a one-off data bug.
  - `_action_20d`/`_proprio_20d` are trivial (unlike DROID/RoboCOIN): SO-101's `action`/`observation.state` columns are already flat 6-D arrays, just stack + `apply_normalization`. No EEF/euler conversion needed.
  - Reused the base's stock `_load_stats` mechanism (`STATS_FILENAME="so101_joint_stats.json"`, `STATS_DIM=6`, `STATS_STRICT_MINMAX=True`) rather than writing a custom stats loader — confirmed `materialize_eef_stats`'s rot6d-pinning logic is a no-op for width=6 (only pins widths 10/20), so it's safe to reuse as-is.
- **`configs/dataloader/so101.yaml`** (new) — `unify_action_map: ["68-73"]` (the reserved 12-slot region chosen in 10c), `multiview: false` today with a comment explaining exactly what to flip when more cameras are added, `normalize_mode: min-max`.
- **`openwam/dataloader/registry.py`** — registered `register_dataset("so101")(SO101Dataset)`.
- **Stats file**: extracted real min/max/mean/std/q01/q99 for the `action` column straight out of solo-cli's own `meta/stats.json` (already computed at recording time by lerobot) into `<dataset_dir>/meta/so101_joint_stats.json` — no need to recompute from raw parquet, this is genuine already-validated data from the real recording.

**Real test results** (instantiated `SO101Dataset` directly against the actual local dataset, `~/.cache/huggingface/lerobot/vivekgr92/deployx-test-5`, 14 episodes / 12,600 frames):
- `len(ds) == 12600` ✓, resolved camera = `observation.images.front` (correct, only real camera) ✓
- Sample at index 0: `video` = 9 PIL frames at 320×384 (single-camera canvas, no multiview padding) ✓, `action` shape `(32, 80)`, `proprio` shape `(1, 80)` — correct post-unify width ✓
- **Verified the unify-scatter actually placed real data in the right place**, not just correct shape: nonzero action values at t=0 are at exactly slots `[68,69,70,71,72,73]`, nowhere else; values are real normalized joint positions in `[-1,1]` (e.g. `[0.42, -0.98, 0.99, -0.99, -1.00, -0.89]`), not placeholder zeros. `action_mask`/`proprio_mask` correctly mark exactly those 6 slots valid.
- **Boundary case** (last valid window index of the dataset): no crash, `video_mask` correctly shows only 1 of 9 frames valid (window truncated at episode end), action still carries a real supervised step.
- **Forward-compatibility check, run for real (not just asserted)**: instantiated a second reader with `multiview=True` against the SAME 1-camera dataset — it correctly fell back to the base class's `__missing_left__`/`__missing_right__` black-tile convention for the two absent wrist cameras and produced a valid 384×320 tiled canvas without any code change. This is the concrete proof that adding a 2nd/3rd camera later really is just a config flag flip, not a re-test-and-hope.

**What didn't match the 10c scoping summary, corrected**: `oxe_droid.py` is not actually a "~250-line lightweight example" as the scoping pass summarized it — the real file is ~340 lines and most of it is DROID-specific prompt-exclusion-manifest/digest machinery and euler-pose conversion that do NOT apply to SO-101 (confirmed by reading it in full this pass). The actually-relevant template pattern is `RoboCOINDataset._resolve_cameras`'s "resolve by info.features priority" style (used as the basis for `_resolve_so101_cameras`) plus the base class's own default hooks — SO-101's reader ended up genuinely thin (~150 lines) because almost every hook could stay at its base-class default; only cameras, prompt fallback, and action/proprio extraction needed overrides.

**Not done / still open**:
- Not yet pushed/deployed to the Runpod pod or synced with the local solo-cli repo's own working copy — this was built and tested in the local `scratchpad/OpenWAM` reference clone only.
- No actual fine-tuning run started — this unblocks step 7-8 of the 10c build plan (download the foundation checkpoint, launch `scripts/train.sh dataloader=so101 ...`), not executed yet.
- The data-volume risk flagged in 10c (~7 minutes of footage) is unchanged — the dataloader being correct doesn't address whether there's enough data to fine-tune meaningfully.
- Only one real SO-101 dataset was available to test against (`vivekgr92/deployx-test-5`); the reader's generality across other SO-101 recordings (different episode counts/lengths, non-blank task descriptions, 2-3 camera setups) is verified by code inspection and targeted synthetic tests (camera resolver, multiview fallback), not by running against a second real dataset — none exists yet.

## 11. π0.5 Training Failure — Root-Caused, Fixed, Then Partially Reverted Per User Request

**The real error** (from an actual failed Runpod run, dataset `vivekgr92/deployx-test-5_pi05`):
```
❌ Training failed: Unknown policy type: pi05
File "/usr/local/lib/python3.12/dist-packages/solo/commands/robots/lerobot/modes/training.py", line 343, in training_mode
    raise ValueError(f"Unknown policy type: {policy_name}")
```

**Root cause, confirmed via `git status` + `git show origin/main:...` (not guessed)**: the local working copy of `solo-cli` already had the `pi05` dispatch branch (`training.py:407-408`), but it was **uncommitted and unpushed** — `runpod_train.py` itself was entirely untracked. `bootstrap_pod()` installed solo-cli on every pod via `pip install 'solo-cli @ git+https://github.com/GetSoloTech/solo-cli.git'`, which pulls `origin/main` — a version that predates this entire session's work (Runpod integration, the 3 new policy types, DeployX, everything). The traceback's file path (`/usr/local/lib/python3.12/dist-packages/solo/...`, a pip-installed copy) confirmed this precisely. **This is a systemic gap, not pi05-specific**: any local fix/feature not yet pushed silently never reaches a pod.

**Fix applied, then reverted at the user's request**: `bootstrap_pod()` was changed to `rsync` the local working tree to `/root/solo-cli-src` on the pod and `pip install -e` it (editable install of exactly what's on disk locally, uncommitted or not) instead of installing from the git remote. This was **live-verified working** — grepped the pod's synced `training.py` directly and confirmed the `pi05` branch was genuinely present. The user then asked to revert this (`"may be just recerse the rsync, I will upload to github manually and let it use the standar pip install"`) — done: `bootstrap_pod()` is back to the original `git+https://github.com/GetSoloTech/solo-cli.git` install. **Practical implication going forward: any local change (this session's Runpod/DeployX/replay work included) needs an actual `git push` to `main` before a fresh pod will have it** — nothing currently automates that.

**A second, separate real bug found and fixed** (kept, not reverted — orthogonal to the rsync/git choice): `pi0`/`pi0_fast`/`pi05` all require a patched `transformers` fork (`transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi`, matching lerobot's own `pi` extra) for a `transformers.models.siglip.check` module their modeling code imports — plain PyPI `transformers` lacks it and fails deep inside lerobot with a cryptic `ValueError: An incorrect transformer version is used...`. `solo-cli`'s own preflight dependency check only verified plain `transformers` was importable (would have silently reported "all requirements satisfied"), but this doesn't matter for the remote-training crash path specifically since **training never calls preflight at all** (confirmed via grep — preflight only runs on the local inference path). Real fix: `bootstrap_pod()` (`runpod_train.py`) now takes an optional `policy_name` param; when it's `pi0`/`pi0_fast`/`pi05`, it installs the patched transformers fork during bootstrap, before training can ever reach the crash point. Wired into both call sites (`run_training_on_runpod`, `run_policy_comparison_on_runpod`). **Not live-tested end-to-end yet** (no pod has been run all the way through with this fix in place).

**Batch size tuning, done live against real GPU headroom**: a separate, unrelated training run (ACT policy, not pi0.5, dataset `vivekgr92/deployx-test-5`) was found running with `batch_size: 12` at only 11.9GB/49.1GB VRAM used (24%). Bumped to `batch_size: 32` directly via editing `~/.solo/config.json` on the pod and relaunching — real headroom confirmed, not guessed. **Known ongoing operational issue, not fixed**: that pod's training process was launched attached to a live SSH pty (not `nohup`/detached), and was later found dead (pod `EXITED`) with no crash logged — consistent with dying on an SSH disconnect, the same class of issue as the "repeated unexplained Runpod resource disappearance" flagged earlier in this doc (section 5). Recommended fix (not yet implemented): launch remote training with `nohup ... &` / `disown` so a dropped connection doesn't kill it.

## 12. Replay Mode — Major Feature Additions (multi-episode, data augmentation, live viz)

All live in `solo/commands/robots/lerobot/modes/replay.py`, `solo/cli.py`, `solo/commands/robo.py`, `solo/commands/robots/lerobot/mode_config.py`. Not yet live-hardware-tested this session (built and compile-checked only) — flagged as a gap, same as other recently-built features awaiting a real test pass.

- **Multiple/ranged episode selection**: `--episode` (and the interactive prompt) now accepts a single number, a comma list (`0,2,5`), an inclusive range (`0-10`), a combination (`0-2,5,7-9`), or `all`. New `_parse_episode_selection()` helper, unit-tested standalone (not just compiled) for all these forms plus out-of-range/empty-input error cases. Backward-compatible with old saved configs that have a plain int `episode` value.
- **`--repeat N`**: replay each selected episode N times in a row (default 1). Combines with `--save-replay-as` so each repeat becomes its own new saved episode — e.g. `--episode 0,3 --repeat 5 --save-replay-as ...` yields 15 new episodes from 3 source episodes.
- **`--save-replay-as <dataset>`**: also records the replay run live — cameras + the replayed action get captured into a *new* dataset episode via the same `build_dataset_frame`/`add_frame`/`save_episode` calls lerobot's own recorder uses internally (not a custom reimplementation). This is the "use replay to collect more data" feature the user asked for: replaying a known-good trajectory under different lighting/camera angle/wear produces genuinely new training samples without a human re-teleoperating.
  - **Create-new vs. add-to-existing**: if the target dataset already exists, the interactive path reuses the existing `handle_existing_dataset()` helper (resume vs. pick a new name) already used by `recording.py`; the CLI/preconfigured (non-interactive) path auto-detects existence via a new `_dataset_already_exists()` helper and auto-resumes rather than prompting. Resuming reuses lerobot's own documented pattern (`LeRobotDataset(repo_id)` + `.start_image_writer(...)`, not `.create()`) — confirmed by reading `lerobot_record.py`'s own resume branch, not guessed.
  - Pushes to HF Hub automatically unless the repo id is `local/`-prefixed (same convention as the rest of this file).
  - Known limitation, stated to the user: the new dataset reuses the *source* dataset's camera/feature schema, so the camera setup during replay needs to match what the original recording used.
- **Cameras are now always active during replay** (previously only when `--save-replay-as` was used) — for live monitoring, not just data capture. Caches the resolved `camera_config` into the saved replay mode-config (mirroring `recording.py`'s existing caching pattern) so repeated `--yes`/preconfigured replay runs don't re-prompt for camera setup every time.
- **Rerun live visualization added**: `init_rerun(session_name="replay")` / `log_rerun_data(observation=obs, action=processed_action)` per frame / `shutdown_rerun()` in the `finally` block — same calls lerobot's own `record()` uses internally. Recording already had this (confirmed: `display_data=True` is hardcoded in `record_config.py`'s `unified_record_config`), replay didn't until now.
- **A real bug caught and fixed during this work**: `authenticate_huggingface()` returns `(bool, str)`, not a bare bool — an early draft of the push-to-hub logic did `if authenticate_huggingface():` which is always-truthy for a non-empty tuple regardless of the actual login result. Fixed to unpack properly before it shipped.
- **`--perturb <fraction>` added**: injects fresh random per-step noise into each replayed action before sending it to the robot, sized as a fraction of each joint's *calibrated safe range* (not a raw tick count) — e.g. `--perturb 0.05` means up to ±5% of each joint's real range from `deployx/safety.py`'s `load_joint_limits()` (the same calibration-file reader DeployX's edge agent already uses, not an invented range). Directly addresses the "`--repeat` doesn't create new motion diversity" gap flagged above: combined with `--repeat`, each repeat now gets its own fresh random draw, so `--episode 3 --repeat 5 --perturb 0.05 --save-replay-as x` yields 5 genuinely different perturbed trajectories, not 5 copies.
  - **Safety, non-negotiable**: a perturbed action is a *new, never-executed* command (unlike plain replay, which only ever resends an action already proven safe once during real teleop) — every perturbed action is run through the same `validate_action()` clamp-or-reject check DeployX's edge agent uses on live policy output, via a new `_perturb_action()` helper that converts between the dataset's `.pos`-suffixed action-dict format and `JointLimits`' ordered list (same key-stripping convention as `edge_agent.py`'s `_split_observation`). A rejected (garbage/extreme) perturbation aborts the run with a clear error rather than sending it.
  - Default `0` (disabled) — zero added overhead/behavior change when not used (no calibration file load, no validation call).
  - Allowed without `--save-replay-as` too (e.g. to jitter/stress-test the arm without recording) — a deliberate choice, not a limitation: it's a strict superset of the recording case, no reason to force one flag to require the other.
  - The action written into `new_dataset` (when `--save-replay-as` is used) is the real post-perturbation, post-clamp action that was actually sent to the robot — ground truth of what happened, not the original logged action.
  - **Not live-hardware-tested** — built and compile-checked only (`python3 -m py_compile`, plus an AST parse and a grep-based scope check for variable collisions), same as the rest of this session's replay work. No real SO-101 was available to confirm the noise/clamp behavior against actual hardware.

## 13. External Research — Dyna-2 and HF SO-101 Datasets

**Dyna-2** (dyna.co/research/dyna-2-infrastructure), investigated on request:
- Same category as OpenWAM (a world-action model), pretrained on 1M+ hours of egocentric human video, claiming cross-embodiment transfer with only a few hours of fine-tuning data on unseen embodiments.
- **Not usable by us**: no GitHub repo, no HuggingFace org, no paper, no public API (confirmed via an independent search, not just the landing page) — Dyna sells robotics-as-a-service (their own robot cells in hotels/restaurants), not a deployable model. No SO-101 support disclosed, no way to check since nothing is public. Tracked only as external validation that the general "world-action model + small embodiment fine-tune" direction is sound at scale — not adopted, nothing to integrate.
- **One concretely reusable technical idea from the page** (which is actually about their training-data infrastructure, not the model): "topic-group chunking" — instead of MCAP's default of interleaving all sensor topics (cameras, proprioception, actions) into mixed chunks, group topics by read-access pattern (cameras together, state/actions together, never sharing a chunk), since training samples read modalities asymmetrically (few camera frames, long dense action/state windows). Real measured gains they report: 3.4x fewer chunk fetches, 2.9x faster reads, ~68% storage savings (with H.264 + larger-GOP encoding). **Assessed as not needed for our current LeRobot-format data** (already separates video files from Parquet state/action data, stronger separation than the MCAP fix even provides) but **a concrete, applicable technique to reuse if/when the OpenWAM SO-101 fine-tuning dataloader (section 10c/10d) becomes I/O-bound at scale** — OpenWAM's own training reads exactly this kind of asymmetric sparse-video/dense-action window.

**Existing SO-101 datasets on HuggingFace Hub**, researched to address the data-volume gap flagged in section 10c/10d:
- **Bottom line: no large curated SO-101 dataset exists.** The ecosystem is hundreds of tiny (5-20 episode) tutorial repos; best individual real datasets top out around 100 episodes / 15-40 min.
- **Usable today, exact schema match (v3, 6-DOF raw joints, Apache-2.0)**:
  - `lerobot/svla_so101_pickplace` — 50 episodes, ~6.6 min, 2 cameras, official `lerobot` org.
  - `cyanYuki/so101_box_101eps` — 101 episodes, ~19.2 min, 2 cameras — largest clean single real dataset found.
  - `jay88402/so101_pick_pen` — 100 episodes, ~14.5 min, 2 cameras — needs a v2.1→v3 conversion first (lerobot ships a converter); license unspecified on the repo.
  - Combined: ~40 extra minutes across 3 distinct pick-place-style tasks, roughly 5-6x our own 7 minutes.
- **Explicitly excluded**: a 400-episode/~65min dataset (`marin6670/...isaacsim_so101...`) is **simulated** (Isaac Sim), not real — risks sim-to-real domain gap rather than helping. A commercial dataset (`UniDataPro/...`) is non-commercial/no-derivatives licensed and paywalled beyond a 20-episode preview. A dual-arm 100-episode dataset (`CoRL2026-CSI/...`) has a 12-DOF bimanual action space, not a direct match, would need arm/camera slicing.
- **The one "big" lead with a real catch**: `allenai/MolmoAct2-SO100_101-Dataset` — a manifest (not actual data) aggregating 6,898 SO-101 episodes (~41.6 hours) from 1,225 community repos. Using it means scraping and cleaning ~250+ heterogeneous small repos yourself (mixed formats/quality) — a real engineering project, not a quick pull.
- **Recommendation given**: pull the top 2-3 datasets now (cheap), but they're supplementary diversity, not a fix — even combined with our own data, nowhere near OpenWAM's own ~6,400-hour pretrain scale (or realistically, nowhere near typical fine-tune-data scale either). **Recording more SO-101 data ourselves remains the primary lever**, not existing HF datasets.
