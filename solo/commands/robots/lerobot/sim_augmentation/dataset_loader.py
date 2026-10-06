"""
Generic loader for LeRobot v3 datasets, used as input to the sim-augmentation
pipeline (FK batch conversion -> Isaac Mimic). Works with any SO-101 LeRobot
dataset: pass either a HuggingFace Hub repo id ("org/name") or a local path
to an already-materialized dataset directory.

Deliberately has zero Isaac/torch dependency so it can be imported standalone
(e.g. for inspection on this Mac) as well as from inside the Isaac Lab venv
on a GPU pod.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


@dataclass
class EpisodeInfo:
    episode_index: int
    task: str
    length: int


@dataclass
class LeRobotDatasetHandle:
    """Resolved, locally-available LeRobot dataset."""

    root: Path
    info: dict
    episodes: list[EpisodeInfo]

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def joint_names(self) -> list[str]:
        """Real joint names as recorded, e.g. ['shoulder_pan', 'shoulder_lift', ...]."""
        action_names = self.info["features"]["action"]["names"]
        return [n[: -len(".pos")] if n.endswith(".pos") else n for n in action_names]

    def episode(self, episode_index: int) -> EpisodeInfo:
        for ep in self.episodes:
            if ep.episode_index == episode_index:
                return ep
        raise KeyError(f"episode {episode_index} not found in dataset at {self.root}")

    def load_episode_frames(self, episode_index: int) -> pd.DataFrame:
        """
        Real per-frame action/state data for one episode, sorted by frame_index.
        Does not assume 1 parquet file == 1 episode (LeRobot v3 may pack several
        small episodes into one chunk file) -- filters by episode_index instead.
        """
        data_files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"no data parquet files found under {self.root / 'data'}")

        frames = []
        for f in data_files:
            df = pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "observation.state", "timestamp"])
            match = df[df["episode_index"] == episode_index]
            if not match.empty:
                frames.append(match)
                # LeRobot v3 chunk files are contiguous per episode in practice;
                # once we've seen the episode and left its range, later files
                # won't contain it again, but we don't rely on that -- just
                # keep scanning all files for correctness on any packing.
        if not frames:
            raise KeyError(f"episode {episode_index} has no rows in any data file under {self.root / 'data'}")

        out = pd.concat(frames, ignore_index=True)
        return out.sort_values("frame_index").reset_index(drop=True)


def _looks_like_repo_id(dataset: str) -> bool:
    return _REPO_ID_RE.match(dataset) is not None and not Path(dataset).exists()


def resolve_dataset(dataset: str, cache_dir: Optional[str] = None) -> LeRobotDatasetHandle:
    """
    Resolve `dataset` into a local LeRobotDatasetHandle.

    `dataset` is either:
      - a HuggingFace Hub dataset repo id, e.g. "vivekgr92/lerobot-dataset"
        (downloaded via huggingface_hub.snapshot_download), or
      - a local filesystem path to an already-materialized LeRobot v3 dataset
        directory (containing meta/info.json, data/, etc).
    """
    if _looks_like_repo_id(dataset):
        from huggingface_hub import snapshot_download

        # Default cache location matches LeRobot's own convention
        # (~/.cache/huggingface/lerobot/<repo_id>) rather than a pod-specific
        # path -- this is portable across this Mac and any GPU pod, and
        # reuses a dataset already cached there by this project's own
        # recording/replay commands instead of re-downloading it.
        default_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / dataset
        local_dir = snapshot_download(
            repo_id=dataset,
            repo_type="dataset",
            local_dir=cache_dir or str(default_dir),
        )
        root = Path(local_dir)
    else:
        root = Path(dataset).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"dataset path does not exist: {root}")

    info = json.loads((root / "meta" / "info.json").read_text())

    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ep_rows = []
    for f in episode_files:
        edf = pd.read_parquet(f, columns=["episode_index", "tasks", "length"])
        ep_rows.append(edf)
    episodes_df = pd.concat(ep_rows, ignore_index=True).sort_values("episode_index")

    episodes = []
    for _, row in episodes_df.iterrows():
        task_list = list(row["tasks"]) if row["tasks"] is not None else []
        task = str(task_list[0]) if len(task_list) > 0 else ""
        episodes.append(
            EpisodeInfo(
                episode_index=int(row["episode_index"]),
                task=task,
                length=int(row["length"]),
            )
        )

    return LeRobotDatasetHandle(root=root, info=info, episodes=episodes)


def parse_episode_selector(selector: str, available: list[EpisodeInfo]) -> list[int]:
    """
    Parse an episode-selection string into a concrete list of episode indices.
    Supports: "all", "0,2,5", "0-3", or a mix "0-2,5,7-9".
    """
    all_indices = [e.episode_index for e in available]
    if selector.strip().lower() == "all":
        return all_indices

    result: list[int] = []
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))

    missing = [i for i in result if i not in all_indices]
    if missing:
        raise ValueError(f"requested episode(s) {missing} not present in dataset (available: {all_indices})")
    return result
