"""
Generic LeRobot dataset -> Isaac Mimic HDF5 exporter.

Run as a module (this directory contains an unrelated `lerobot.py` that
shadows the real `lerobot` package under direct-script invocation -- same
issue documented in vlm_judge_playground.py):

    python3 -m solo.commands.robots.lerobot.sim_augmentation.mimic_hdf5_export \\
        --dataset vivekgr92/lerobot-dataset --output ~/.solo/mimic_export.hdf5 \\
        --tag-ids '{"cup_a": 0, "cup_b": 1}'

Produces the `obs/datagen_info/*` annotations isaaclab_mimic's
DataGenInfoPool reads via a flat static HDF5 read -- confirmed against the
real v2.3.1 source (datagen_info_pool.py's `_add_episode`), NOT assumed:
    obs/datagen_info/eef_pose/<eef_name>           [T,4,4]  (one group per robot eef -- CHANGED
                                                              from a flat [T,4,4] dataset: confirmed
                                                              via live isaaclab 3.0.0-beta2 testing
                                                              that DataGenerator.select_source_demo
                                                              indexes DatagenInfo.eef_pose by eef_name,
                                                              same as object_poses; the 2.3.1-era flat
                                                              layout this originally matched no longer
                                                              works. <eef_name> must match the key used
                                                              in the env cfg's subtask_configs dict
                                                              (e.g. "robot").)
    obs/datagen_info/object_pose/<name>            [T,4,4]  (one group per tracked object)
    obs/datagen_info/target_eef_pose/<eef_name>    [T,4,4]  (same per-eef-name grouping as eef_pose)
    obs/datagen_info/subtask_term_signals/<name>   [T,1]    (one per NON-FINAL subtask only --
                                                              the last subtask needs no signal,
                                                              its end is just episode end)
    actions                                        [T,6]    (gripper_action is derived from this
                                                              BY MIMIC ITSELF via
                                                              actions_to_gripper_actions = actions[:,-1:],
                                                              matching our gripper-is-last-channel layout
                                                              -- confirmed from franka_stack_ik_rel_mimic_env.py)

Top-level HDF5 structure matches isaaclab.utils.datasets.HDF5DatasetFileHandler
exactly (confirmed from its real v2.3.1 source -- deliberately re-implemented
here with plain h5py rather than importing that class, so this script stays
usable with zero Isaac Lab installed, same portability goal as dataset_loader.py):
    /data                           (attrs: total=<int>, env_args=<json str {"env_name":..., "type":2}>)
      /data/demo_0                  (attrs: num_samples=<int>)
        actions
        obs/datagen_info/...
      /data/demo_1
        ...

NOT produced here (needs a live Isaac Lab scene to capture real physics
state, out of scope for this offline/video-only script): `states`/
`initial_state`, which DataGenerator.generate()'s live physics
re-execution step needs to reset each generated trajectory's starting scene.
Authoring that (a real SO-101 Isaac Lab task + MimicEnvCfg/SubTaskConfig) is
separate, already-flagged follow-on work -- this script only produces the
annotation data DataGenInfoPool itself reads from a finished HDF5 file.

target_eef_pose definition (verified against NVIDIA's real Franka IK-rel env,
franka_stack_ik_rel_mimic_env.py): their action_to_target_eef_pose() computes
target_pos = curr_eef_pos + delta_position, i.e. "the pose this frame's
action is trying to reach next." Our recordings are absolute joint-space
actions, not IK-rel deltas, so the direct equivalent is eef_pose shifted one
frame forward: target_eef_pose[t] = eef_pose[t+1], held at eef_pose[t] for
the final frame (no next frame to target).

subtask_term_signals design: our task has no live env to auto-annotate from
(see get_subtask_term_signals() in NVIDIA's real env -- that path needs a
running sim), so this is a real, explicit heuristic derived from the FK
eef_pose + AprilTag object_pose streams this script already computes. Exactly
one non-final subtask boundary, "grasped": for each episode, the "active"
object is whichever tracked object is closest to the gripper at the first
frame where the recorded gripper action crosses below
`grasp_threshold_pct` (open -> closed); the signal is a clean 0...0,1...1
step function from that frame on, as DataGenInfoPool's single-rising-edge
parser requires. The final subtask ("place") gets no signal entry, matching
NVIDIA's own convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import typer

from solo.commands.robots.lerobot.sim_augmentation.apriltag_pose import detect_object_poses_in_video
from solo.commands.robots.lerobot.sim_augmentation.dataset_loader import (
    LeRobotDatasetHandle,
    parse_episode_selector,
    resolve_dataset,
)
from solo.commands.robots.lerobot.sim_augmentation.so101_fk import compute_gripper_pose

# Recorded gripper value below which the gripper is considered closed enough
# to be grasping. Picked conservatively mid-range rather than near either
# extreme, since the exact open/close direction at 0% is UNVERIFIED (see
# so101_joint_mapping.py's GRIPPER_PCT_MAPS_TO_LOWER_AT_ZERO) -- override via
# --grasp-threshold-pct if this doesn't match a given gripper/dataset's real
# convention.
DEFAULT_GRASP_THRESHOLD_PCT = 30.0


def _episode_video_path(dataset_root: Path, episode_index: int, camera_key: str, segment: dict) -> Path:
    """Real LeRobot v3 per-episode video path, from the real chunk_index/
    file_index recorded in this episode's own metadata (meta/episodes/*.parquet)
    -- NOT guessed from the episode index. LeRobot v3 can (and, confirmed on a
    real dataset -- vivekgr92/tags episodes 0 and 1 -- does) pack multiple
    episodes into one physical mp4 per (camera, chunk, file); `segment`'s
    from_timestamp/to_timestamp (see detect_object_poses_in_video's
    start_time_s/end_time_s) is what slices out just this episode's frames."""
    path = (
        dataset_root / "videos" / camera_key
        / f"chunk-{segment['chunk_index']:03d}" / f"file-{segment['file_index']:03d}.mp4"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Episode {episode_index}'s video file does not exist: {path}")
    return path


def _compute_eef_poses(frames: pd.DataFrame, joint_names: list[str]) -> np.ndarray:
    out = []
    for action in frames["action"]:
        recorded = {name: float(action[i]) for i, name in enumerate(joint_names)}
        out.append(compute_gripper_pose(recorded))
    return np.stack(out, axis=0).astype(np.float32)


def _compute_target_eef_pose(eef_pose: np.ndarray) -> np.ndarray:
    target = np.empty_like(eef_pose)
    target[:-1] = eef_pose[1:]
    target[-1] = eef_pose[-1]
    return target


def _compute_grasped_signal(
    eef_pose: np.ndarray,
    object_poses: dict[str, np.ndarray],
    gripper_action: np.ndarray,
    grasp_threshold_pct: float,
) -> tuple[np.ndarray, Optional[str]]:
    """Returns (signal [T,1] float32, active_object_name).

    Detects EVERY grasp/release cycle (contiguous closed segment), not just
    the first -- a real recorded dataset (vivekgr92/tags) confirmed, by
    direct user confirmation, that episodes deliberately contain multiple
    pick-and-place cycles. The earlier "latch forever after first dip below
    threshold" version produced a signal stuck at 1.0 for the entire rest of
    the episode after the first close, which is wrong for this real case --
    the signal now goes back to 0.0 whenever the gripper reopens, correctly
    reflecting repeated grasp/release.

    active_object_name is None if the gripper never closes past threshold
    (suspect episode -- an all-zero signal will likely fail isaaclab_mimic's
    monotonic-boundary check). If different grasp segments are nearest to
    DIFFERENT objects (switching which object is picked within one episode),
    the majority-voted object is returned but this is flagged loudly to the
    caller via the echoed warning, since the current schema (one object_ref
    per subtask) can't represent a per-segment-varying target -- a real,
    not-yet-solved limitation for that specific case, not silently hidden."""
    T = eef_pose.shape[0]
    is_closed = gripper_action.reshape(-1) < grasp_threshold_pct
    if not is_closed.any():
        return np.zeros((T, 1), dtype=np.float32), None

    # Contiguous closed segments = individual grasp events.
    edges = np.diff(is_closed.astype(np.int8))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    if is_closed[0]:
        starts = [0] + starts
    ends = list(np.nonzero(edges == -1)[0] + 1)
    if is_closed[-1]:
        ends = ends + [T]
    segments = list(zip(starts, ends))

    signal = np.zeros((T, 1), dtype=np.float32)
    segment_objects = []
    for seg_start, seg_end in segments:
        signal[seg_start:seg_end] = 1.0
        gripper_pos = eef_pose[seg_start][:3, 3]
        best_name, best_dist = None, None
        for name, poses in object_poses.items():
            dist = float(np.linalg.norm(poses[seg_start][:3, 3] - gripper_pos))
            if best_dist is None or dist < best_dist:
                best_dist, best_name = dist, name
        segment_objects.append(best_name)

    distinct = set(segment_objects)
    if len(distinct) > 1:
        import collections

        counts = collections.Counter(segment_objects)
        active_name = counts.most_common(1)[0][0]
        typer.echo(
            f"⚠️  {len(segments)} grasp segments nearest to DIFFERENT objects {dict(counts)} -- "
            f"using majority '{active_name}'. The current HDF5 schema assumes one object per "
            "episode; a per-segment-varying target isn't represented here."
        )
    else:
        active_name = segment_objects[0] if segment_objects else None

    return signal, active_name


def export_episode(
    dataset: LeRobotDatasetHandle,
    episode_index: int,
    camera_key: str,
    camera_angle: str,
    name_to_tag_id: dict[str, int],
    tag_sizes_m,
    grasp_threshold_pct: float,
) -> dict:
    frames = dataset.load_episode_frames(episode_index)
    joint_names = dataset.joint_names
    actions = np.stack(frames["action"].to_numpy()).astype(np.float32)  # [T, num_joints]

    eef_pose = _compute_eef_poses(frames, joint_names)
    target_eef_pose = _compute_target_eef_pose(eef_pose)

    ep_info = dataset.episode(episode_index)
    segment = ep_info.video_segment(camera_key)
    video_path = _episode_video_path(dataset.root, episode_index, camera_key, segment)
    object_poses = detect_object_poses_in_video(
        str(video_path), camera_angle, name_to_tag_id, tag_sizes_m,
        start_time_s=segment["from_timestamp"], end_time_s=segment["to_timestamp"],
    )
    for name, poses in object_poses.items():
        if poses.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Episode {episode_index}: video frame count ({poses.shape[0]}) for object "
                f"'{name}' does not match recorded action frame count ({actions.shape[0]}) -- "
                "video and parquet data are out of sync for this episode, refusing to silently "
                "truncate/pad."
            )

    gripper_action = actions[:, -1:]
    grasped_signal, active_object = _compute_grasped_signal(
        eef_pose, object_poses, gripper_action, grasp_threshold_pct
    )
    if active_object is None:
        typer.echo(
            f"⚠️  Episode {episode_index}: gripper never closed past {grasp_threshold_pct}% -- no "
            "'grasped' subtask boundary found. subtask_term_signals will be all-zero for this "
            "episode, which will likely fail isaaclab_mimic's monotonic-boundary check downstream. "
            "Check --grasp-threshold-pct and the gripper open/close direction (UNVERIFIED -- see "
            "so101_joint_mapping.py's GRIPPER_PCT_MAPS_TO_LOWER_AT_ZERO) against this dataset."
        )

    return {
        "actions": actions,
        "eef_pose": eef_pose,
        "target_eef_pose": target_eef_pose,
        "object_pose": object_poses,
        "subtask_term_signals": {"grasped": grasped_signal},
        "active_object": active_object,
        "num_samples": int(actions.shape[0]),
    }


def write_hdf5(output_path: Path, env_name: str, episodes_data: list[dict]) -> None:
    """Writes the exact schema confirmed from isaaclab's real
    HDF5DatasetFileHandler.create()/write_episode() source (v2.3.1) --
    deliberately NOT importing that class (heavy torch/isaaclab dependency),
    just matching its real on-disk layout with plain h5py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env_args"] = json.dumps({"env_name": env_name, "type": 2})

        total = 0
        for i, ep in enumerate(episodes_data):
            ep_grp = data_grp.create_group(f"demo_{i}")
            ep_grp.attrs["num_samples"] = ep["num_samples"]

            ep_grp.create_dataset("actions", data=ep["actions"], compression="gzip")

            obs_grp = ep_grp.create_group("obs")
            dg_grp = obs_grp.create_group("datagen_info")
            # Grouped per eef_name ("robot", matching the env cfg's subtask_configs
            # key) rather than a flat dataset -- see module docstring for why.
            eef_pose_grp = dg_grp.create_group("eef_pose")
            eef_pose_grp.create_dataset("robot", data=ep["eef_pose"], compression="gzip")
            target_eef_pose_grp = dg_grp.create_group("target_eef_pose")
            target_eef_pose_grp.create_dataset("robot", data=ep["target_eef_pose"], compression="gzip")

            obj_grp = dg_grp.create_group("object_pose")
            for name, poses in ep["object_pose"].items():
                obj_grp.create_dataset(name, data=poses.astype(np.float32), compression="gzip")

            sig_grp = dg_grp.create_group("subtask_term_signals")
            for name, sig in ep["subtask_term_signals"].items():
                sig_grp.create_dataset(name, data=sig, compression="gzip")

            total += ep["num_samples"]

        data_grp.attrs["total"] = total


app = typer.Typer(help="Export a LeRobot SO-101 dataset to an Isaac Mimic-ready HDF5 annotation file.")


@app.command()
def export(
    dataset: str = typer.Option(..., help="HF Hub repo id (org/name) or local path to a LeRobot v3 dataset."),
    output: str = typer.Option(..., help="Output .hdf5 path."),
    tag_ids: str = typer.Option(..., help='Object name -> AprilTag ID mapping as JSON, e.g. \'{"cup_a": 0, "cup_b": 1}\'.'),
    episodes: str = typer.Option("all", help="Episode selector: 'all', '0,2,5', '0-3', or a mix."),
    camera_key: str = typer.Option("observation.images.front", help="LeRobot dataset video feature key."),
    camera_angle: str = typer.Option("front", help="Camera calibration label (see camera_calibration.py / hand_eye_calibration.py)."),
    tag_size_mm: float = typer.Option(15.0, help="Physical AprilTag size in mm, same for all tracked objects (real printed cup tags are 15mm)."),
    grasp_threshold_pct: float = typer.Option(DEFAULT_GRASP_THRESHOLD_PCT, help="Recorded gripper value below which the gripper is considered closed/grasping."),
    env_name: str = typer.Option("so101_pick_place", help="env_name recorded in the HDF5 env_args (cosmetic, robomimic convention)."),
):
    name_to_tag_id: dict[str, int] = json.loads(tag_ids)
    handle = resolve_dataset(dataset)
    ep_indices = parse_episode_selector(episodes, handle.episodes)

    typer.echo(f"📦 Dataset: {handle.root} ({len(ep_indices)} episode(s) selected)")
    episodes_data = []
    for idx in ep_indices:
        ep_info = handle.episode(idx)
        typer.echo(f"\n▶️  Episode {idx}: \"{ep_info.task}\" ({ep_info.length} frames)")
        ep_data = export_episode(
            handle, idx, camera_key, camera_angle, name_to_tag_id,
            tag_size_mm / 1000.0, grasp_threshold_pct,
        )
        typer.echo(f"   grasped object: {ep_data['active_object']}")
        episodes_data.append(ep_data)

    output_path = Path(output).expanduser()
    write_hdf5(output_path, env_name, episodes_data)
    typer.echo(f"\n✅ Wrote {len(episodes_data)} episode(s) to {output_path}")
    typer.echo(
        "   Note: this file has obs/datagen_info/* + actions only -- it does NOT include "
        "states/initial_state, which isaaclab_mimic's DataGenerator.generate() needs for live "
        "physics re-execution. That requires a real SO-101 Isaac Lab task/scene authored "
        "separately (see project notes)."
    )


if __name__ == "__main__":
    app()
