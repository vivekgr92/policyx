"""
AprilTag-based object pose extraction from recorded LeRobot video, for the
Isaac Mimic offline-annotation pipeline (see mimic_hdf5_export.py).

Generic by design: pass whatever name->tag_id mapping matches what's
physically tagged for a given dataset/task (e.g. {"cup_a": 0, "cup_b": 1}) --
nothing here is hardcoded to a specific object count or dataset.

Depends on two real calibration artifacts produced elsewhere in this package:
  - ~/.solo/camera_calibration.json   (camera_calibration.py -- intrinsics)
  - ~/.solo/camera_extrinsics.json    (hand_eye_calibration.py -- camera-to-
    robot-base transform, keyed "R_cam_to_base"/"t_cam_to_base")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from solo.commands.robots.lerobot.camera_calibration import load_camera_calibration

_EXTRINSICS_PATH = Path.home() / ".solo" / "camera_extrinsics.json"

# cv2.aruco's predefined AprilTag dictionaries - verified empirically (not just
# assumed) that a tag generated via cv2.aruco.generateImageMarker is detected
# by pupil_apriltags (the library apriltag_pose.py's detection side uses) with
# zero hamming error and high decision_margin, so these two libraries' tag36h11
# encodings are bit-compatible despite being different codebases.
_ARUCO_FAMILY_MAP = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag25h9": "DICT_APRILTAG_25H9",
    "tag16h5": "DICT_APRILTAG_16h5",
}


def generate_apriltag_image(
    tag_id: int,
    output_path: str,
    tag_family: str = "tag36h11",
    size_mm: float = 30.0,
    dpi: int = 300,
    quiet_zone_modules: float = 1.0,
) -> str:
    """Render a single AprilTag at an exact, known physical size, with a white
    quiet-zone border (AprilTag detection needs a clear white margin around
    the black tag border - without one, detection becomes unreliable
    precisely at the corners/edges where you need it most). `size_mm` is the
    tag's own black-bordered square (the quiet zone is added on top of that,
    not counted within it), matching how you'd actually measure it with a
    ruler after printing."""
    import cv2

    if tag_family not in _ARUCO_FAMILY_MAP:
        raise ValueError(f"Unsupported tag family '{tag_family}'. Supported: {list(_ARUCO_FAMILY_MAP)}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _ARUCO_FAMILY_MAP[tag_family]))

    mm_per_inch = 25.4
    px_per_mm = dpi / mm_per_inch
    tag_px = int(round(size_mm * px_per_mm))
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_px)

    # tag36h11 is a 10x10-module grid (8x8 data+inner-border modules plus the
    # 1-module black border aruco already renders) - one module width is a
    # reasonable, standard quiet-zone size.
    module_px = tag_px // 10
    quiet_px = int(round(module_px * quiet_zone_modules))

    label_h = max(40, tag_px // 6)
    img_w = tag_px + 2 * quiet_px
    img_h = tag_px + 2 * quiet_px + label_h
    img = np.full((img_h, img_w), 255, dtype=np.uint8)
    img[label_h + quiet_px : label_h + quiet_px + tag_px, quiet_px : quiet_px + tag_px] = marker

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, tag_px / 500.0)
    label = f"{tag_family} ID={tag_id}  {size_mm:.1f}mm - PRINT AT 100% SCALE"
    cv2.putText(img, label, (quiet_px, int(label_h * 0.7)), font, font_scale, 0, 2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path


app = typer.Typer(help="AprilTag generation + object-pose extraction for the Mimic pipeline.")


@app.command("generate")
def cmd_generate(
    output_dir: str = typer.Option("~/.solo/apriltags", help="Directory to save generated tag images."),
    tag_ids: str = typer.Option("0,1,2,3", help="Comma-separated tag IDs to generate."),
    tag_family: str = typer.Option("tag36h11", help="AprilTag family."),
    size_mm: float = typer.Option(30.0, help="Physical size of the tag's own black-bordered square, in mm."),
    dpi: int = typer.Option(300, help="Print resolution."),
):
    """Generate printable AprilTag images, e.g. for cup A (ID 0), cup B (ID 1),
    cup C (ID 2), and the gripper calibration target (ID 3)."""
    output_dir = os.path.expanduser(output_dir)
    ids = [int(x.strip()) for x in tag_ids.split(",") if x.strip()]
    for tag_id in ids:
        path = os.path.join(output_dir, f"apriltag_{tag_family}_id{tag_id}.png")
        generate_apriltag_image(tag_id, path, tag_family, size_mm, dpi)
        typer.echo(f"✅ Saved tag ID {tag_id} to {path}")
    typer.echo("Print at 100% scale ('no fit to page') and verify size with a ruler before mounting.")


if __name__ == "__main__":
    app()


def _load_extrinsics(camera_angle: str) -> np.ndarray:
    """Real 4x4 cam_to_base transform saved by hand_eye_calibration.py."""
    if not _EXTRINSICS_PATH.is_file():
        raise FileNotFoundError(
            f"No hand-eye extrinsics calibration found at {_EXTRINSICS_PATH}. Run "
            "hand_eye_calibration.py (run_hand_eye_calibration()) first."
        )
    data = json.loads(_EXTRINSICS_PATH.read_text())
    entry = data.get(camera_angle)
    if entry is None and len(data) == 1:
        entry = next(iter(data.values()))
    if entry is None:
        raise KeyError(
            f"No extrinsics entry for camera '{camera_angle}' in {_EXTRINSICS_PATH}. "
            f"Keys present: {list(data.keys())}"
        )
    T = np.eye(4)
    T[:3, :3] = np.array(entry["R_cam_to_base"], dtype=np.float64)
    T[:3, 3] = np.array(entry["t_cam_to_base"], dtype=np.float64).reshape(3)
    return T


def detect_object_poses_in_video(
    video_path: str,
    camera_angle: str,
    name_to_tag_id: dict[str, int],
    tag_sizes_m: "dict[str, float] | float" = 0.015,  # real printed cup tags are 15mm
    tag_family: str = "tag36h11",
    start_time_s: Optional[float] = None,
    end_time_s: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """
    Detect AprilTags over every frame of `video_path`, transform each
    detection from camera frame to robot-base frame via the real hand-eye
    extrinsics, and return {object_name: [T,4,4] poses in base frame}.

    `start_time_s`/`end_time_s` (both optional, default = whole video) slice
    out just one episode's frames when `video_path` is a SHARED file packing
    multiple episodes -- real confirmed case on a real dataset
    (vivekgr92/tags: episodes 0 and 1 share one mp4, with real
    from_timestamp/to_timestamp recorded per episode in its own metadata).
    Without this, every episode sharing a file would read the WHOLE file
    (wrong frame count, silently misaligned against that episode's own
    recorded actions) -- see mimic_hdf5_export.py's _episode_video_path.

    Frames where a tag isn't detected (occlusion, typically during grasp)
    are filled by `_interpolate_gaps` -- SLERP for rotation, linear for
    translation, holding the nearest valid pose past either end of the
    episode (no extrapolation).
    """
    import cv2
    from pupil_apriltags import Detector

    calib = load_camera_calibration(camera_angle)
    if calib is None:
        raise FileNotFoundError(
            f"No camera intrinsics calibration found for '{camera_angle}'. Run "
            "camera_calibration.py (generate-pattern/capture/solve, or `run`) first."
        )
    camera_matrix = np.array(calib["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float64)
    cam_to_base = _load_extrinsics(camera_angle)

    if isinstance(tag_sizes_m, (int, float)):
        tag_sizes_m = {name: float(tag_sizes_m) for name in name_to_tag_id}
    distinct_sizes = sorted(set(tag_sizes_m.values()))

    detector = Detector(families=tag_family)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(round(start_time_s * fps)) if start_time_s is not None else 0
    end_frame = int(round(end_time_s * fps)) if end_time_s is not None else None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_poses: dict[str, list[Optional[np.ndarray]]] = {name: [] for name in name_to_tag_id}
    new_camera_matrix = None
    frame_count = 0
    try:
        while True:
            if end_frame is not None and start_frame + frame_count >= end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            h, w = frame.shape[:2]
            if new_camera_matrix is None:
                new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                    camera_matrix, dist_coeffs, (w, h), alpha=0
                )
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            fx, fy = new_camera_matrix[0, 0], new_camera_matrix[1, 1]
            cx, cy = new_camera_matrix[0, 2], new_camera_matrix[1, 2]

            # pupil_apriltags takes one tag_size per detect() call; if objects
            # use different physical tag sizes, detect once per distinct size
            # and merge by tag_id (last write wins, which is fine -- a given
            # tag_id is only ever detected under its own true size's pass).
            detections_by_id = {}
            for size in distinct_sizes:
                for d in detector.detect(
                    gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=size
                ):
                    detections_by_id[d.tag_id] = d

            for name, tag_id in name_to_tag_id.items():
                d = detections_by_id.get(tag_id)
                if d is None:
                    raw_poses[name].append(None)
                    continue
                T_cam = np.eye(4)
                T_cam[:3, :3] = d.pose_R
                T_cam[:3, 3] = d.pose_t.reshape(3)
                raw_poses[name].append(cam_to_base @ T_cam)
    finally:
        cap.release()

    return {name: _interpolate_gaps(poses, name) for name, poses in raw_poses.items()}


def _interpolate_gaps(poses: "list[Optional[np.ndarray]]", object_name: str) -> np.ndarray:
    """SLERP (rotation) + linear (translation) between nearest valid
    detections on either side of a gap; holds the nearest valid pose past
    either end of the sequence rather than extrapolating."""
    from scipy.spatial.transform import Rotation, Slerp

    n = len(poses)
    valid_idx = [i for i, p in enumerate(poses) if p is not None]
    if not valid_idx:
        raise ValueError(
            f"Tag for object '{object_name}' was never detected in any frame of this "
            "episode's video -- cannot interpolate from zero valid detections. Check tag "
            "placement/visibility, tag ID mapping, and tag size for this object."
        )

    out: list[Optional[np.ndarray]] = list(poses)
    for i in range(n):
        if out[i] is not None:
            continue
        before = max((j for j in valid_idx if j < i), default=None)
        after = min((j for j in valid_idx if j > i), default=None)
        if before is None:
            out[i] = poses[after]
        elif after is None:
            out[i] = poses[before]
        else:
            t = (i - before) / (after - before)
            rotations = Rotation.from_matrix(np.stack([poses[before][:3, :3], poses[after][:3, :3]]))
            slerp = Slerp([0, 1], rotations)
            R_interp = slerp(t).as_matrix()
            trans_interp = poses[before][:3, 3] * (1 - t) + poses[after][:3, 3] * t
            T = np.eye(4)
            T[:3, :3] = R_interp
            T[:3, 3] = trans_interp
            out[i] = T

    return np.stack(out, axis=0)
