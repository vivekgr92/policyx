"""
Camera intrinsics calibration - step 1 of the AprilTag-based object-pose
pipeline for Isaac Mimic / sim-augmentation work (see project history: Isaac
Lab Mimic needs per-frame object poses our recorded data doesn't have; the
plan is to backfill them via AprilTags detected in the recorded video, which
first requires knowing this camera's real intrinsics to turn a 2D tag
detection into a metric pose).

Run as a module, NOT as a direct script path - this directory contains an
unrelated `lerobot.py` file that shadows the real `lerobot` package under
direct-script invocation (same issue documented in vlm_judge_playground.py):

    python3 -m solo.commands.robots.lerobot.camera_calibration generate-pattern
    python3 -m solo.commands.robots.lerobot.camera_calibration capture --output-dir ~/.solo/camera_calib_images
    python3 -m solo.commands.robots.lerobot.camera_calibration solve --image-dir ~/.solo/camera_calib_images --label front
    python3 -m solo.commands.robots.lerobot.camera_calibration run --label front   # capture + solve in one go

Output schema (~/.solo/camera_calibration.json), keyed by an arbitrary camera
label (e.g. the camera "angle" used elsewhere in this project - "front",
"top", etc. - see cameras.py's setup_camera_mapping):

    {
      "<label>": {
        "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "dist_coeffs": [k1, k2, p1, p2, k3],
        "image_width": int,
        "image_height": int,
        "reprojection_error_px": float,          # overall RMS, OpenCV's own quality metric
        "per_image_reprojection_error_px": [float, ...],
        "num_images_used": int,
        "num_images_skipped": int,
        "pattern": {"squares_x": int, "squares_y": int, "square_size_mm": float},
        "calibrated_at": "<ISO8601 UTC>"
      },
      ...
    }

Downstream consumers (the AprilTag pose pipeline, the hand-eye/extrinsics
calibration step) should import and call `load_camera_calibration(label)`
rather than reading this file directly, in case the schema needs to change
later.
"""

import datetime
import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import typer

from solo.config import CONFIG_DIR

CALIBRATION_PATH = os.path.join(CONFIG_DIR, "camera_calibration.json")

# "Squares" here means full checkerboard squares (e.g. a 10x7-square board),
# NOT the internal-corner count cv2.findChessboardCorners/calibrateCamera
# actually operate on (squares - 1 per dimension) - see _pattern_size().
# 10x7 squares @ 25mm is a common, easy-to-print default with 54 internal
# corners - enough points per view for a stable solve without needing a huge
# sheet of paper.
DEFAULT_SQUARES_X = 10
DEFAULT_SQUARES_Y = 7
DEFAULT_SQUARE_SIZE_MM = 25.0
DEFAULT_DPI = 300
DEFAULT_NUM_IMAGES = 25


def _pattern_size(squares_x: int, squares_y: int) -> Tuple[int, int]:
    """cv2's chessboard functions key on internal corner count, not square
    count - a 10x7-square board has 9x6 internal corners."""
    return (squares_x - 1, squares_y - 1)


def generate_checkerboard_pattern(
    output_path: str,
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
    dpi: int = DEFAULT_DPI,
) -> str:
    """Render a checkerboard at an exact, known physical size (given the
    printer actually prints at 100% scale / "no fit to page" - a 50mm scale
    bar is included so the user can verify this with a ruler after
    printing, since a miscalibrated square size would silently corrupt every
    downstream metric pose)."""
    import cv2

    mm_per_inch = 25.4
    px_per_mm = dpi / mm_per_inch
    square_px = int(round(square_size_mm * px_per_mm))
    board_w = squares_x * square_px
    board_h = squares_y * square_px

    margin = square_px
    header_h = square_px  # room for the instruction label above the board
    footer_h = int(square_px * 0.8)  # room for the ruler/scale bar below

    img_h = header_h + board_h + footer_h + 2 * margin
    img_w = board_w + 2 * margin
    img = np.full((img_h, img_w), 255, dtype=np.uint8)

    board_y0 = margin + header_h
    board_x0 = margin
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                y0 = board_y0 + row * square_px
                x0 = board_x0 + col * square_px
                img[y0 : y0 + square_px, x0 : x0 + square_px] = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, square_px / 180.0)
    label = (
        f"{squares_x}x{squares_y} squares, {square_size_mm:.1f}mm/square "
        f"- PRINT AT 100% SCALE (no 'fit to page')"
    )
    cv2.putText(img, label, (margin, margin + int(header_h * 0.6)), font, font_scale, 0, 2)

    # 50mm scale bar with tick marks, for a physical ruler check post-print.
    bar_y = board_y0 + board_h + int(footer_h * 0.4)
    bar_x0 = board_x0
    bar_len_px = int(round(50.0 * px_per_mm))
    tick_h = max(4, square_px // 10)
    cv2.line(img, (bar_x0, bar_y), (bar_x0 + bar_len_px, bar_y), 0, 2)
    cv2.line(img, (bar_x0, bar_y - tick_h), (bar_x0, bar_y + tick_h), 0, 2)
    cv2.line(img, (bar_x0 + bar_len_px, bar_y - tick_h), (bar_x0 + bar_len_px, bar_y + tick_h), 0, 2)
    cv2.putText(
        img, "50mm - verify with a ruler after printing", (bar_x0, bar_y + tick_h + int(0.35 * footer_h)),
        font, font_scale * 0.8, 0, 1,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path


def _object_points(pattern_size: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return objp


def find_corners_in_image(image, pattern_size: Tuple[int, int]):
    """`image` may be a file path or an already-loaded array (BGR or
    grayscale). Returns (found, corners, (width, height))."""
    import cv2

    if isinstance(image, str):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
    )
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners, (gray.shape[1], gray.shape[0])


def solve_camera_intrinsics(
    image_paths: List[str],
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
) -> dict:
    import cv2

    pattern_size = _pattern_size(squares_x, squares_y)
    objp = _object_points(pattern_size, square_size_mm)

    objpoints, imgpoints, used, skipped = [], [], [], []
    image_size = None
    for path in image_paths:
        found, corners, size = find_corners_in_image(path, pattern_size)
        if not found:
            skipped.append(path)
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        image_size = size
        used.append(path)

    if len(objpoints) < 5:
        raise ValueError(
            f"Only {len(objpoints)} of {len(image_paths)} images had detectable "
            f"{squares_x}x{squares_y}-square corners (need >=5 for a meaningful "
            f"solve, 15-20+ recommended for a good one). Skipped: {skipped}"
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    per_image_errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_image_errors.append(float(err))

    return {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "reprojection_error_px": float(rms),
        "per_image_reprojection_error_px": per_image_errors,
        "num_images_used": len(objpoints),
        "num_images_skipped": len(skipped),
        "pattern": {"squares_x": squares_x, "squares_y": squares_y, "square_size_mm": square_size_mm},
    }


def save_camera_calibration(label: str, result: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = {}
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, "r") as f:
            data = json.load(f)
    result = dict(result)
    result["calibrated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    data[label] = result
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_camera_calibration(label: str) -> Optional[dict]:
    """Real loader for downstream consumers (AprilTag pipeline, hand-eye
    extrinsics calibration) - returns None if this camera hasn't been
    calibrated yet rather than raising, since callers may want to prompt
    the user to run calibration instead of crashing."""
    if not os.path.exists(CALIBRATION_PATH):
        return None
    with open(CALIBRATION_PATH, "r") as f:
        data = json.load(f)
    return data.get(label)


def _quality_label(rms_error_px: float) -> str:
    # OpenCV's own documented guidance for RMS reprojection error quality.
    if rms_error_px < 0.5:
        return "excellent"
    if rms_error_px < 1.0:
        return "good"
    if rms_error_px < 1.5:
        return "marginal - consider recapturing with more varied angles/distances"
    return "poor - recapture recommended, calibration is unreliable at this error"


def _print_quality_report(result: dict) -> None:
    err = result["reprojection_error_px"]
    typer.echo(f"📐 RMS reprojection error: {err:.3f} px ({_quality_label(err)})")
    typer.echo(
        f"   Used {result['num_images_used']} images, skipped {result['num_images_skipped']} "
        f"(no corners detected)"
    )


def capture_calibration_images(
    output_dir: str,
    camera_id: Optional[int] = None,
    num_target: int = DEFAULT_NUM_IMAGES,
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    label: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Live capture loop: SPACE saves the current frame (only when the board
    is actually detected in it - no point saving a frame calibrateCamera
    will just skip later), Q/Esc finishes early. Reuses this project's own
    camera detection rather than guessing an OpenCV index, same pattern as
    vlm_judge_playground.py's _pick_camera_index()."""
    import cv2

    if camera_id is None:
        from solo.commands.robots.lerobot.cameras import find_available_cameras

        cameras = find_available_cameras()
        if not cameras:
            raise RuntimeError("No cameras detected (checked OpenCV + RealSense).")
        camera_id = cameras[0].get("id", 0)
        label = label or cameras[0].get("angle") or str(camera_id)
        typer.echo(f"📷 Using camera: {cameras[0].get('type', 'Unknown')} (ID: {camera_id})")
    label = label or str(camera_id)

    os.makedirs(output_dir, exist_ok=True)
    pattern_size = _pattern_size(squares_x, squares_y)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_id}.")

    typer.echo("Move the checkerboard to vary angle/distance/tilt. SPACE = capture, Q = finish.")
    saved: List[str] = []
    try:
        while len(saved) < num_target:
            ret, frame = cap.read()
            if not ret:
                continue
            found, corners, _ = find_corners_in_image(frame, pattern_size)

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern_size, corners, found)
            status = f"Captured: {len(saved)}/{num_target}  corners_found={found}"
            color = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Camera calibration capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and found:
                path = os.path.join(output_dir, f"calib_{label}_{len(saved):03d}.png")
                cv2.imwrite(path, frame)
                saved.append(path)
                typer.echo(f"✅ Captured {path} ({len(saved)}/{num_target})")
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return label, saved


app = typer.Typer(help="Camera intrinsics calibration for the AprilTag object-pose pipeline.")


@app.command("generate-pattern")
def cmd_generate_pattern(
    output: str = typer.Option("checkerboard_pattern.png", help="Output PNG path."),
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
    dpi: int = DEFAULT_DPI,
):
    path = generate_checkerboard_pattern(output, squares_x, squares_y, square_size_mm, dpi)
    typer.echo(f"✅ Saved pattern to {path}")
    typer.echo("   Print at 100% scale ('no fit to page') and verify the 50mm scale bar with a ruler.")


@app.command("capture")
def cmd_capture(
    output_dir: str = typer.Option("~/.solo/camera_calib_images", help="Where to save captured frames."),
    camera_id: Optional[int] = typer.Option(None, help="OpenCV camera index; auto-detected if omitted."),
    num_images: int = DEFAULT_NUM_IMAGES,
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    label: Optional[str] = typer.Option(None, help="Label to save this calibration under."),
):
    output_dir = os.path.expanduser(output_dir)
    label, saved = capture_calibration_images(output_dir, camera_id, num_images, squares_x, squares_y, label)
    typer.echo(f"Captured {len(saved)} images to {output_dir} for camera '{label}'.")


@app.command("solve")
def cmd_solve(
    image_dir: str = typer.Option(..., help="Directory of captured calibration images."),
    label: str = typer.Option(..., help="Label to save this calibration under."),
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
):
    image_dir = os.path.expanduser(image_dir)
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    if not paths:
        typer.echo(f"❌ No .png/.jpg images found in {image_dir}")
        raise typer.Exit(1)
    result = solve_camera_intrinsics(paths, squares_x, squares_y, square_size_mm)
    save_camera_calibration(label, result)
    _print_quality_report(result)
    typer.echo(f"✅ Saved calibration for '{label}' to {CALIBRATION_PATH}")


@app.command("run")
def cmd_run(
    camera_id: Optional[int] = typer.Option(None, help="OpenCV camera index; auto-detected if omitted."),
    label: Optional[str] = typer.Option(None, help="Label to save this calibration under."),
    num_images: int = DEFAULT_NUM_IMAGES,
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
):
    """Capture + solve in one go."""
    output_dir = os.path.expanduser("~/.solo/camera_calib_images")
    label, saved = capture_calibration_images(output_dir, camera_id, num_images, squares_x, squares_y, label)
    if not saved:
        typer.echo("❌ No images captured.")
        raise typer.Exit(1)
    result = solve_camera_intrinsics(saved, squares_x, squares_y, square_size_mm)
    save_camera_calibration(label, result)
    _print_quality_report(result)
    typer.echo(f"✅ Saved calibration for '{label}' to {CALIBRATION_PATH}")


if __name__ == "__main__":
    app()
