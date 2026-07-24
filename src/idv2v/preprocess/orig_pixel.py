#!/usr/bin/env python3
"""
Foreground-on-gray condition video: for each frame, keep pixels inside the
per-frame SAM3 mask, fill the rest with gray (127). Preserves the source
video's FPS, resolution, and frame count.

Inputs:
  --video_path             source video
  --mask_folder            dir of frame_<k>/ subfolders (produced by sam3.py)
  --mask_image_file_name   mask filename inside each frame_<k>/  (default sam3Mask_id_all.png)
  --result_save_path       output .mp4

If a frame_<k>/ folder or its mask is missing, frame k becomes fully gray.
Mask resolution must match video resolution exactly (no resizing).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, Iterable

import imageio
import numpy as np


FRAME_DIR_RE = re.compile(r"^frame_(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply per-frame masks to a video and gray-out the background.")
    p.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
    p.add_argument("--mask_folder", type=str, required=True, help="Folder containing frame_{i}/ subfolders.")
    p.add_argument(
        "--mask_image_file_name",
        type=str,
        default="sam3Mask_id_all.png",
        help="Mask filename inside each frame_{i}/ folder (default: sam3Mask_id_all.png).",
    )
    p.add_argument("--result_save_path", type=str, required=True, help="Path to the output video (e.g., out.mp4).")
    return p.parse_args()


def _get_video_meta(reader) -> Tuple[float, Optional[int]]:
    """
    Return (fps, nframes_if_known).
    Note: some formats don't provide reliable nframes; we still preserve exact frames by streaming.
    """
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 25.0))
    nframes = meta.get("nframes", None)
    try:
        nframes = int(nframes) if nframes is not None else None
    except Exception:
        nframes = None
    return fps, nframes


def _list_existing_frame_indices(mask_folder: Path) -> Iterable[int]:
    """Yield all indices i for which a directory named frame_{i} exists under mask_folder."""
    if not mask_folder.exists():
        return []
    out = []
    for child in mask_folder.iterdir():
        if not child.is_dir():
            continue
        m = FRAME_DIR_RE.match(child.name)
        if m:
            out.append(int(m.group(1)))
    out.sort()
    return out


def _load_mask_bool(mask_path: Path, expected_hw: Tuple[int, int]) -> np.ndarray:
    """
    Load a mask image and convert it to a boolean mask of shape (H, W).
    Non-zero pixels are treated as inside-mask.
    """
    mask = imageio.imread(mask_path)  # uint8, could be (H,W) or (H,W,C)
    if mask.ndim == 3:
        # If RGB/RGBA, collapse channels: any non-zero in any channel counts as inside.
        mask = np.max(mask, axis=2)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D after processing, got shape {mask.shape} from {mask_path}")

    h, w = mask.shape
    exp_h, exp_w = expected_hw
    if (h, w) != (exp_h, exp_w):
        raise ValueError(
            f"Mask resolution mismatch for {mask_path}: got {(h, w)}, expected {(exp_h, exp_w)} "
            f"(video resolution)."
        )

    return mask > 0


def apply_masks_to_video(
    video_path: Path,
    mask_folder: Path,
    mask_image_file_name: str,
    result_save_path: Path,
    gray_value: int = 127,
) -> None:
    """
    Stream the input video frame-by-frame, apply the per-frame mask if it exists,
    and write the composited frames to result_save_path.
    """
    assert video_path.exists(), f"Video does not exist: {video_path}"
    assert mask_folder.exists(), f"Mask folder does not exist: {mask_folder}"
    result_save_path.parent.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(str(video_path), format="ffmpeg")
    fps, _ = _get_video_meta(reader)

    # Determine video resolution from the first frame (then restart reader).
    try:
        first_frame = reader.get_data(0)
    except Exception as e:
        reader.close()
        raise RuntimeError(f"Failed to read first frame from video: {video_path}") from e

    if first_frame.ndim != 3 or first_frame.shape[2] < 3:
        reader.close()
        raise ValueError(f"Expected RGB video frames, got shape {first_frame.shape} from {video_path}")

    H, W = first_frame.shape[0], first_frame.shape[1]

    # Every frame_{i} folder index must be within the video's frame count.
    # Prefer count_frames(); if it's unavailable, defer the bound check to the
    # streaming loop below (where the true frame count is known).
    try:
        video_num_frames = reader.count_frames()
    except Exception:
        video_num_frames = None

    existing_indices = list(_list_existing_frame_indices(mask_folder))
    if video_num_frames is not None:
        for i in existing_indices:
            assert 0 <= i < video_num_frames, (
                f"Found mask directory frame_{i} but video has only {video_num_frames} frames "
                f"(0..{video_num_frames - 1})."
            )

    # Re-open reader to stream from frame 0 (more robust than seeking in some codecs).
    reader.close()
    reader = imageio.get_reader(str(video_path), format="ffmpeg")

    # macro_block_size=None prevents ffmpeg writers from padding to multiples of 16 (e.g., 1088).
    writer = imageio.get_writer(
        str(result_save_path),
        fps=fps,
        macro_block_size=None,
    )

    frame_count = 0
    try:
        for frame_idx, frame in enumerate(reader):
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise ValueError(f"Frame {frame_idx} has unexpected shape {frame.shape} (expected HxWx3+).")
            if (frame.shape[0], frame.shape[1]) != (H, W):
                raise ValueError(
                    f"Video resolution changed at frame {frame_idx}: got {(frame.shape[0], frame.shape[1])}, "
                    f"expected {(H, W)}."
                )

            # Default: no mask -> all gray
            out = np.full((H, W, 3), gray_value, dtype=np.uint8)

            # If a mask exists for this frame, keep original pixels inside it
            mask_path = mask_folder / f"frame_{frame_idx}" / mask_image_file_name
            if mask_path.exists() and mask_path.is_file():
                mask_bool = _load_mask_bool(mask_path, (H, W))
                # Ensure frame is uint8 RGB
                rgb = frame[:, :, :3].astype(np.uint8, copy=False)
                out[mask_bool] = rgb[mask_bool]

            writer.append_data(out)
            frame_count += 1

        # Fallback bound check when count_frames() was unavailable: now that the
        # whole video has streamed, validate mask indices against the real count.
        if video_num_frames is None:
            for i in existing_indices:
                assert 0 <= i < frame_count, (
                    f"Found mask directory frame_{i} but video has only {frame_count} frames "
                    f"(0..{frame_count - 1})."
                )

    finally:
        reader.close()
        writer.close()


def main() -> None:
    args = parse_args()
    apply_masks_to_video(
        video_path=Path(args.video_path),
        mask_folder=Path(args.mask_folder),
        mask_image_file_name=args.mask_image_file_name,
        result_save_path=Path(args.result_save_path),
    )


if __name__ == "__main__":
    main()
