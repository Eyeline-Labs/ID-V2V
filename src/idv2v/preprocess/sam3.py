"""
SAM3 segmentation + Secret Panda mask cleanup (per-frame).

For each frame, writes <output_dir>/frame_k/ containing:
  sam3Mask_id_all.png          union mask of all detected objects (used by orig_pixel)
  sam3Mask_id_{i}.png          per-object mask
  cropped_bbox_..._input.png   per-object RGB crop
  cropped_bbox_..._sam3mask.png   per-object mask crop

Secret Panda fixes SAM3 mask artifacts (holes, jagged edges, disconnected blobs)
by: hole-fill -> morph close -> bridge gaps -> hole-fill. Applied per-object;
--joint_mask_post_proc also applies it to the union mask.

SAM3 weights: run `hf auth login` first (the model is gated).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

# Monkey-patch: transformers (git main) has a bug in Sam3TrackerVideoConfig where
# `initializer_range` is typed as `int` but defaults to 0.02 (float).
# huggingface_hub >=1.5 strict dataclass validation rejects this mismatch.
# Fix here so it survives `uv sync`. Upstream: configuration_sam3_tracker_video.py:214
from transformers import Sam3TrackerVideoConfig
Sam3TrackerVideoConfig.__dataclass_fields__["initializer_range"].type = float
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Post-processing
try:
    from .secret_panda import secret_panda
except ImportError:
    from secret_panda import secret_panda


# =============================================================================
# Utilities
# =============================================================================
def _as_pil_rgb(frame: Any) -> Image.Image:
    """Normalize a frame (numpy/PIL) to PIL RGB."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        frame = frame[:, :, :3]
    return Image.fromarray(frame.astype(np.uint8), mode="RGB")


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> Tuple[List[Image.Image], float]:
    """
    Load video into a list of PIL RGB frames + fps using Transformers' loader.
    Handles both older (frames, fps) and newer (frames, VideoMetadata) outputs.
    """
    out = load_video(video_path)

    # transformers.video_utils.load_video sometimes returns (frames, fps) or (frames, metadata)
    if isinstance(out, tuple) and len(out) == 2:
        frames, meta = out
    else:
        raise TypeError(f"Unexpected load_video(...) return type: {type(out)}")

    frames = [_as_pil_rgb(f) for f in frames]
    if max_frames is not None:
        frames = frames[:max_frames]

    fps_val = None
    # Newer: meta is VideoMetadata with .fps
    if hasattr(meta, "fps"):
        fps_val = getattr(meta, "fps")
    # Possible dict-like
    elif isinstance(meta, dict):
        fps_val = meta.get("fps", None)

    fps = float(fps_val) if fps_val is not None else 25.0
    return frames, fps



def init_sam3_video(model_path: str = "facebook/sam3", dtype: torch.dtype = torch.bfloat16) -> Tuple[Sam3VideoModel, Sam3VideoProcessor, torch.device]:
    """
    Load SAM3 Video model + processor.
    model_path: HF repo id or local path to checkpoint directory.
    """
    device = Accelerator().device
    model = Sam3VideoModel.from_pretrained(model_path).to(device, dtype=dtype)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained(model_path)
    return model, processor, device


def run_sam3_video(
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    device: torch.device,
    video_frames: List[Image.Image],
    prompt: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[int, Dict[str, Any]]:
    """
    Run Promptable Concept Segmentation (PCS) on the full video with a text prompt (e.g. "person").
    Returns a dict: frame_idx -> processed_outputs.

    Key fields per frame (from processor.postprocess_outputs):
      - object_ids: (N,) stable tracked IDs across frames
      - masks:      (N, H, W) per-instance masks (commonly binary, sometimes float)
      - boxes:      (N, 4) XYXY boxes in absolute pixels
      - scores:     (N,) confidence scores
      - prompt_to_obj_ids: mapping prompt -> list of object IDs (useful for multi-prompt)
    """
    # Pre-loaded video session (best quality vs streaming)
    session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device=device,
        video_storage_device=device,
        dtype=dtype,
    )
    session = processor.add_text_prompt(inference_session=session, text=prompt)  # tracks all instances for this concept

    outputs_per_frame: Dict[int, Dict[str, Any]] = {}
    max_frame_num_to_track = len(video_frames) - 1

    with torch.inference_mode():
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=max_frame_num_to_track,
        ):
            processed = processor.postprocess_outputs(session, model_outputs)
            outputs_per_frame[int(model_outputs.frame_idx)] = processed

    return outputs_per_frame


def pack_per_frame_instances(outputs_per_frame: Dict[int, Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Convert raw processed outputs into a simpler Python structure:
      frame_idx -> list of {id, score, box_xyxy, mask(HxW float/bool)}
    If a frame has no objects, it maps to [].
    """
    packed: Dict[int, List[Dict[str, Any]]] = {}

    for frame_idx, out in outputs_per_frame.items():
        obj_ids = out.get("object_ids", None)
        if obj_ids is None or len(obj_ids) == 0:
            packed[frame_idx] = []
            continue

        # All are typically torch tensors
        obj_ids = obj_ids.detach().cpu().numpy().astype(int)
        scores = out["scores"].detach().cpu().numpy()
        boxes = out["boxes"].detach().cpu().numpy()  # (N,4) XYXY
        masks = out["masks"].detach().cpu().numpy()  # (N,H,W) bool/float

        instances: List[Dict[str, Any]] = []
        for i in range(len(obj_ids)):
            instances.append(
                {
                    "id": int(obj_ids[i]),                 # stable identity across frames (tracking)
                    "score": float(scores[i]),
                    "box_xyxy": boxes[i].astype(float),
                    "mask": masks[i],                      # keep raw mask (bool or float) as SAM3 returns it
                }
            )

        packed[frame_idx] = instances

    return packed


def _color_from_id(obj_id: int) -> Tuple[int, int, int]:
    """Deterministic bright-ish RGB color from an integer ID."""
    x = (obj_id * 2654435761) & 0xFFFFFFFF
    r = 64 + (x & 0xFF) // 2
    g = 64 + ((x >> 8) & 0xFF) // 2
    b = 64 + ((x >> 16) & 0xFF) // 2
    return int(r), int(g), int(b)


def overlay_instances(
    frame: Image.Image,
    instances: List[Dict[str, Any]],
    alpha: float = 0.45,
    mask_bin_threshold: float = 0.5,
) -> Image.Image:
    """
    Draw colored mask overlays + boxes + "ID:score" labels.
    """
    img = frame.convert("RGB")
    arr = np.array(img).astype(np.float32)

    # Apply mask overlays
    for inst in instances:
        m = inst["mask"]
        if m.dtype != np.bool_:
            m = m > mask_bin_threshold
        m = np.asarray(m, dtype=bool)

        color = np.array(_color_from_id(inst["id"]), dtype=np.float32)
        arr[m] = arr[m] * (1.0 - alpha) + color * alpha

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)

    # Draw boxes + labels
    for inst in instances:
        x1, y1, x2, y2 = inst["box_xyxy"].tolist()
        color = _color_from_id(inst["id"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"ID {inst['id']}  {inst['score']:.2f}"
        draw.text((x1 + 3, y1 + 3), label, fill=color)

    return out


def save_overlay_video(
    video_frames: List[Image.Image],
    per_frame_instances: Dict[int, List[Dict[str, Any]]],
    save_path: str,
    fps: float,
) -> None:
    """
    Save an MP4 overlay video. Uses imageio (ffmpeg backend).
    """
    import imageio.v2 as imageio  # keep import local

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        save_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,  # avoids 1088 padding surprises on some resolutions
    )

    try:
        for i, frame in enumerate(video_frames):
            vis = overlay_instances(frame, per_frame_instances.get(i, []))
            writer.append_data(np.asarray(vis))
    finally:
        writer.close()


def save_sam3_per_frame_folders(
    video_frames: List[Image.Image],
    per_frame_instances: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    mask_bin_threshold: float = 0.5,
    close_kernel: int = 10,
    bridge_distance: int = 15,
    joint_mask_post_proc: bool = False,
) -> None:
    """
    Save per-frame results with Secret Panda post-processing on each mask.

    For each frame, saves to output_dir/frame_{k}/:
        - sam3Mask_id_all.png                         (HxW, binary union of all post-processed masks, 0/255)
        - sam3Mask_id_{id_index}.png                  (HxW, post-processed binary mask per object, 0/255)
        - cropped_bbox_x{x}_y{y}_w{w}_h{h}_m{id_index}_input.png   (cropped RGB pixels around the mask)
        - cropped_bbox_x{x}_y{y}_w{w}_h{h}_m{id_index}_sam3mask.png (cropped post-processed SAM3 mask)

    Post-processing per mask (via secret_panda):
        1. Fill internal holes
        2. Morphological close (kernel=close_kernel)
        3. Bridge gaps between regions (distance=bridge_distance)
        4. Final hole fill

    If joint_mask_post_proc=True, also apply secret_panda on the union mask (sam3Mask_id_all.png)
    after combining per-object masks. This fills inter-object gaps (e.g., slivers between adjacent people).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    H = video_frames[0].height
    W = video_frames[0].width

    # Per-frame worker: post-process masks, save PNGs, return mask updates
    def _process_single_frame(frame_idx):
        frame_dir = output_dir / f"frame_{frame_idx}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        instances = per_frame_instances.get(frame_idx, [])
        if len(instances) == 0:
            # Folder exists but is empty
            return []

        # Deterministic per-frame "i-th object" ordering
        instances_sorted = sorted(instances, key=lambda d: int(d["id"]))

        # Union-of-masks (binary) for sam3Mask_id_all.png
        union_mask = np.zeros((H, W), dtype=bool)

        frame_rgb = video_frames[frame_idx].convert("RGB")
        frame_np = np.asarray(frame_rgb)  # HxWx3 uint8

        mask_updates = []  # list of (id_index, processed_mask) for main thread

        for id_index, inst in enumerate(instances_sorted):
            m = inst["mask"]
            if m.dtype != np.bool_:
                m = m > mask_bin_threshold
            m = np.asarray(m, dtype=bool)

            # Safety: ensure mask matches video resolution
            if m.shape != (H, W):
                raise ValueError(f"Mask shape {m.shape} != video shape {(H, W)} at frame {frame_idx}")

            # --- Post-process mask with Secret Panda ---
            m = secret_panda(
                m.astype(np.uint8),
                fill_holes_first=True,
                close_kernel=close_kernel,
                bridge_distance=bridge_distance,
            ).astype(bool)

            # Verify dimensions after post-processing
            if m.shape != (H, W):
                raise ValueError(f"secret_panda changed mask shape from {(H, W)} to {m.shape} at frame {frame_idx}")

            # Collect mask update for main thread
            mask_updates.append((id_index, m))

            # Update union mask
            union_mask |= m

            # Save per-object binary mask
            bin_mask_u8 = (m.astype(np.uint8) * 255)
            Image.fromarray(bin_mask_u8, mode="L").save(frame_dir / f"sam3Mask_id_{id_index}.png")

            # Crop bbox from mask pixels
            ys, xs = np.where(m)
            if len(xs) == 0 or len(ys) == 0:
                # Mask empty after post-processing: skip crop (still saved binary mask above)
                continue

            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max()) + 1  # exclusive
            y2 = int(ys.max()) + 1  # exclusive
            w = int(x2 - x1)
            h = int(y2 - y1)

            crop = frame_np[y1:y2, x1:x2, :]  # RGB crop (original pixels)
            crop_img = Image.fromarray(crop, mode="RGB")
            crop_name = f"cropped_bbox_x{x1}_y{y1}_w{w}_h{h}_m{id_index}_input.png"
            crop_img.save(frame_dir / crop_name)

            # Save the post-processed SAM3 mask cropped to the same bbox
            mask_crop_u8 = (m[y1:y2, x1:x2].astype(np.uint8) * 255)  # (h, w), 0/255
            mask_crop_name = f"cropped_bbox_x{x1}_y{y1}_w{w}_h{h}_m{id_index}_sam3mask.png"
            Image.fromarray(mask_crop_u8, mode="L").save(frame_dir / mask_crop_name)

        # Optionally fill inter-object gaps in the union mask (slivers between adjacent people)
        if joint_mask_post_proc:
            union_mask = secret_panda(
                union_mask.astype(np.uint8),
                fill_holes_first=True,
                close_kernel=close_kernel,
                bridge_distance=bridge_distance,
            ).astype(bool)

        # Save union mask as 0/255
        union_u8 = (union_mask.astype(np.uint8) * 255)
        Image.fromarray(union_u8, mode="L").save(frame_dir / "sam3Mask_id_all.png")

        return mask_updates

    # Process all frames in parallel, apply mask mutations in main thread
    with ThreadPoolExecutor() as pool:
        futures = {
            pool.submit(_process_single_frame, frame_idx): frame_idx
            for frame_idx in range(len(video_frames))
        }
        for future in tqdm(as_completed(futures), total=len(video_frames), desc="Processing & saving frames"):
            frame_idx = futures[future]
            mask_updates = future.result()  # re-raises any worker exception
            # Apply inst["mask"] = m mutations back to per_frame_instances
            instances = per_frame_instances.get(frame_idx, [])
            instances_sorted = sorted(instances, key=lambda d: int(d["id"]))
            for id_index, m in mask_updates:
                instances_sorted[id_index]["mask"] = m

def argument_parser():
    parser = argparse.ArgumentParser(description="Run SAM3 on a single video with a text prompt, with Secret Panda mask post-processing.")
    parser.add_argument("--video_path", type=str, required=True, help="Path or URL to the input video.")
    parser.add_argument("--sam_prompt", type=str, default="human", help="Text prompt for segmentation (default: 'human').")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the output results. Default: sam3_outputs/{video_name}_postproc/")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process (None = all).")
    parser.add_argument("--close_kernel", type=int, default=10, help="Kernel size for morphological close in Secret Panda (default: 10).")
    parser.add_argument("--bridge_distance", type=int, default=15, help="Distance for bridging gaps between mask regions in Secret Panda (default: 15).")
    parser.add_argument("--joint_mask_post_proc", action="store_true", default=False,
        help="If set, also apply Secret Panda post-processing on the joint/union mask (sam3Mask_id_all.png) after per-object masks are combined.")
    parser.add_argument("--model_path", type=str, default="./models/sam3", help="Local path or HF repo id for SAM3 model.")
    return parser.parse_args()

if __name__ == "__main__":

    # =============================================================================
    # User inputs
    # =============================================================================
    args = argument_parser()
    video_path = args.video_path
    prompt = args.sam_prompt
    output_dir = args.output_dir
    max_frames = args.max_frames

    # If output_dir not specified, create default based on video name
    if output_dir is None:
        video_stem = Path(video_path).stem if "://" not in video_path else "video"
        output_dir = f"sam3_outputs/{video_stem}_postproc"

    # =============================================================================
    # Load model
    # =============================================================================
    model, processor, device = init_sam3_video(model_path=args.model_path, dtype=torch.bfloat16)
    print(f"Finished loading SAM3 Video model on device: {device}")

    # =============================================================================
    # Run SAM3 Inference
    # =============================================================================
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    video_frames, fps = load_video_frames(video_path=video_path, max_frames=max_frames)

    outputs_per_frame = run_sam3_video(
        model=model,
        processor=processor,
        device=device,
        video_frames=video_frames,
        prompt=prompt,
        dtype=torch.bfloat16,
    )

    # =============================================================================
    # Save SAM3 Result with post-processing
    # =============================================================================
    per_frame_instances = pack_per_frame_instances(outputs_per_frame)

    # Save per-frame folders FIRST (modifies masks in place with Secret Panda)
    frames_out_dir = str(Path(output_dir))
    save_sam3_per_frame_folders(
        video_frames=video_frames,
        per_frame_instances=per_frame_instances,
        output_dir=frames_out_dir,
        mask_bin_threshold=0.5,
        close_kernel=args.close_kernel,
        bridge_distance=args.bridge_distance,
        joint_mask_post_proc=args.joint_mask_post_proc,
    )

    # Save visualized overlay video (uses post-processed masks from save_sam3_per_frame_folders)
    stem = Path(video_path).stem if "://" not in video_path else "video"
    overlay_path = str(Path(output_dir) / f"{stem}__prompt_{prompt.replace(' ', '_')}__sam3_overlay.mp4")
    save_overlay_video(video_frames, per_frame_instances, save_path=overlay_path, fps=fps)

    print(f"Finished SAM3 processing with Secret Panda post-processing. Results saved under: {output_dir}")
