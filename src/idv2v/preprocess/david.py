"""
Frame-wise DAViD surface-normal prediction + canvas composition.

Per frame, runs DAViD on each SAM3 cropped bbox (input: cropped_bbox_..._input.png)
and writes a cropped normal map next to it. Then composites all per-frame normals
onto a gray (127) canvas the same size as the source video, using the SAM3 mask
(not DAViD's own foreground) to decide which pixels are foreground. Output is
written to <preprocessing_folder>/david_normal.mp4.

Frames with no SAM3 detection (no frame_<k>/ folder) become fully gray.
"""
# =========================
# Imports & environment
# =========================
import os, glob, re
import sys
import re, glob, csv
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import imageio
from tqdm import tqdm
import random
from pathlib import Path
import argparse

# DAViD runtime is vendored under david_runtime/ as a sibling package
from .david_runtime.multi_task_estimator import MultiTaskEstimator

### Local file writer (simple passthrough)
from contextlib import contextmanager
@contextmanager
def temp_file_writer(file_path: str, is_folder: bool = False, not_s3: bool = False):
    """Yields the file_path directly for local writes."""
    yield file_path


##### David prediction functions, for david_predict_image() #####

# this is for finding normal for the final composition on grey canvas
_CROP_RE = re.compile(
    r"cropped_bbox_x(?P<x>-?\d+)_y(?P<y>-?\d+)_w(?P<w>\d+)_h(?P<h>\d+)_m(?P<m>\d+)_DavidNormal\.png$"
)

def get_normal_output(
    normals: np.ndarray,
    mask: Optional[np.ndarray] = None,
    background_color: Tuple[int, int, int] = (127, 127, 127),
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize surface normals and (optionally) blend with a solid background color.
    Applies a hard then a soft mask:
      - hard: zero out where mask == 0
      - soft: out = normals_vis * mask + bg * (1 - mask)
    """
    if normals.ndim != 3 or normals.shape[2] != 3:
        raise ValueError("`normals` must be (H, W, 3).")

    H, W, _ = normals.shape
    vis_normals = ((normals / 2.0 + 0.5) * 255.0)
    vis_normals = vis_normals[:, :, ::-1].astype(np.uint8)  # RGB->BGR

    if mask is None:
        if save_path:
            cv2.imwrite(save_path, vis_normals)
        print("Warning: no mask provided; returning full normal map.")
        return vis_normals

    if mask.shape[:2] != (H, W):
        raise ValueError("The dimensions of 'normals' and 'mask' must match in H,W.")

    vis_normals[mask == 0] = 0  # hard step

    mask_f = np.expand_dims(np.clip(mask, 0, 1).astype(np.float32), -1)  # (H,W,1)
    bg = np.full((H, W, 3), background_color, dtype=np.float32)
    vis = (vis_normals.astype(np.float32) * mask_f + bg * (1.0 - mask_f)).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, vis)
    return vis


def get_foreground_output(
    mask: np.ndarray,
    save_path: Optional[str] = None,
    foreground_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Soft-mask composite for foreground:
      out = FG_color * mask + BG_color * (1 - mask)
    """
    mask = np.expand_dims(mask, -1)                     # (H,W,1)
    mask = np.clip(mask, 0, 1).astype(np.float32)
    fg = np.full((*mask.shape[:2], 3), foreground_color, dtype=np.float32)
    bg = np.full((*mask.shape[:2], 3), background_color, dtype=np.float32)
    vis_mask = (fg * mask + bg * (1.0 - mask)).astype(np.uint8)
    if save_path:
        cv2.imwrite(save_path, vis_mask)
    return vis_mask

def david_predict_image(input_image_path, output_normal_path, output_foreground_path, multitask_estimator, s3 = False, mask_image_path = None):
    '''
    Run DAViD on a single crop and save both the normal map and the foreground
    mask to the given paths. The normal map shows surface normals on the
    foreground area and gray (127) on the background.

    When s3=True, outputs are written via temp_file_writer (S3-aware); otherwise
    they are written directly to the local paths.

    mask_image_path selects which mask defines the foreground of the normal map:
      - if given, that mask image is loaded and used.
      - if None, a ValueError is raised. The SAM3 mask must be supplied because
        DAViD's own predicted foreground may include extra objects we don't want.
    '''
    input_image = cv2.imread(input_image_path)
    result = multitask_estimator.estimate_all_tasks(input_image)

    if mask_image_path is None:
        mask = result["foreground"]
        raise ValueError("mask_image_path is None, but we should use sam3 mask instead of DAViD mask because the latter might have more objects which is not desired.")
    else:
        # Read as grayscale (H, W), values in [0,255]
        mask_u8 = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise FileNotFoundError(f"Failed to read mask image: {mask_image_path}")
        # Convert to float mask in [0,1] with shape (H, W)
        # (white=1, black=0; also works if the PNG is anti-aliased / soft)
        mask = (mask_u8.astype(np.float32) / 255.0)
        

    if not s3:
        get_normal_output(
            normals=result["normal"],
            mask=mask,
            background_color=(127, 127, 127),
            save_path=output_normal_path,
        )
        get_foreground_output(
            mask=result["foreground"],
            foreground_color=(255, 255, 255),
            background_color=(0, 0, 0),
            save_path=output_foreground_path,
        )
    else:
        with temp_file_writer(output_normal_path) as temp_normal_path:
            get_normal_output(
                normals=result["normal"],
                mask=mask,
                background_color=(127, 127, 127),
                save_path=temp_normal_path,
            )
        with temp_file_writer(output_foreground_path) as temp_foreground_path:
            get_foreground_output(
                mask=result["foreground"],
                foreground_color=(255, 255, 255),
                background_color=(0, 0, 0),
                save_path=temp_foreground_path,
            )
##### David prediction functions ends #####

##### Canvas Composition starts #####
def _parse_crop_path(p: Path):
    """Return (x, y, w, h, m) from filename or None if pattern doesn't match."""
    m = _CROP_RE.search(p.name)
    if not m:
        return None
    x = int(m.group("x")); y = int(m.group("y"))
    w = int(m.group("w")); h = int(m.group("h"))
    mid = int(m.group("m"))
    return x, y, w, h, mid

def _video_meta(video_path: str):
    """Get (W, H, N, fps) from a reference video, preferring container metadata and
    only decoding frames as a fallback when size / frame-count are unavailable."""
    r = imageio.get_reader(video_path)
    try:
        meta = r.get_meta_data()
        fps = float(meta.get("fps", 30.0)) or 30.0
        if "size" in meta:
            W, H = meta["size"]
        else:
            # Fallback from first frame if needed
            f0 = r.get_next_data()
            H, W = f0.shape[:2]
        # Determine number of frames robustly
        N = meta.get("nframes", None)
        if not isinstance(N, int) or N <= 0 or N == float("inf"):
            try:
                N = r.count_frames()
            except Exception:
                # Last resort: iterate (avoids incorrect inf from some backends)
                N = sum(1 for _ in r)
        return W, H, int(N), fps
    finally:
        r.close()


### Old function: result is loss of information when two crops overlap
# def compose_full_canvas(allFrame_folder: str, canvas_reference_video_path: str, save_path: str) -> None:
#     """
#     Reconstruct a full-size video by pasting framewise face crops back onto a gray canvas.

#     Rules:
#     - Canvas resolution & frame count match the reference video exactly.
#     - Crops come from: allFrame_folder/frame_{index}/cropped_bbox_x{..}_y{..}_w{..}_h{..}_m{mask_id}_DavidNormal.png
#     - On overlap within the same frame, keep pixels from the smaller mask_id (i.e., later crops never overwrite earlier ones).
#     - Saves with imageio while preserving resolution exactly.
#     """
#     allFrame_folder = Path(allFrame_folder)
#     save_path = Path(save_path)
#     save_path.parent.mkdir(parents=True, exist_ok=True)

#     W, H, N, fps = _video_meta(canvas_reference_video_path)

#     # Prepare writer; macro_block_size=None avoids encoder resizing/padding
#     writer = imageio.get_writer(str(save_path), fps=fps, macro_block_size=None)
#     try:
#         for idx in range(N):
#             # 1) start with gray canvas
#             canvas = np.full((H, W, 3), 127, dtype=np.uint8)
#             filled = np.zeros((H, W), dtype=bool)  # tracks pixels already filled (for overlap rule)

#             # 2) collect crops for this frame (if any), sorted by mask_id ascending
#             frame_dir = allFrame_folder / f"frame_{idx}"
#             if frame_dir.is_dir():
#                 crop_infos = []
#                 for p in frame_dir.glob("*.png"):
#                     parsed = _parse_crop_path(p)
#                     if parsed:
#                         crop_infos.append((p, *parsed))  # (path, x,y,w,h,m)
#                 crop_infos.sort(key=lambda t: t[-1])  # by mask_id

#                 # 3) paste crops in ascending mask_id; later (larger id) never overwrite filled pixels
#                 for p, x, y, w, h, _mid in crop_infos:
#                     try:
#                         crop = imageio.imread(p)
#                     except Exception:
#                         continue  # skip unreadable files

#                     # Ensure crop is HxWx3 uint8
#                     if crop.ndim == 2:
#                         crop = np.stack([crop]*3, axis=-1)
#                     elif crop.ndim == 3 and crop.shape[2] == 4:
#                         # If RGBA, drop alpha (assume fully opaque face crop)
#                         crop = crop[:, :, :3]
#                     crop = crop.astype(np.uint8, copy=False)

#                     # Clip bbox to canvas
#                     x0 = max(0, x); y0 = max(0, y)
#                     x1 = min(W, x + w); y1 = min(H, y + h)
#                     if x1 <= x0 or y1 <= y0:
#                         continue  # fully out of bounds

#                     # Align crop to clipped region
#                     cw = x1 - x0; ch = y1 - y0
#                     # If the on-disk crop dims don't match (rare), center-crop/pad to match target region
#                     ch0, cw0 = crop.shape[:2]
#                     if (ch0, cw0) != (ch, cw):
#                         # quick safe fallback: resize with a basic nearest interpolation
#                         # (keeps code short; replace with cv2 if you need higher quality)
#                         crop = np.array(
#                             imageio.imresize(crop, (ch, cw), "nearest"), dtype=np.uint8
#                         )

#                     # Composite without overwriting already-filled pixels (enforces smaller mask_id precedence)
#                     target_region = canvas[y0:y1, x0:x1]
#                     already = filled[y0:y1, x0:x1]
#                     # mask of where we are allowed to write (not yet filled)
#                     write_mask = ~already
#                     if write_mask.any():
#                         target_region[write_mask] = crop[write_mask]
#                         filled[y0:y1, x0:x1][write_mask] = True

#             # 4) append the composed frame
#             writer.append_data(canvas)
#     finally:
#         writer.close()

def compose_full_canvas(allFrame_folder: str, canvas_reference_video_path: str, save_path: str) -> None:
    """
    Reconstruct a full-size video by pasting framewise face-normal crops back onto a gray canvas.

    Behavior:
    - For each normal crop `cropped_bbox_x..._DavidNormal.png`, load the corresponding
      SAM3 mask `cropped_bbox_x..._sam3mask.png`.
    - Paste only pixels where the SAM3 mask is foreground (white), i.e., mask >= 128.
    - Use a per-pixel `filled` mask so that once a pixel is written by a smaller
      mask_id crop, larger mask_id crops cannot overwrite it.

    Strictness:
    - If the normal crop cannot be read, an exception is raised.
    - If the corresponding SAM3 mask file is missing or cannot be read, an exception is raised.
    - If the crop / mask sizes cannot be consistently aligned with the clipped
      bbox region, an exception is raised (no resizing fallbacks).
    """

    allFrame_folder = Path(allFrame_folder)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    W, H, N, fps = _video_meta(canvas_reference_video_path)

    # Prepare writer; macro_block_size=None avoids encoder resizing/padding
    writer = imageio.get_writer(str(save_path), fps=fps, macro_block_size=None)
    try:
        for idx in range(N):
            # 1) Start with gray canvas and a "filled" mask (tracks pixels already set)
            canvas = np.full((H, W, 3), 127, dtype=np.uint8)
            filled = np.zeros((H, W), dtype=bool)

            # 2) Collect normal crops for this frame, sorted by mask_id ascending
            frame_dir = allFrame_folder / f"frame_{idx}"
            if frame_dir.is_dir():
                crop_infos = []
                for p in frame_dir.glob("*.png"):
                    parsed = _parse_crop_path(p)
                    if parsed:
                        crop_infos.append((p, *parsed))  # (path, x, y, w, h, m)
                crop_infos.sort(key=lambda t: t[-1])  # by mask_id

                # 3) Paste crops with SAM3 mask-based compositing
                for p, x, y, w, h, _mid in crop_infos:
                    # Corresponding SAM3 mask path for this crop
                    mask_path = p.with_name(p.name.replace("_DavidNormal.png", "_sam3mask.png"))
                    if not mask_path.exists():
                        raise FileNotFoundError(f"SAM3 mask not found for crop: {mask_path}")

                    # Read normal crop (let exceptions propagate if any issue)
                    crop = imageio.imread(p)
                    if crop is None:
                        raise RuntimeError(f"Failed to read normal crop image: {p}")

                    # Ensure crop is HxWx3 uint8
                    if crop.ndim == 2:
                        crop = np.stack([crop] * 3, axis=-1)
                    elif crop.ndim == 3 and crop.shape[2] == 4:
                        # If RGBA, drop alpha (assume fully opaque)
                        crop = crop[:, :, :3]
                    crop = crop.astype(np.uint8, copy=False)

                    # Read SAM3 mask as grayscale
                    mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_u8 is None:
                        raise RuntimeError(f"Failed to read SAM3 mask image: {mask_path}")

                    if mask_u8.shape[:2] != crop.shape[:2]:
                        raise ValueError(
                            f"Mask and crop size mismatch for {p} and {mask_path}: "
                            f"crop={crop.shape[:2]}, mask={mask_u8.shape[:2]}"
                        )

                    # Clip bbox to canvas
                    x0 = max(0, x)
                    y0 = max(0, y)
                    x1 = min(W, x + w)
                    y1 = min(H, y + h)

                    # If fully out of bounds, nothing to paste; this is not treated as an error
                    if x1 <= x0 or y1 <= y0:
                        continue

                    # Align crop/mask to clipped region by slicing, not resizing.
                    # Original crop covers [x:x+w, y:y+h] on the canvas.
                    cw = x1 - x0
                    ch = y1 - y0
                    ch0, cw0 = crop.shape[:2]

                    # Compute corresponding region in the crop/mask
                    crop_x0 = x0 - x  # offset inside crop
                    crop_y0 = y0 - y
                    crop_x1 = crop_x0 + cw
                    crop_y1 = crop_y0 + ch

                    # Strict bounds check: must fit inside crop/mask
                    if not (0 <= crop_x0 < cw0 and 0 < crop_x1 <= cw0 and
                            0 <= crop_y0 < ch0 and 0 < crop_y1 <= ch0):
                        raise ValueError(
                            f"Clipped bbox for crop {p} does not fit inside crop image:\n"
                            f"  canvas bbox (clipped): (x0={x0}, y0={y0}, x1={x1}, y1={y1})\n"
                            f"  crop bbox: (x={x}, y={y}, w={w}, h={h}), crop size={crop.shape[:2]}"
                        )

                    crop_region = crop[crop_y0:crop_y1, crop_x0:crop_x1]
                    mask_region = mask_u8[crop_y0:crop_y1, crop_x0:crop_x1]

                    if crop_region.shape[:2] != (ch, cw) or mask_region.shape != (ch, cw):
                        raise ValueError(
                            f"Internal region size mismatch for crop {p}: "
                            f"crop_region={crop_region.shape[:2]}, mask_region={mask_region.shape}, "
                            f"expected={(ch, cw)}"
                        )

                    target_region = canvas[y0:y1, x0:x1]
                    filled_region = filled[y0:y1, x0:x1]

                    # Build boolean mask of foreground pixels from SAM3 (white = inside)
                    # Threshold at 128 to be robust to minor variations.
                    mask_fg = (mask_region >= 128)

                    # Only write where:
                    #   - SAM3 mask says foreground, AND
                    #   - pixel has not been filled yet (preserve smaller mask_id precedence)
                    write_mask = mask_fg & (~filled_region)

                    if write_mask.any():
                        target_region[write_mask] = crop_region[write_mask]
                        filled_region[write_mask] = True

            # 4) Append the composed frame
            writer.append_data(canvas)
    finally:
        writer.close()



def _to_rgb_frame(arr: np.ndarray) -> np.ndarray:
    """Ensure frame is HxWx3 uint8 (drop alpha / expand gray)."""
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:  # RGBA -> RGB
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.uint8, copy=False)

def _normalize_bg_color(color) -> np.ndarray:
    """Accept int (0-255) or RGB tuple/list; return (3,) uint8."""
    if isinstance(color, (int, np.integer)):
        c = [int(color)] * 3
    else:
        assert len(color) == 3, "background_color must be an int or RGB tuple/list of length 3"
        c = [int(v) for v in color]
    return np.array(c, dtype=np.uint8)


# def compose_make(video_path: str,
#                  mask_video_path: str,
#                  save_path: str,
#                  background_color=127) -> None:
#     """
#     Create a new video where, for each frame, pixels inside the (white) mask
#     come from `video_path` and pixels outside are set to `background_color`.

#     - Asserts same resolution and number of frames for `video_path` and `mask_video_path`.
#     - Saves with the exact same resolution & frame count using imageio.
#     """
#     # 1) Validate meta
#     W1, H1, N1, fps = _video_meta(video_path)
#     W2, H2, N2, _ = _video_meta(mask_video_path)
#     assert (W1, H1, N1) == (W2, H2, N2), \
#         f"Video/mask mismatch: video={(W1,H1,N1)} vs mask={(W2,H2,N2)}"

#     # 2) Prepare I/O
#     vid_reader = imageio.get_reader(video_path)
#     msk_reader = imageio.get_reader(mask_video_path)
#     writer = imageio.get_writer(str(save_path), fps=fps, macro_block_size=None)

#     bg = _normalize_bg_color(background_color)
#     try:
#         # Iterate sequentially (keeps memory low and preserves order)
#         for i, (vf, mf) in enumerate(zip(vid_reader, msk_reader)):
#             vf = _to_rgb_frame(vf)
#             mf = _to_rgb_frame(mf)

#             # 3) Build mask: "white" means inside; threshold (>= 128 on any channel)
#             # Using max across channels is tolerant to slightly off-white masks.
#             mask_bool = (mf.max(axis=2) >= 128)

#             # 4) Compose
#             out = np.empty_like(vf)
#             # Set background first, then overwrite masked-in region with video pixels
#             out[:] = bg  # broadcast (H, W, 3)
#             out[mask_bool] = vf[mask_bool]

#             writer.append_data(out)
#     finally:
#         writer.close()
#         vid_reader.close()
#         msk_reader.close()

##### Canvas Composition ends #####

def parse_args():
    parser = argparse.ArgumentParser(description="Run DAViD surface-normal prediction on SAM3 crops and composite them onto a gray canvas video.")
    parser.add_argument("--DAViD_ckpt", type=str, default="./models/DAViD/multi-task-model-vitl16_384.onnx", help="Path to the DAViD ONNX model checkpoint.")
    parser.add_argument("--preprocessing_folder", type=str, help="Folder produced by sam3.py; holds frame_<k>/ subfolders with cropped_bbox_..._input.png crops (e.g. frame_0/cropped_bbox_x1882_y31_w314_h534_m0_input.png).", required=True)
    parser.add_argument("--canvas_reference_video_path", type=str, help="Reference video whose resolution and frame count define the output canvas.", required=True)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    preprocessing_folder = args.preprocessing_folder 
    canvas_reference_video_path = args.canvas_reference_video_path

    ### load model 
    multitask_model_path = args.DAViD_ckpt
    assert os.path.exists(multitask_model_path), f"DAViD model checkpoint not found at {multitask_model_path}"
    multitask_estimator = MultiTaskEstimator(
            onnx_model=multitask_model_path, is_inverse_depth=False
        )
    
    use_s3 = "mnt-s3" in preprocessing_folder.lower()
    # path of the canvas to be saved (the final goal of this script)
    canvas_save_path = os.path.join(preprocessing_folder, "david_normal.mp4")

    # if os.path.exists(canvas_save_path):
    #     print(f"Already processed: Canvas with frame-wise DAViD prediction already exists for {preprocessing_folder}, skipping...")
    #     exit(0)

    ### Gathering all input images for per-frame prediction, and output paths
    input_image_path_list = []
    output_normal_path_list = []
    output_foreground_path_list = []  # written but unused downstream; we composite using sam3 mask
    sam3_mask_path_list = []

    # regex to match cropped bbox pattern
    pattern = re.compile(r"cropped_bbox_x\d+_y\d+_w\d+_h\d+_m\d+_input\.png$")
    for root, dirs, files in os.walk(preprocessing_folder):
        # only go into subfolders named frame_*
        if os.path.basename(root).startswith("frame_"):
            for f in files:
                if pattern.match(f):
                    input_image_path_list.append(os.path.abspath(os.path.join(root, f)))

    # print(f"Found {len(input_image_path_list)} input images")

    for input_image_path in input_image_path_list:
        output_normal_path = input_image_path.replace("_input.png", "_DavidNormal.png")
        output_foreground_path = input_image_path.replace("_input.png", "_DavidForeground.png")
        output_normal_path_list.append(output_normal_path)
        output_foreground_path_list.append(output_foreground_path)

        sam3_mask_path_list.append(input_image_path.replace("_input.png", "_sam3mask.png"))


    ### Per-frame prediction loop
    for idx, input_image_path in enumerate(tqdm(input_image_path_list, desc="DAViD prediction")):
        output_normal_path = output_normal_path_list[idx]
        output_foreground_path = output_foreground_path_list[idx]
        # print(f"Processing {idx+1}/{len(input_image_path_list)}: {input_image_path}")
        # if os.path.exists(output_normal_path) and os.path.exists(output_foreground_path):
        #     continue
        david_predict_image(input_image_path, output_normal_path, output_foreground_path, multitask_estimator, s3 = use_s3, mask_image_path = sam3_mask_path_list[idx])

    ### Canvas Composition
    if not use_s3:
        ## compose the normal canvas using the SAM3 masks
        compose_full_canvas(allFrame_folder = preprocessing_folder, canvas_reference_video_path = canvas_reference_video_path, save_path = canvas_save_path)
        # ## then get canvas with normal using tight face mask
        # compose_make(video_path = canvas_save_path_expandedMask, mask_video_path = used_mask_video_path, save_path = canvas_save_path_tightMask, background_color = 127)
    else:
        with temp_file_writer(canvas_save_path) as temp_canvas_save_path:
            compose_full_canvas(allFrame_folder = preprocessing_folder, canvas_reference_video_path = canvas_reference_video_path, save_path = temp_canvas_save_path)
        # with temp_file_writer(canvas_save_path_tightMask) as temp_canvas_save_path_tightMask:
        #     compose_make(video_path = canvas_save_path_expandedMask, mask_video_path = used_mask_video_path, save_path = temp_canvas_save_path_tightMask, background_color = 127)
    
    print(f"Finished DAViD prediction and canvas composition for video folder: {preprocessing_folder}")