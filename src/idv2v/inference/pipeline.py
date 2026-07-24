"""
clip-by-clip I2V + VACE multi-control inference (Wan 2.1).

Loads I2V-14B (DiT + T5 + VAE + CLIP) + VACE-14B ControlNet, optionally overrides DiT+VACE
with a finetuned checkpoint, then generates video clip-by-clip:
  - Conditions:  one or more VACE control videos (idv2v uses one: foreground-on-gray pixels).
  - I2V anchor:  --input_image  (and optional --key_frame_paths/--key_frame_indices to anchor identity over long clips).
  - Anti-drift (SVI) pad:     --ref_pad_num -1 fills unused I2V frames with the stylized first frame.
  - Multi-clip:  auto-scheduled from condition length; clips chain via previous clip's splice frame.

Outputs land in --result_save_folder: generated_video.mp4, per-clip mp4s, conditions, viz.
USP sequence parallel auto-enabled when launched under torchrun with --use_usp.

Don't call this directly — use the inference scripts under scripts/, which build the right torchrun command.
"""

import os

# --- transformers 5.x compat shims (must run before diffusers/xfuser/diffsynth imports) ---
# transformers 5.x dropped slow tokenizers; diffusers' HunyuanDiT (pulled by xfuser at import)
# still does `from transformers import MT5Tokenizer`. Patch the LazyModule __getattr__ to
# alias MT5Tokenizer -> T5Tokenizer so the import succeeds.
import transformers as _tx
_orig_tx_getattr = type(_tx).__getattr__
def _patched_tx_getattr(self, name):
    if name == "MT5Tokenizer":
        return self.T5Tokenizer
    return _orig_tx_getattr(self, name)
type(_tx).__getattr__ = _patched_tx_getattr
# --- end shims ---

import time
import torch
import numpy as np
import imageio
from os import path
from typing import List
from PIL import Image
from tqdm import tqdm
import argparse
import torch.distributed as dist

import subprocess
import shutil
import sys

def ensure_ffmpeg_installed():
    """Check if ffmpeg/ffprobe are available, install via apt if missing.

    All progress messages (and apt's own output) go to STDERR, not stdout, on
    purpose: scripts/infer.sh resolves this module's path with
    `PIPELINE="$(python -c 'import idv2v.inference.pipeline as m; print(m.__file__)')"`,
    and that command substitution captures stdout. If the first-run install
    messages went to stdout they would contaminate $PIPELINE and torchrun would
    get a garbage file path. Running this at import time (before torchrun) also
    means ffmpeg is installed once, single-process, so the parallel ranks never
    race on `apt`.
    """
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        print("ffmpeg/ffprobe not found. Installing via apt...", file=sys.stderr)
        subprocess.run(["apt", "install", "-y", "ffmpeg"], check=True, stdout=sys.stderr)
        print("ffmpeg installed successfully.", file=sys.stderr)

ensure_ffmpeg_installed()

# `diffsynth` package is installed editable from ./diffsynth_studio (see pyproject.toml)
from diffsynth.pipelines.wan_video_new_multiVace_svi import WanVideoPipeline, ModelConfig, load_vace_from_checkpoint

# Default Wan 2.1 negative prompt (Chinese quality-degradation terms). Override via --negative_prompt.
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
from diffsynth import VideoData, load_state_dict
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_vace import VaceWanModel
from diffsynth.models.utils import init_weights_on_device


### Helper functions: video I/O, frame loading, clip scheduling

def save_video(output_path, video, fps, quality=None, imageio_params=None):
    imageio_params = imageio_params if imageio_params is not None else {}
    if quality is not None:
        imageio_params["quality"] = quality

    writer = imageio.get_writer(output_path, fps=fps, **imageio_params)

    for i in range(video.shape[0]):
        writer.append_data(video[i])
    writer.close()

def save_video_fn(video, output_path, quality=8, save_gif = True, fps=16):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_video(output_path, video, fps=fps, quality=quality)
    if save_gif:
        gif_output_path = f"{path.splitext(output_path)[0]}.gif"
        save_video(gif_output_path, video, fps=fps, quality=quality, imageio_params={"loop": 0})


def _infer_fps_or_default(video_path: str, default_fps: int = 24) -> int:
    try:
        r = imageio.get_reader(video_path)
        meta = r.get_meta_data()
        fps = int(round(meta.get("fps", default_fps)))
        r.close()
        return fps if fps > 0 else default_fps
    except Exception:
        return default_fps


def get_source_fps(video_path: str, default_fps: float = 16.0) -> float:
    """Return a video's frame rate as a float, read via ffprobe's exact `r_frame_rate`
    rational (e.g. "30000/1001" -> 29.97). Falls back to default_fps on any error.
    Used to encode outputs at the SOURCE video's fps when --output_fps=source."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=nokey=1:noprint_wrappers=1", video_path],
            capture_output=True, text=True,
        ).stdout.strip()
        num, den = (out.split("/") + ["1"])[:2]
        fps = float(num) / float(den)
        return fps if fps > 0 else default_fps
    except Exception:
        return default_fps

def save_frames_to_video(frames: List[Image.Image], save_path: str, fps: int):
    """Save a list of PIL RGB frames to a lossless mp4 via ffmpeg pipe."""
    if len(frames) == 0:
        raise ValueError("No frames to save.")

    w, h = frames[0].size

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "slow",
        "-crf", "0",
        "-pix_fmt", "yuv420p",
        save_path
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for img in frames:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        process.stdin.write(arr.tobytes())

    process.stdin.close()
    process.wait()


def stack_videos_left_right_and_save(video_paths: List[str], save_path: str, fps: int = None) -> None:
    """Stack videos horizontally (side-by-side) and save to a single mp4."""
    assert isinstance(video_paths, list) and len(video_paths) >= 2, "Need >=2 videos to stack."
    vds = [VideoData(video_file=p) for p in video_paths]

    lengths = [len(vd) for vd in vds]
    if len(set(lengths)) != 1:
        print("[Warning] Videos have different frame counts: " +
              ", ".join(f"{os.path.basename(p)}={L}" for p, L in zip(video_paths, lengths)) +
              f". Using the minimal frame count: {min(lengths)}.")
    n_frames = min(lengths)

    tol = 0.15
    first_frames = [vd[0].convert("RGB") for vd in vds]
    sizes = [im.size for im in first_frames]
    ars = [w / h for (w, h) in sizes]

    if max(ars) - min(ars) > tol:
        raise ValueError(f"Videos do not share the same aspect ratio (tol={tol}). "
                         f"Got sizes={sizes}, ARs={[round(a,6) for a in ars]}.")

    areas = [w * h for (w, h) in sizes]
    smallest_idx = int(np.argmin(areas))
    target_size = sizes[smallest_idx]

    if any(sz != target_size for sz in sizes):
        print("[Warning] Videos have different sizes but same aspect ratio. "
              f"Sizes={sizes}. Resizing all to the smallest size {target_size}.")

    target_w, target_h = target_size

    if fps is None:
        fps = _infer_fps_or_default(video_paths[0])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    writer = imageio.get_writer(save_path, fps=fps, macro_block_size=None)
    try:
        resized0 = [im if im.size == target_size else im.resize(target_size, Image.BILINEAR)
                    for im in first_frames]
        stacked0 = np.concatenate([np.asarray(im) for im in resized0], axis=1)
        writer.append_data(stacked0)

        for i in range(1, n_frames):
            imgs = [vd[i].convert("RGB") for vd in vds]
            imgs = [im if im.size == (target_w, target_h) else im.resize((target_w, target_h), Image.BILINEAR)
                    for im in imgs]
            stacked = np.concatenate([np.asarray(img) for img in imgs], axis=1)
            writer.append_data(stacked)
    finally:
        writer.close()

def read_video_rgb24_ffmpeg(video_path: str) -> List[Image.Image]:
    """Read video frames using ffmpeg -> rgb24. Returns list of PIL RGB frames."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "default=nokey=1:noprint_wrappers=1", video_path],
        capture_output=True, text=True
    ).stdout.strip().split("\n")

    width = int(probe[0])
    height = int(probe[1])

    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

    frames = []
    frame_size = width * height * 3

    while True:
        raw = pipe.stdout.read(frame_size)
        if len(raw) != frame_size:
            break
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        frames.append(Image.fromarray(arr, "RGB"))

    pipe.stdout.close()
    pipe.wait()
    return frames


def center_crop_and_resize(img: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop and resize a single PIL image to (width, height) using BICUBIC.
    Resizes to match the target aspect ratio first, then center-crops to exact size.
    """
    w, h = img.size
    target_aspect = height / width
    aspect = h / w

    if (h == height) and (w == width):
        return img

    if abs(aspect - target_aspect) < 1e-6:
        return img.resize((width, height), Image.BICUBIC)

    if aspect > target_aspect:  # Too tall -> resize width to target, crop height
        new_w = width
        new_h = int(aspect * new_w)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
    else:  # Too wide -> resize height to target, crop width
        new_h = height
        new_w = int(new_h / aspect)
        resized = img.resize((new_w, new_h), Image.BICUBIC)

    # Center crop to exact target size
    rw, rh = resized.size
    left = (rw - width) // 2
    top = (rh - height) // 2
    return resized.crop((left, top, left + width, top + height))


def load_subsampled_center_cropped_frames(
    video_path: str,
    height: int,
    width: int,
    num_frames: int = None,
) -> List[Image.Image]:
    """
    Load a video, center-crop and resize each frame to (width, height).
    If num_frames is given, truncate or pad (repeat last frame) to exactly that count.
    If num_frames is None, return all frames without truncation/padding.
    """
    assert os.path.exists(video_path), f"Video path does not exist: {video_path}"

    raw_frames = read_video_rgb24_ffmpeg(video_path)

    if num_frames is not None:
        if len(raw_frames) >= num_frames:
            pil_frames = raw_frames[:num_frames]
        else:
            pil_frames = raw_frames + [raw_frames[-1]] * (num_frames - len(raw_frames))
    else:
        pil_frames = raw_frames

    return [center_crop_and_resize(img, width, height) for img in pil_frames]


### Multi-clip schedule and frame slicing

def compute_clip_schedule(total_frames: int, num_frames_per_clip: int) -> List[tuple]:
    """
    Compute (start, end) frame indices for multi-clip generation.

    Regular clips advance by stride = num_frames_per_clip - 1 (1-frame overlap).
    The last clip is anchored at total_frames - num_frames_per_clip so it always
    has exactly num_frames_per_clip frames without needing padding. This may create
    a larger overlap between the last two clips.

    Example: total_frames=222 (e.g. 81+81+60), num_frames_per_clip=81
      stride = 80
      Clip 0: [0,   81)   — 81 frames
      Clip 1: [80,  161)  — 81 frames, 1-frame overlap with clip 0
      Clip 2: [141, 222)  — 81 frames, 20-frame overlap with clip 1
      Stitching (on overlap, keep the LATER clip's frames):
        positions 0-79    from clip 0  (80 frames, drop 1 overlapping tail)
        positions 80-140  from clip 1  (61 frames, drop 20 overlapping tail)
        positions 141-221 from clip 2  (81 frames)
        Total: 80 + 61 + 81 = 222

    If total_frames <= num_frames_per_clip: single clip [(0, total_frames)].
    The caller should pad conditions to num_frames_per_clip via slice_frames.

    Returns:
        List of (start, end) tuples. Each clip covers frames[start:end].
    """
    if total_frames <= num_frames_per_clip:
        return [(0, total_frames)]

    clips = []
    start = 0
    stride = num_frames_per_clip - 1  # 1-frame overlap between adjacent clips
    # Add regular clips while a full clip fits before the end
    while start + num_frames_per_clip < total_frames:
        clips.append((start, start + num_frames_per_clip))
        start += stride
    # Anchor last clip at the end so it has exactly num_frames_per_clip frames
    clips.append((total_frames - num_frames_per_clip, total_frames))
    return clips


def slice_frames(all_frames: List[Image.Image], start: int, end: int) -> List[Image.Image]:
    """
    Slice frames[start:end]. If end > len(all_frames), pad by repeating the last frame.
    """
    n = len(all_frames)
    if end <= n:
        return all_frames[start:end]
    result = list(all_frames[start:n])
    result += [all_frames[-1]] * (end - n)
    return result


### Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(description="VACE multi-control + I2V inference")

    # sample info
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation, or a path to a .txt file whose contents are read as the prompt")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the stylized first frame image (or a video, whose first frame is used)")
    parser.add_argument("--condition_0_path", type=str, default=None, help="Path to condition 0 video file")
    parser.add_argument("--condition_1_path", type=str, default=None, help="Path to condition 1 video file")
    parser.add_argument("--condition_2_path", type=str, default=None, help="Path to condition 2 video file")
    parser.add_argument("--original_video_path", type=str, default=None, help="Path to the original (source) video. Used for visualization (saved copy, flip-test, side-by-side) and, when --output_fps=source, to read the source frame rate")

    # output
    parser.add_argument("--result_save_folder", type=str, default="result_test/", help="Path to save output data")
    parser.add_argument("--output_fps", type=str, default="source",
        help="Output mp4 FPS: a number, or 'source' (default) to match the --original_video_path "
             "source video's frame rate. Save-layer only — does NOT affect generation. Falls back "
             "to 16 if 'source' is requested but no source video is available.")

    # model
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to model checkpoint file")

    # inference config
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--num_frames_per_clip", type=int, default=49, help="Number of frames per clip; must be 1 plus a multiple of 4 (e.g. 45, 49, 81, 121)")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--vace_scale", type=float, default=1.0, help="VACE conditioning scale")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for generation")

    # Anti-drift (SVI) padding (fills unused I2V frames with the stylized first frame)
    parser.add_argument("--ref_pad_num", type=int, default=None, help="Anti-drift (SVI) padding with the stylized first frame: -1=full padding, 0/None=zero padding (original), N=N-frame padding")

    # multi-clip control
    parser.add_argument("--max_num_frames", type=int, default=None, help="Cap on total output frames. If set, truncate condition videos to this length before computing clip schedule. Avoids wasting compute on frames beyond the cap.")
    parser.add_argument("--different_seed_per_clip", action="store_true", help="If set, each clip uses a different seed (seed + 42*clip_idx). Default: all clips use the same seed.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt. Defaults to the Wan 2.1 default Chinese quality-degradation prompt (see DEFAULT_NEGATIVE_PROMPT in source).")

    # base model paths (default includes CLIP encoder for I2V)
    parser.add_argument("--local_model_path", type=str, default="./models", help="Local HF cache path")
    parser.add_argument("--model_id_with_origin_paths", type=str, default="Wan-AI/Wan2.1-I2V-14B-720P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        help="Comma-separated model_id:pattern pairs. Default includes I2V-14B-720P, T5, VAE, and CLIP.")
    parser.add_argument("--vace_checkpoint_path", type=str, default=None,
        help="Path to VACE-14B checkpoint for selective VACE loading.")

    # keyframe injection (disabled by default; both must be provided together)
    parser.add_argument("--key_frame_paths", type=str, default=None,
        help="Comma-separated paths to keyframe images. Must pair with --key_frame_indices.")
    parser.add_argument("--key_frame_indices", type=str, default=None,
        help="Comma-separated 0-based frame indices for keyframes. Index 0 is reserved for --input_image. Must pair with --key_frame_paths.")

    # acceleration and debug
    parser.add_argument("--use_usp", action="store_true", help="Whether to use USP mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (forces num_inference_steps=2 for a quick smoke test)")

    # output verbosity: when off, only the always-saved set (run_config.json,
    # first_frame.png, original_video.mp4, generated_video.mp4, flip_test.mp4)
    # is written. when on, also save per-condition mp4s, per-clip mp4s, and
    # the generated-vs-original side-by-side viz.
    parser.add_argument("--save_verbose", action=argparse.BooleanOptionalAction, default=True,
        help="Save the verbose set of intermediates (conditions, per-clip mp4s, side-by-side viz). "
             "Use --no-save_verbose for a minimal output set.")

    args = parser.parse_args()

    if args.model_checkpoint is not None and args.model_checkpoint.lower() == "none":
        args.model_checkpoint = None

    # The Wan VACE engine rounds the per-clip frame count up to the nearest (4k+1)
    # internally (time_division_factor=4). The multi-clip schedule + stitching below
    # assume each clip is EXACTLY num_frames_per_clip frames, so any value not
    # congruent to 1 (mod 4) would desync the schedule from the real output length
    # (misaligned stitch or a shape mismatch). Require 4k+1 up front with a clear error.
    if (args.num_frames_per_clip - 1) % 4 != 0:
        raise ValueError(
            f"--num_frames_per_clip must be 1 plus a multiple of 4 (e.g. 45, 49, 81, 121); "
            f"got {args.num_frames_per_clip}. The Wan VACE engine rounds frame counts to the "
            f"nearest 4k+1, which would desync the multi-clip schedule from the generated length.")

    # If --prompt is a .txt file path, read its contents as the prompt
    if args.prompt.endswith(".txt") and os.path.isfile(args.prompt):
        with open(args.prompt, "r") as f:
            args.prompt = f.read().strip()

    # Parse --key_frame_paths and --key_frame_indices (comma-separated, robust parsing)
    has_kf_paths = args.key_frame_paths is not None and args.key_frame_paths.strip().lower() not in ("", "none")
    has_kf_indices = args.key_frame_indices is not None and args.key_frame_indices.strip().lower() not in ("", "none")
    if has_kf_paths != has_kf_indices:
        raise ValueError("--key_frame_paths and --key_frame_indices must both be provided or both omitted.")
    if has_kf_paths:
        kf_paths = [p.strip() for p in args.key_frame_paths.split(",") if p.strip()]
        kf_indices = []
        for s in args.key_frame_indices.split(","):
            s = s.strip()
            if s:
                kf_indices.append(int(s))
        if len(kf_paths) != len(kf_indices):
            raise ValueError(
                f"--key_frame_paths has {len(kf_paths)} entries but --key_frame_indices has {len(kf_indices)}. They must match.")
        # Indices are 0-based; index 0 is reserved for --input_image
        for idx in kf_indices:
            if idx < 1:
                raise ValueError(
                    f"key_frame_indices are 0-based and must be >= 1 (index 0 is --input_image). Got {idx}.")
        args.key_frame_paths = kf_paths
        args.key_frame_indices = kf_indices
    else:
        args.key_frame_paths = None
        args.key_frame_indices = None

    return args


### Fast-load path: skip base DiT + VACE-14B when finetuned checkpoint is available

# Configs for I2V-14B-720P DiT and VACE-14B
# Source of truth: WanModel.state_dict_converter().from_civitai (hash 6bfcfb3b...)
#                  VaceWanModel.state_dict_converter().from_civitai (hash 3b272638...)
I2V_14B_DIT_CONFIG = {
    "has_image_input": True, "patch_size": [1, 2, 2], "in_dim": 36,
    "dim": 5120, "ffn_dim": 13824, "freq_dim": 256, "text_dim": 4096,
    "out_dim": 16, "num_heads": 40, "num_layers": 40, "eps": 1e-6,
}
VACE_14B_CONFIG = {
    "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35), "vace_in_dim": 96,
    "patch_size": (1, 2, 2), "has_image_input": False, "dim": 5120,
    "num_heads": 40, "ffn_dim": 13824, "eps": 1e-6,
}

def _load_finetuned_dit_vace(pipe, checkpoint_path, torch_dtype=torch.bfloat16):
    """Instantiate empty DiT + VACE, load finetuned weights directly.
    Skips loading base I2V DiT and VACE-14B (~56GB disk I/O saved).
    """
    print(f"  Instantiating empty DiT + VACE with hardcoded configs...")
    with init_weights_on_device():
        pipe.dit = WanModel(**I2V_14B_DIT_CONFIG)
        pipe.vace = VaceWanModel(**VACE_14B_CONFIG)

    print(f"  Loading checkpoint: {checkpoint_path}")
    start_time = time.time()
    # Memory-map the checkpoint instead of copying it fully into RAM. Big win for the
    # multi-GPU (USP) path: every rank loads the SAME file, and with mmap they share the
    # OS page cache, so the checkpoint is read from disk ONCE instead of once-per-rank —
    # much faster on slow/network storage and far lower peak CPU RAM (weights stay
    # file-backed until the .to(dtype) below materializes them). Identical weights.
    # (.safetensors already mmaps via safe_open; use torch.load(mmap=True) for .pth/.pt.)
    if checkpoint_path.endswith((".safetensors", ".sft")):
        state_dict = load_state_dict(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    print(f"  Checkpoint loaded in {time.time() - start_time:.1f}s")

    state_dict_vace = {k: v for k, v in state_dict.items() if "vace" in k}
    state_dict_dit = {k: v for k, v in state_dict.items() if "vace" not in k}

    pipe.dit.load_state_dict(state_dict_dit, assign=True)
    pipe.vace.load_state_dict(state_dict_vace, assign=True)
    pipe.dit = pipe.dit.to(dtype=torch_dtype)
    pipe.vace = pipe.vace.to(dtype=torch_dtype)
    print(f"  Loaded DiT ({len(state_dict_dit)} params) + VACE ({len(state_dict_vace)} params)")


if __name__ == "__main__":
    args = parse_args()

    if not args.use_usp:
        print("Not Using USP mode")

    if args.debug:
        print("Debug mode is on. Setting num_inference_steps=2 and only run one sample.")
        args.num_inference_steps = 2

    # Resolve --output_fps before ANY video is written. "source" (default) -> encode all outputs
    # at the source video's fps so they play at the same rate as the source; a number -> use it
    # verbatim. Save-layer only (no effect on generation). FPS must come from --original_video_path
    # (the input image, if any, has no fps); fall back to 16 if no source video is available.
    if isinstance(args.output_fps, str) and args.output_fps.strip().lower() == "source":
        if args.original_video_path is not None and os.path.isfile(args.original_video_path):
            args.output_fps = get_source_fps(args.original_video_path, default_fps=16.0)
            print(f"output_fps=source -> {args.output_fps:.6g} fps (from {args.original_video_path})")
        else:
            args.output_fps = 16.0
            print("output_fps=source but no --original_video_path available; falling back to 16 fps")
    else:
        args.output_fps = float(args.output_fps)

    # Load and validate keyframe images before model loading (fail fast on bad inputs)
    if args.key_frame_paths is not None:
        key_frame_images = []
        for kf_path in args.key_frame_paths:
            assert os.path.exists(kf_path), f"Keyframe image not found: {kf_path}"
            key_frame_images.append(Image.open(kf_path).convert("RGB"))
        # Check all keyframes have the same resolution
        kf_sizes = [img.size for img in key_frame_images]
        if len(set(kf_sizes)) != 1:
            raise ValueError(f"All keyframe images must have the same resolution. Got: {kf_sizes}")
        print(f"Loaded {len(key_frame_images)} keyframe images (resolution {kf_sizes[0]})")
    else:
        key_frame_images = None

    ### Step 0: Load & validate inputs (fail fast before model loading)

    # Load the stylized first frame (or extract the first frame if a video is given)
    _img_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if os.path.splitext(args.input_image)[1].lower() in _img_ext:
        input_image = Image.open(args.input_image).convert("RGB")
    else:
        input_image = read_video_rgb24_ffmpeg(args.input_image)[0]
        print(f"Extracted first frame from video: {args.input_image}")

    # Center-crop and resize to match condition video processing (same spatial transform)
    input_image = center_crop_and_resize(input_image, args.width, args.height)

    # Filter out None/"none" conditions
    def is_valid_condition_path(p):
        if p is None:
            return False
        p_lower = p.strip().lower()
        return p_lower != "" and p_lower != "none"

    all_condition_paths = [args.condition_0_path, args.condition_1_path, args.condition_2_path]
    conditions_path = [p for p in all_condition_paths if is_valid_condition_path(p)]

    if len(conditions_path) == 0:
        raise ValueError("At least one valid condition path must be provided (condition_0, condition_1, or condition_2)")

    # Load ALL frames from each condition video (no truncation)
    print(f"Using {len(conditions_path)} conditions: {conditions_path}")
    frames_conditions = [load_subsampled_center_cropped_frames(p, args.height, args.width) for p in conditions_path]

    # Check that all condition videos have the same number of frames
    condition_frame_counts = [len(fc) for fc in frames_conditions]
    if len(set(condition_frame_counts)) != 1:
        raise ValueError(
            f"All condition videos must have the same number of frames. "
            f"Got: {dict(zip(conditions_path, condition_frame_counts))}"
        )
    total_frames = condition_frame_counts[0]
    print(f"Condition videos: {total_frames} frames each")

    # Inject keyframes into condition videos (before truncation so indices refer to full video)
    if key_frame_images is not None:
        # Check keyframe aspect ratio vs condition frames (tolerance 0.1)
        cond_w, cond_h = frames_conditions[0][0].size
        cond_ar = cond_w / cond_h
        for i, kf_img in enumerate(key_frame_images):
            kf_w, kf_h = kf_img.size
            kf_ar = kf_w / kf_h
            if abs(kf_ar - cond_ar) > 0.1:
                raise ValueError(
                    f"Keyframe {i} aspect ratio ({kf_ar:.4f}, size {kf_w}x{kf_h}) differs from "
                    f"condition video ({cond_ar:.4f}, size {cond_w}x{cond_h}) by more than 0.1.")
        # Validate indices are within range. Indices must lie within the EFFECTIVE
        # output length: min(total_frames, --max_num_frames). Keyframes beyond this
        # would be injected then truncated away.
        effective_max = total_frames if args.max_num_frames is None else min(total_frames, args.max_num_frames)
        for idx in args.key_frame_indices:
            if idx >= effective_max:
                raise ValueError(
                    f"key_frame_indices {idx} >= effective output length {effective_max} "
                    f"(min of total_frames={total_frames} and --max_num_frames={args.max_num_frames}). "
                    f"Max valid index is {effective_max - 1}.")
        # Replace frames in each condition video at the specified indices
        for kf_img, kf_idx in zip(key_frame_images, args.key_frame_indices):
            resized_kf = center_crop_and_resize(kf_img, args.width, args.height)
            for c in range(len(frames_conditions)):
                frames_conditions[c][kf_idx] = resized_kf
            print(f"  Replaced frame {kf_idx} in all {len(frames_conditions)} conditions with keyframe")

    # Truncate to --max_num_frames if set (avoid computing clips beyond the cap)
    if args.max_num_frames is not None and total_frames > args.max_num_frames:
        print(f"Truncating from {total_frames} to --max_num_frames={args.max_num_frames}")
        frames_conditions = [fc[:args.max_num_frames] for fc in frames_conditions]
        total_frames = args.max_num_frames

    # Load ALL frames from original video (for visualization, optional)
    if args.original_video_path is not None:
        frames_original_video = load_subsampled_center_cropped_frames(args.original_video_path, args.height, args.width)

    # Compute multi-clip schedule from condition video length
    clip_schedule = compute_clip_schedule(total_frames, args.num_frames_per_clip)
    num_clips = len(clip_schedule)
    print(f"Clip schedule ({num_clips} clips, num_frames_per_clip={args.num_frames_per_clip}):")
    for i, (s, e) in enumerate(clip_schedule):
        overlap = clip_schedule[i-1][1] - s if i > 0 else 0
        print(f"  Clip {i}: frames [{s}, {e}) = {e - s} frames" +
              (f", overlap={overlap} with clip {i-1}" if overlap > 0 else ""))

    has_finetuned_checkpoint = args.model_checkpoint is not None

    ### Step 1: Load base models (skip DiT if finetuned checkpoint will replace it)
    model_configs = []
    model_id_with_origin_paths = args.model_id_with_origin_paths.split(",")
    for entry in model_id_with_origin_paths:
        model_id, origin_pattern = entry.split(":")
        if has_finetuned_checkpoint and origin_pattern.startswith("diffusion_pytorch_model"):
            print(f"Skipping base DiT load (finetuned checkpoint will replace it): {entry}")
            continue
        model_configs.append(
            ModelConfig(model_id=model_id, origin_file_pattern=origin_pattern, offload_device="cpu")
        )
    tokenizer_config = \
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="google/*", offload_device="cpu")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_usp=args.use_usp,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        local_model_path=args.local_model_path,
        checkpoint_path=None,
        skip_download=True,
        redirect_common_files=False,
    )

    ### Step 2: Load DiT + VACE weights
    if has_finetuned_checkpoint:
        # Fast path: skip base DiT + VACE-14B, load finetuned weights directly
        print(f"Fast-loading finetuned checkpoint (skipping base DiT + VACE-14B)...")
        _load_finetuned_dit_vace(pipe, args.model_checkpoint, torch_dtype=torch.bfloat16)
    else:
        # Original path: DiT already loaded from from_pretrained; load VACE separately
        if args.vace_checkpoint_path is not None and pipe.vace is None:
            print(f"Loading VACE selectively from {args.vace_checkpoint_path}")
            pipe.vace = load_vace_from_checkpoint(
                args.vace_checkpoint_path, torch_dtype=torch.bfloat16, device="cpu")
        print("No model checkpoint provided, using the original pretrained weights")

    # Verify I2V DiT loaded correctly
    assert pipe.dit is not None, "DiT was not loaded. Provide --model_checkpoint or include DiT in --model_id_with_origin_paths."
    assert pipe.dit.has_image_input, "Expected I2V DiT. Check --model_id_with_origin_paths."
    print(f"  dit.has_image_input = {pipe.dit.has_image_input}")
    print(f"  dit.patch_embedding.weight.shape = {pipe.dit.patch_embedding.weight.shape}")

    # Enable USP after dit is loaded (skipped inside from_pretrained when dit was None)
    if args.use_usp and not getattr(pipe, "use_unified_sequence_parallel", False):
        pipe.enable_usp()

    if args.use_usp:
        pipe.enable_vram_management(vram_buffer=25)
    else:
        pipe.enable_vram_management(vram_buffer=10) # 5 is enough for 720p

    ### Step 3: Save processed inputs
    if (args.use_usp and dist.get_rank() == 0) or (not args.use_usp):
        os.makedirs(args.result_save_folder, exist_ok=True)
        # Always-saved set
        input_image.save(os.path.join(args.result_save_folder, "first_frame.png"))
        if args.original_video_path is not None:
            save_frames_to_video(frames_original_video[:total_frames], os.path.join(args.result_save_folder, "original_video.mp4"), fps=args.output_fps)
        # Resolved run config (incl. prompt) as JSON
        import json
        run_config = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT,
            "model_checkpoint": args.model_checkpoint,
            "width": args.width, "height": args.height,
            "num_frames_per_clip": args.num_frames_per_clip,
            "max_num_frames": args.max_num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale, "vace_scale": args.vace_scale,
            "ref_pad_num": args.ref_pad_num, "seed": args.seed,
            "output_fps": args.output_fps,
            "key_frame_indices": list(args.key_frame_indices) if args.key_frame_indices else [],
            "save_verbose": args.save_verbose,
        }
        with open(os.path.join(args.result_save_folder, "run_config.json"), "w") as f:
            json.dump(run_config, f, indent=2)
        # Verbose-only: condition videos as fed to the model
        if args.save_verbose:
            for i, frames in enumerate(frames_conditions):
                save_frames_to_video(frames, os.path.join(args.result_save_folder, f"condition_{i}.mp4"), fps=args.output_fps)

    ### Step 4: Run inference with VACE + I2V (multi-clip with per-clip condition slicing)
    NEGATIVE_PROMPT = args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

    # Construct VACE mask in memory: white (reactive=1) everywhere, black (inactive=0)
    # at keyframe indices so the injected keyframe RGB is hard-pinned via the passthrough
    # branch instead of being regenerated. Frame 0 is the I2V anchor and never a keyframe
    # (key_frame_indices are validated >= 1 above), so it stays white.
    white_frame = Image.new("RGB", (args.width, args.height), (255, 255, 255))
    black_frame = Image.new("RGB", (args.width, args.height), (0, 0, 0))
    kf_set = set(args.key_frame_indices) if args.key_frame_indices else set()
    full_mask_frames = [black_frame if i in kf_set else white_frame for i in range(total_frames)]

    all_clips = []
    current_input_image = input_image  # PIL Image; updated each clip for multi-clip chaining

    for clip_idx, (frame_start, frame_end) in enumerate(clip_schedule):
        # Seed: same for all clips unless --different_seed_per_clip
        clip_seed = (args.seed + 42 * clip_idx) if args.different_seed_per_clip else args.seed
        print(f"\n--- Clip {clip_idx+1}/{num_clips} frames=[{frame_start}, {frame_end}) (seed={clip_seed}) ---")

        if clip_idx > 0:
            # Use the previous clip's frame at the splice point as the I2V anchor.
            # This clip takes over at global frame clip_schedule[clip_idx][0],
            # so B0 ≈ A[splice] for a smooth handoff.
            splice_idx = clip_schedule[clip_idx][0] - clip_schedule[clip_idx - 1][0]
            current_input_image = all_clips[-1][splice_idx]

        # Slice the correct temporal portion of each condition for this clip
        # Always pad to num_frames_per_clip (slice_frames repeats last frame if needed)
        clip_end = frame_start + args.num_frames_per_clip
        clip_conditions = [slice_frames(cond, frame_start, clip_end) for cond in frames_conditions]
        # Slice the global mask (black at keyframe indices, white elsewhere) for this clip
        clip_mask_single = slice_frames(full_mask_frames, frame_start, clip_end)
        clip_mask = [clip_mask_single] * len(conditions_path)

        generated_video = pipe(
            prompt=args.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            input_image=current_input_image,
            random_ref_frame=input_image,  # Always the user-provided stylized first frame for padding
            ref_pad_num=args.ref_pad_num,
            vace_video=clip_conditions,
            vace_video_mask=clip_mask,
            seed=clip_seed, num_inference_steps=args.num_inference_steps,
            use_multi_control_vace=True,
            height=args.height, width=args.width, num_frames=args.num_frames_per_clip,
            cfg_scale=args.cfg_scale,
            tiled=False,
            vace_scale=args.vace_scale,
        )
        all_clips.append(generated_video)
        print(f"  Clip {clip_idx+1}: {len(generated_video)} frames generated")

        # Save each clip immediately after generation (crash-safe incremental save) — verbose only
        if args.save_verbose and ((args.use_usp and dist.get_rank() == 0) or (not args.use_usp)):
            clip_arr = np.stack([np.array(frame) for frame in generated_video], axis=0)
            save_video_fn(clip_arr, os.path.join(args.result_save_folder, f"generated_clip{clip_idx}.mp4"), save_gif=False, fps=args.output_fps)
            print(f"  Saved clip {clip_idx} to {args.result_save_folder}/generated_clip{clip_idx}.mp4")

    # Stitch clips: on overlapping frames, keep the later clip's frames
    if num_clips == 1:
        combined_video = all_clips[0]
        # If condition video was shorter than num_frames_per_clip, trim to total_frames
        if total_frames < args.num_frames_per_clip:
            combined_video = combined_video[:total_frames]
    else:
        combined_video = list(all_clips[0])
        for i in range(1, num_clips):
            overlap = clip_schedule[i-1][1] - clip_schedule[i][0]
            combined_video = combined_video[:-overlap] + list(all_clips[i])  # keep later clip's frames
        print(f"Stitched {num_clips} clips into {len(combined_video)} frames")

    ### Step 5: Save results + visualization
    generated_video_save_path = os.path.join(args.result_save_folder, "generated_video.mp4")

    if (args.use_usp and dist.get_rank() == 0) or (not args.use_usp):
        # Per-clip mp4s (verbose, multi-clip only). Redundant with the
        # incremental per-clip saves in the loop above; re-emitted here as a
        # safety net so the final output always includes every clip.
        if args.save_verbose and num_clips > 1:
            for i, clip in enumerate(all_clips):
                clip_arr = np.stack([np.array(frame) for frame in clip], axis=0)
                save_video_fn(clip_arr, os.path.join(args.result_save_folder, f"generated_clip{i}.mp4"), save_gif=False, fps=args.output_fps)

        # Final combined video (always)
        video_to_save = np.stack([np.array(frame) for frame in combined_video], axis=0)
        save_video_fn(video_to_save, generated_video_save_path, save_gif=False, fps=args.output_fps)

        # Flip-test viz (always). Operates on in-memory PIL frames — no re-reads.
        if args.original_video_path is not None and len(frames_original_video) > 0:
            from idv2v.inference.flip_test import build_flip_test
            n_compare = min(len(combined_video), len(frames_original_video))
            build_flip_test(
                combined_video[:n_compare],
                frames_original_video[:n_compare],
                os.path.join(args.result_save_folder, "flip_test.mp4"),
                output_fps=args.output_fps,
            )

        # Side-by-side viz (verbose only)
        if args.save_verbose and args.original_video_path is not None:
            gen_orig_out_path = generated_video_save_path.replace(".mp4", "_withOrig.mp4")
            stack_videos_left_right_and_save(
                [generated_video_save_path, args.original_video_path],
                gen_orig_out_path, fps=args.output_fps,
            )

        print(f"Saved generated video to {generated_video_save_path}")

        print("Inference done.")
