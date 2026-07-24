"""
In-memory flip-test video utility.

Builds an mp4 that primarily plays the generated video, but periodically
freezes on a frame and alternates between generated and source ("flip")
for a quick A/B comparison. Each frame carries a corner pill labelling it
Generated (green) or Source (red); freeze segments also get a colored border.

Inputs are PIL Image lists already in memory (no file I/O reads).

Schematic of the output timeline (`play_block_frames=20`, `flip_pairs_per_freeze=2`):

    [GEN 0..19]  [GEN19, SRC19, GEN19, SRC19]  [GEN 20..39]  [GEN39, SRC39, ...]  ...
     ^playback   ^freeze: 4 holds = 3 s        ^playback     ^freeze
"""
from __future__ import annotations

import subprocess
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --- Visual style ---------------------------------------------------------

PILL_GREEN = (31, 181, 75)      # GEN
PILL_RED = (214, 61, 61)        # SRC
PILL_TEXT = (255, 255, 255)
PILL_MARGIN_PX = 16             # distance from top-right corner
PILL_PAD_X = 14                 # horizontal padding inside pill
PILL_PAD_Y = 6                  # vertical padding inside pill
PILL_FONT_SIZE = 22             # ~3% of 720p height
FREEZE_BORDER_THICKNESS = 6

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# --- Drawing primitives ---------------------------------------------------

def _stamp_pill(img: Image.Image, label: str, fill_rgb, font: ImageFont.ImageFont) -> Image.Image:
    """Draw a rounded pill in the top-right corner with `label` text."""
    out = img.copy()
    draw = ImageDraw.Draw(out, mode="RGBA")
    # Measure text
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pill_w = text_w + 2 * PILL_PAD_X
    pill_h = text_h + 2 * PILL_PAD_Y
    w, _ = out.size
    x0 = w - PILL_MARGIN_PX - pill_w
    y0 = PILL_MARGIN_PX
    # Pill body (slightly transparent)
    draw.rounded_rectangle(
        (x0, y0, x0 + pill_w, y0 + pill_h),
        radius=pill_h // 2,
        fill=fill_rgb + (235,),
    )
    # Text centered inside pill
    text_x = x0 + (pill_w - text_w) // 2 - bbox[0]
    text_y = y0 + (pill_h - text_h) // 2 - bbox[1]
    draw.text((text_x, text_y), label, fill=PILL_TEXT, font=font)
    return out


def _stamp_border(img: Image.Image, fill_rgb, thickness: int = FREEZE_BORDER_THICKNESS) -> Image.Image:
    """Draw a colored rectangle around the frame edge (for freeze segments)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for i in range(thickness):
        draw.rectangle((i, i, w - 1 - i, h - 1 - i), outline=fill_rgb)
    return out


def _stamp_gen(img: Image.Image, font, with_border: bool = False) -> Image.Image:
    out = _stamp_border(img, PILL_GREEN) if with_border else img
    return _stamp_pill(out, "Generated", PILL_GREEN, font)


def _stamp_src(img: Image.Image, font, with_border: bool = False) -> Image.Image:
    out = _stamp_border(img, PILL_RED) if with_border else img
    return _stamp_pill(out, "Source", PILL_RED, font)


# --- mp4 writer (H.264 CRF 18, same pattern as save_frames_to_video) ------

def _write_mp4(frames: List[Image.Image], save_path: str, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    w, h = frames[0].size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", save_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for img in frames:
        proc.stdin.write(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


# --- Public entry point ---------------------------------------------------

def build_flip_test(
    frames_gen: List[Image.Image],
    frames_src: List[Image.Image],
    out_path: str,
    output_fps: int = 16,
    play_block_frames: int = 20,
    flip_pairs_per_freeze: int = 2,
    flip_hold_sec: float = 0.75,
    max_flips: int = 5,
    show_border_on_freeze: bool = True,
) -> None:
    """Write a flip-test mp4 to `out_path`.

    Plays `frames_gen` in blocks of `play_block_frames` frames at `output_fps`.
    After selected blocks, freezes on the last frame of the block and alternates
    Generated <-> Source for `flip_pairs_per_freeze` pairs (each side held for
    `flip_hold_sec` seconds).

    Total freeze points are capped at `max_flips` (uniformly distributed across
    the video). If the video has at most `max_flips` blocks, every block
    boundary gets a freeze.
    """
    assert len(frames_gen) == len(frames_src), \
        f"frame count mismatch: gen={len(frames_gen)}, src={len(frames_src)}"
    assert len(frames_gen) > 0, "no frames to compare"
    assert frames_gen[0].size == frames_src[0].size, \
        f"resolution mismatch: gen={frames_gen[0].size}, src={frames_src[0].size}"

    n = len(frames_gen)
    hold_n = max(1, int(round(flip_hold_sec * output_fps)))
    font = _load_font(PILL_FONT_SIZE)
    out: List[Image.Image] = []

    # Pick which block boundaries get a freeze. Block i ends at frame (i+1)*play_block_frames - 1.
    total_blocks = (n + play_block_frames - 1) // play_block_frames
    if total_blocks <= max_flips:
        freeze_block_set = set(range(total_blocks))
    else:
        # Uniformly distribute `max_flips` freezes across all blocks.
        idxs = np.linspace(0, total_blocks - 1, max_flips)
        freeze_block_set = {int(round(i)) for i in idxs}

    pos = 0
    block_idx = 0
    while pos < n:
        block_end = min(pos + play_block_frames, n)
        # 1) Playback: GEN frames with pill, no border.
        for i in range(pos, block_end):
            out.append(_stamp_gen(frames_gen[i], font, with_border=False))

        # 2) Freeze at the last frame of the block — but only if this block
        #    is one of the (uniformly chosen) freeze blocks.
        if block_idx in freeze_block_set:
            freeze_idx = block_end - 1
            g_frozen = _stamp_gen(frames_gen[freeze_idx], font, with_border=show_border_on_freeze)
            s_frozen = _stamp_src(frames_src[freeze_idx], font, with_border=show_border_on_freeze)
            for _ in range(flip_pairs_per_freeze):
                out.extend([g_frozen] * hold_n)
                out.extend([s_frozen] * hold_n)

        pos = block_end
        block_idx += 1

    _write_mp4(out, out_path, output_fps)
