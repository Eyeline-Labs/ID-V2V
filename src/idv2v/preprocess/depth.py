"""
DepthAnything-V2 dense-depth condition video. Runs the vendored VACE depth
annotator on the source video, writes <save_path>/src_video-depthv2.mp4, and
verifies the output keeps the source's resolution and frame count.

The DepthV2 checkpoint path comes from env var IDV2V_DEPTHV2_CKPT (set by
scripts/idv2v_with_normal_depth/preprocess_with_depth.sh) via
vace_annotators/configs/video_preproccess.py. Depth is only used by the alternate
3-condition "idv2v_with_normal_depth" model.
"""
from .vace_annotators.vace_preproccess import run_vace_preprocess

import cv2
import os
import csv
import random
from tqdm import tqdm
import shutil
import csv
import argparse

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def annotate_single_video(video_path, save_path):
    task = "depthv2"

    assert os.path.exists(video_path), f"the input {os.path.abspath(video_path)} does not exist"

    save_fps = get_video_fps(video_path)
    result = run_vace_preprocess(
        task=task,
        video=video_path,
        save_fps=save_fps,
        pre_save_dir=save_path
    )  # writes save_path/src_video-{task}.mp4

    # Validate that the depth video preserves source resolution + frame count.
    cap = cv2.VideoCapture(video_path)
    orig_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    depth_video_path = f"{save_path}/src_video-{task}.mp4"
    cap_depth = cv2.VideoCapture(depth_video_path)
    depth_frame_count = int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))
    depth_width = int(cap_depth.get(cv2.CAP_PROP_FRAME_WIDTH))
    depth_height = int(cap_depth.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_depth.release()

    frame_count_match = False
    resolution_match = False

    if orig_frame_count == depth_frame_count:
        frame_count_match = True
    if orig_width == depth_width and orig_height == depth_height:
        resolution_match = True

    if not frame_count_match or not resolution_match:
        # On mismatch, drop the broken output so the next run starts clean.
        os.remove(f"{save_path}/src_video-{task}.mp4")
        raise ValueError(f"WARNING: Frame count or resolution mismatch! Original: {orig_frame_count}x{orig_width}x{orig_height}, Depth: {depth_frame_count}x{depth_width}x{depth_height}")

    return f"{save_path}/src_video-{task}.mp4"

def parse_args():
    parser = argparse.ArgumentParser(description="DepthAnything-V2 dense depth on a single video.")
    parser.add_argument("--video_path", type=str, help="Input video.")
    parser.add_argument("--save_path", type=str, help="Output directory; depth video is written to <save_path>/src_video-depthv2.mp4.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    video_path = args.video_path
    save_path = args.save_path

    print(f"start processing depthv2 of {os.path.abspath(video_path)}")
    save_path_print = annotate_single_video(video_path, save_path)
    print(f'Finished DepthV2: Saved the depthv2 result to {os.path.abspath(f"{save_path}/src_video-depthv2.mp4")}')