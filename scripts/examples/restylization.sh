#!/bin/bash
# EXAMPLE — Video RESTYLIZATION: change the scene AND lighting, preserve the human.
#
# How it works: SAM3 segments the person and grays out the background (scripts/preprocess.sh ->
# preprocessing/orig_pixel.mp4), then the idv2v model regenerates a NEW background/scene + lighting
# around the preserved subject (scripts/infer.sh, single VACE condition = orig_pixel.mp4).
#
# NON-ALIGNED KEYFRAME (feature) uses this SAME command — just set
#   SAMPLE_DIR=test_samples/non_aligned_keyframe/suits   (or .../woman_phone)
# There the stylized_first_frame.png is deliberately a very different scene from the source;
# nothing else about the command changes.
#
# Run from the repo root (fetch weights first: bash scripts/download_checkpoints.sh):
#   bash scripts/examples/restylization.sh
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

# MODEL_CHECKPOINT is optional: defaults to checkpoints/idv2v.pth (fetched by scripts/download_checkpoints.sh);
# set MODEL_CHECKPOINT=/local/idv2v.pth to use your own.
export GPU="${GPU:-0,1,2,3,4,5,6,7}"                                          # CUDA ids; >1 -> multi-GPU (USP)
export GPU_ID="${GPU_ID:-${GPU%%,*}}"                                         # SAM3 preprocess uses ONE GPU (first of $GPU)
export SAMPLE_DIR="${SAMPLE_DIR:-test_samples/restylization/two_sitting_woman}"   # source.mp4 + stylized_first_frame.png + prompt.txt
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-81}"   # cap on total output frames. The engine is ALWAYS chunk-by-chunk;
                                                # 81 -> a single 81-frame clip (see longer_video.sh for multi-clip).

echo "[1/2] Preprocess: SAM3 -> foreground-on-gray condition (preprocessing/orig_pixel.mp4)"
bash scripts/preprocess.sh

echo "[2/2] Inference: idv2v restylization (condition_0 = orig_pixel.mp4)"
bash scripts/infer.sh
