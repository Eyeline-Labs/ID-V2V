#!/bin/bash
# EXAMPLE — LONGER video via chunk-by-chunk generation.
#
# Same idv2v engine; the only change is MAX_NUM_FRAMES = the full source length. The pipeline
# auto-splits into overlapping 81-frame (NUM_FRAMES_PER_CLIP) clips and stitches them
# (240 frames -> 3 clips; adjacent clips overlap 1 frame, and the final clip may overlap more since it is
# anchored to end on the last frame). The chunking is automatic and identical to the short
# examples, which simply resolve to a single 81-frame clip.
#
# Run from the repo root (fetch weights first: bash scripts/download_checkpoints.sh):
#   bash scripts/examples/longer_video.sh
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

# MODEL_CHECKPOINT is optional: defaults to checkpoints/idv2v.pth (fetched by scripts/download_checkpoints.sh);
# set MODEL_CHECKPOINT=/local/idv2v.pth to use your own.
export GPU="${GPU:-0,1,2,3,4,5,6,7}"
export GPU_ID="${GPU_ID:-${GPU%%,*}}"                                         # SAM3 preprocess uses ONE GPU (first of $GPU)
export SAMPLE_DIR="${SAMPLE_DIR:-test_samples/longer_video/woman_dancing}"    # full-length source (240 frames)
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-240}"   # full length -> multi-clip (240 -> 3 x 81-frame chunks).

echo "[1/2] Preprocess: SAM3 -> foreground-on-gray condition (full-length source)"
bash scripts/preprocess.sh

echo "[2/2] Inference: idv2v chunk-by-chunk (MAX_NUM_FRAMES=$MAX_NUM_FRAMES -> multiple 81-frame clips)"
bash scripts/infer.sh
