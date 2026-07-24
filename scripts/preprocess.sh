#!/bin/bash
# ID-V2V preprocessing (idv2v): derive the VACE condition — foreground-on-gray pixels.
#
# Derives the control signal the idv2v model consumes:
#   SAM3 (person segmentation) -> orig_pixel (foreground-on-gray pixels).
#
# Reads:  <SAMPLE_DIR>/source.mp4
# Writes: <SAMPLE_DIR>/preprocessing/orig_pixel.mp4  (+ a SAM3 overlay for debugging)
#
# Usage: edit the variables below (or override via env), then run from anywhere:
#     bash scripts/preprocess.sh
set -e
cd "$(dirname "$0")/.."           # repo root — so relative paths and .venv resolve
source .venv/bin/activate

# ====================================================================
# EDIT THESE 3 VARIABLES BEFORE RUNNING (or override any via environment variable,
# e.g. `SAMPLE_DIR=... bash scripts/preprocess.sh`). See checkpoints/README.md.
# ====================================================================
SAMPLE_DIR="${SAMPLE_DIR:-test_samples/restylization/two_sitting_woman}"   # dir containing source.mp4
# Default assumes you ran `bash scripts/download_checkpoints.sh`. Override if needed.
SAM3_CKPT="${SAM3_CKPT:-checkpoints/sam3}"
GPU_ID="${GPU_ID:-0}"

# Optional
SAM_PROMPT="${SAM_PROMPT:-person}"   # SAM3 text prompt = what to segment: "person" (default), "head", "dog", ...
CLEANUP=true          # delete per-frame frame_*/ folders after pipeline finishes

SOURCE_VIDEO="${SAMPLE_DIR}/source.mp4"
PREPROC_DIR="${SAMPLE_DIR}/preprocessing"
mkdir -p "$PREPROC_DIR"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Step 1: SAM3 segmentation + Secret Panda mask cleanup
python -m idv2v.preprocess.sam3 \
    --video_path "$SOURCE_VIDEO" \
    --sam_prompt "$SAM_PROMPT" \
    --output_dir "$PREPROC_DIR" \
    --model_path "$SAM3_CKPT" \
    --joint_mask_post_proc

# Step 2: foreground-on-gray — the single VACE condition idv2v consumes
python -m idv2v.preprocess.orig_pixel \
    --video_path "$SOURCE_VIDEO" \
    --mask_folder "$PREPROC_DIR" \
    --mask_image_file_name "sam3Mask_id_all.png" \
    --result_save_path "$PREPROC_DIR/orig_pixel.mp4"

if [ "$CLEANUP" = "true" ]; then
    rm -rf "$PREPROC_DIR"/frame_*/
fi

echo ""
echo "Done. Condition video in: $PREPROC_DIR"
echo "  orig_pixel.mp4   (the VACE condition for idv2v)"
echo "  *sam3_overlay.mp4 (debug viz)"
