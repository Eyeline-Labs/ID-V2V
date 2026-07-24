#!/bin/bash
# ID-V2V preprocessing — ALTERNATE "idv2v_with_normal_depth" model (THREE VACE conditions).
# Derives all three control signals: SAM3 -> orig_pixel (foreground-on-gray pixels), DAViD
# (surface-normals), DepthAnything-V2 (depth). The DEFAULT idv2v model needs only the first
# one — see scripts/preprocess.sh. Requires the DAViD + DepthV2 weights (download them with
# `bash scripts/download_checkpoints.sh --with-depth`).
#
# Reads:  <SAMPLE_DIR>/source.mp4
# Writes: <SAMPLE_DIR>/preprocessing/{orig_pixel,david_normal,depth}.mp4 (+ *sam3_overlay.mp4 debug viz)
#
# Usage: edit the variables below (or override via env), then run from anywhere:
#     bash scripts/idv2v_with_normal_depth/preprocess_with_depth.sh
set -e
cd "$(dirname "$0")/../.."        # repo root — so relative paths and .venv resolve
source .venv/bin/activate

# ====================================================================
# EDIT THESE 5 VARIABLES BEFORE RUNNING. See checkpoints/README.md.
# ====================================================================
SAMPLE_DIR="${SAMPLE_DIR:-test_samples/restylization/two_sitting_woman}"   # dir containing source.mp4
# Defaults match the standard checkpoints/ layout (from `scripts/download_checkpoints.sh --with-depth`). Override if needed.
SAM3_CKPT="${SAM3_CKPT:-checkpoints/sam3}"
DAVID_CKPT="${DAVID_CKPT:-checkpoints/david/multi-task-model-vitl16_384.onnx}"
DEPTHV2_CKPT="${DEPTHV2_CKPT:-checkpoints/depthv2/depth_anything_v2_vitl.pth}"
GPU_ID="${GPU_ID:-0}"

# Optional
SAM_PROMPT="${SAM_PROMPT:-person}"   # SAM3 text prompt = what to segment: "person" (default), "head", "dog", ...
CLEANUP=true          # delete per-frame frame_*/ folders after pipeline finishes

SOURCE_VIDEO="${SAMPLE_DIR}/source.mp4"
PREPROC_DIR="${SAMPLE_DIR}/preprocessing"
mkdir -p "$PREPROC_DIR"

# DAViD uses onnxruntime-gpu (CUDA 12) but torch in this env is cu118.
# On first run, install cu12 libs into a sibling dir; always export them via LD_LIBRARY_PATH.
NVIDIA_CU12="$(pwd)/.venv/cuda12_libs/nvidia"
if [ ! -f "$NVIDIA_CU12/cuda_runtime/lib/libcudart.so.12" ]; then
    echo "First-run: installing CUDA 12 libs to .venv/cuda12_libs/ ..."
    uv pip install --target "$(pwd)/.venv/cuda12_libs" --quiet --no-deps \
        nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
        nvidia-curand-cu12 nvidia-cudnn-cu12==9.1.0.70
fi
for d in "$NVIDIA_CU12"/*/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

# DepthV2 ckpt is read by VACE config via this env var.
export IDV2V_DEPTHV2_CKPT="$DEPTHV2_CKPT"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Step 1: SAM3 segmentation + Secret Panda mask cleanup
python -m idv2v.preprocess.sam3 \
    --video_path "$SOURCE_VIDEO" \
    --sam_prompt "$SAM_PROMPT" \
    --output_dir "$PREPROC_DIR" \
    --model_path "$SAM3_CKPT" \
    --joint_mask_post_proc

# Step 2: foreground-on-gray
python -m idv2v.preprocess.orig_pixel \
    --video_path "$SOURCE_VIDEO" \
    --mask_folder "$PREPROC_DIR" \
    --mask_image_file_name "sam3Mask_id_all.png" \
    --result_save_path "$PREPROC_DIR/orig_pixel.mp4"

# Step 3: DAViD surface normals
python -m idv2v.preprocess.david \
    --preprocessing_folder "$PREPROC_DIR" \
    --canvas_reference_video_path "$SOURCE_VIDEO" \
    --DAViD_ckpt "$DAVID_CKPT"

# Step 4: DepthAnything-V2 depth (depth.py writes src_video-depthv2.mp4; rename for consistency)
python -m idv2v.preprocess.depth \
    --video_path "$SOURCE_VIDEO" \
    --save_path "$PREPROC_DIR"
mv "$PREPROC_DIR/src_video-depthv2.mp4" "$PREPROC_DIR/depth.mp4"

if [ "$CLEANUP" = "true" ]; then
    rm -rf "$PREPROC_DIR"/frame_*/
fi

echo ""
echo "Done. Condition videos in: $PREPROC_DIR"
echo "  orig_pixel.mp4   (condition 0)"
echo "  david_normal.mp4 (condition 1)"
echo "  depth.mp4        (condition 2)"
echo "  *sam3_overlay.mp4 (debug viz)"
