#!/bin/bash
# EXAMPLE — Multiple keyframes: FIRST + LAST frame.
#
# stylized_first_frame.png pins frame 0 (the I2V anchor, keyframe index 0); keyframes/80.png pins
# the LAST frame (index 80 of the 81-frame clip). infer.sh auto-discovers keyframes/<N>.png, so no
# extra flags are needed — adding more (e.g. keyframes/40.png) pins more frames. Preprocessing is the
# same as restylization (SAM3 -> orig_pixel).
#
# Run from the repo root (fetch weights first: bash scripts/download_checkpoints.sh):
#   bash scripts/examples/first_last_frame.sh
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

# MODEL_CHECKPOINT is optional: defaults to checkpoints/idv2v.pth (fetched by scripts/download_checkpoints.sh);
# set MODEL_CHECKPOINT=/local/idv2v.pth to use your own.
export GPU="${GPU:-0,1,2,3,4,5,6,7}"
export GPU_ID="${GPU_ID:-${GPU%%,*}}"                                         # SAM3 preprocess uses ONE GPU (first of $GPU)
export SAMPLE_DIR="${SAMPLE_DIR:-test_samples/first_last_frame/two_women_spotlight}"   # + keyframes/80.png (the last frame)
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-81}"

echo "[1/2] Preprocess: SAM3 -> foreground-on-gray condition"
bash scripts/preprocess.sh

echo "[2/2] Inference: idv2v with the stylized first frame + keyframes/80.png (auto-detected -> kf1)"
bash scripts/infer.sh
