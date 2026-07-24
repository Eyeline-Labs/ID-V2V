#!/bin/bash
# EXAMPLE — Video RELIGHTING: change ONLY the lighting; preserve the FULL scene (human + background).
#
# How it works: NO preprocessing (no SAM3, no gray) — the RAW source.mp4 is the single VACE
# condition, so lighting is re-rendered from stylized_first_frame.png + prompt.
# Engine: scripts/infer_relighting.sh.
#
# Run from the repo root (fetch weights first: bash scripts/download_checkpoints.sh):
#   bash scripts/examples/relighting.sh
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

# MODEL_CHECKPOINT is optional: defaults to checkpoints/idv2v.pth (fetched by scripts/download_checkpoints.sh);
# set MODEL_CHECKPOINT=/local/idv2v.pth to use your own.
export GPU="${GPU:-0,1,2,3,4,5,6,7}"
export SAMPLE_DIR="${SAMPLE_DIR:-test_samples/relighting/two_sitting_woman}"  # source.mp4 + stylized_first_frame.png + prompt.txt (NO preprocessing)
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-81}"   # first clip only (81 frames).

echo "Inference: idv2v RELIGHTING (condition_0 = raw source.mp4; no SAM3, no gray, no preprocessing)"
bash scripts/infer_relighting.sh
