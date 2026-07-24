#!/bin/bash
# ID-V2V inference — RELIGHTING mode of the idv2v model.
# Same idv2v model/checkpoint and same clip-by-clip I2V engine as scripts/infer.sh,
# but the ONE VACE condition is the RAW source video itself — NO SAM3, NO gray background, NO
# preprocessing. Because the full scene (subject + background) is handed to the model as the
# control, only the LIGHTING changes (driven by stylized_first_frame.png + prompt); the scene is preserved.
# (For "restylization" — change the scene AND lighting, keep only the human — use scripts/infer.sh,
#  which conditions on the SAM3 foreground-on-gray preprocessing/orig_pixel.mp4 instead.)
#
# Reads:  <SAMPLE_DIR>/{source.mp4, stylized_first_frame.png, prompt.txt, keyframes/}   (NO preprocessing/ needed)
# Writes: <SAMPLE_DIR>/outputs/<RUN_NAME>/
#
# Usage: edit the variables below (or override via env), then run from anywhere:
#     bash scripts/infer_relighting.sh
set -e
cd "$(dirname "$0")/.."           # repo root — so relative paths and .venv resolve
source .venv/bin/activate

# ====================================================================
# EDIT THESE 4 VARIABLES BEFORE RUNNING (or override any of the vars below via environment
# variable, e.g. `SAMPLE_DIR=... MAX_NUM_FRAMES=81 bash scripts/infer_relighting.sh`).
# ====================================================================
SAMPLE_DIR="${SAMPLE_DIR:-test_samples/relighting/two_sitting_woman}"  # input dir (source.mp4 + stylized_first_frame.png + prompt.txt)
# The finetuned idv2v .pth — SAME checkpoint as scripts/infer.sh. Defaults to checkpoints/idv2v.pth,
# fetched by scripts/download_checkpoints.sh; set MODEL_CHECKPOINT to use a local copy.
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-checkpoints/idv2v.pth}"     # from scripts/download_checkpoints.sh; override with a local path
WAN_MODEL_DIR="${WAN_MODEL_DIR:-checkpoints/wan}"                 # default: from scripts/download_checkpoints.sh. Override if Wan weights live elsewhere.
GPU="${GPU:-0,1,2,3,4,5,6,7}"                                     # comma-separated CUDA ids; >1 GPU auto-enables torchrun + USP sequence parallel

RUN_NAME=""   # empty -> auto-derived as r{W}x{H}_f{F}_kf{N}_idv2v_relight

# Inference config (720p defaults) — identical to scripts/infer.sh
WIDTH=1280
HEIGHT=720
NUM_FRAMES_PER_CLIP=81
NUM_INFERENCE_STEPS=30
CFG_SCALE=5
VACE_SCALE=1
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-240}"   # caps total output frames (truncates the condition before scheduling clips)
OUTPUT_FPS=source        # "source" -> encode outputs at the source video's fps (default); or set a number. Save-layer only.
REF_PAD_NUM=-1            # -1 = full anti-drift (SVI) padding; 0 = original VACE zero padding
SEED=123
I2V_RES=720               # 480 or 720; selects the base Wan I2V variant. Unused when MODEL_CHECKPOINT overrides the DiT.
NEGATIVE_PROMPT=""        # empty -> DEFAULT_NEGATIVE_PROMPT in pipeline.py

DEBUG=false               # true forces num_inference_steps=2 (smoke test)
SAVE_VERBOSE=true         # false -> only minimal output set

# Copy the checkpoint into /dev/shm (RAM tmpfs) before loading — fast load + shared across USP ranks.
# Needs FREE RAM >= checkpoint size on top of the model; set false if you hit CPU OOM (mmap is still used).
STAGE_CHECKPOINT_TO_SHM=true

SOURCE_VIDEO="${SAMPLE_DIR}/source.mp4"
STYLIZED_FIRST_FRAME="${SAMPLE_DIR}/stylized_first_frame.png"
PROMPT_FILE="${SAMPLE_DIR}/prompt.txt"

# NOTE: no preprocessing/orig_pixel.mp4 here — relighting conditions directly on the raw source.
for f in "$SOURCE_VIDEO" "$STYLIZED_FIRST_FRAME" "$PROMPT_FILE"; do
    [ -f "$f" ] || { echo "Missing required file: $f"; exit 1; }
done

# The finetuned idv2v.pth is fetched by scripts/download_checkpoints.sh into checkpoints/idv2v.pth.
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "Checkpoint not found: $MODEL_CHECKPOINT"
    echo "Run 'bash scripts/download_checkpoints.sh' to fetch idv2v.pth, or set MODEL_CHECKPOINT to a local .pth."
    exit 1
fi

# Optionally stage the checkpoint into RAM (see STAGE_CHECKPOINT_TO_SHM note above).
if [ "$STAGE_CHECKPOINT_TO_SHM" = "true" ] && [ -f "$MODEL_CHECKPOINT" ]; then
    case "$MODEL_CHECKPOINT" in
        /dev/shm/*) ;;   # already in RAM — nothing to do
        *)
            SHM_CKPT="/dev/shm/idv2v_ckpt_cache/$(basename "$MODEL_CHECKPOINT")"
            mkdir -p "$(dirname "$SHM_CKPT")"
            if [ ! -f "$SHM_CKPT" ] || [ "$(stat -c%s "$MODEL_CHECKPOINT")" != "$(stat -c%s "$SHM_CKPT" 2>/dev/null || echo 0)" ]; then
                echo "Staging checkpoint into RAM (/dev/shm): $MODEL_CHECKPOINT"
                cp -f "$MODEL_CHECKPOINT" "$SHM_CKPT"
            fi
            MODEL_CHECKPOINT="$SHM_CKPT" ;;
    esac
fi

# Auto-discover keyframes from <SAMPLE_DIR>/keyframes/<N>.png (0-based index N>=1; index 0 = stylized_first_frame.png).
# Relighting samples usually have none — this stays empty, which is fine.
KEY_FRAME_PATHS=""; KEY_FRAME_INDICES=""; NUM_KEYFRAMES=0
if [ -d "${SAMPLE_DIR}/keyframes" ]; then
    while IFS= read -r kf; do
        idx="${kf##*/}"; idx="${idx%.png}"
        KEY_FRAME_PATHS="${KEY_FRAME_PATHS:+$KEY_FRAME_PATHS,}$kf"
        KEY_FRAME_INDICES="${KEY_FRAME_INDICES:+$KEY_FRAME_INDICES,}$idx"
        NUM_KEYFRAMES=$((NUM_KEYFRAMES + 1))
    done < <(ls "${SAMPLE_DIR}/keyframes/"*.png 2>/dev/null | awk -F/ '{print $NF"\t"$0}' | sort -n | cut -f2-)
fi

if [ -z "$RUN_NAME" ]; then
    RUN_NAME="r${WIDTH}x${HEIGHT}_f${NUM_FRAMES_PER_CLIP}_kf${NUM_KEYFRAMES}_idv2v_relight"
fi
OUT_DIR="${SAMPLE_DIR}/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

NPROC=$(echo "$GPU" | awk -F',' '{print NF}')
USE_USP=false; [ "$NPROC" -gt 1 ] && USE_USP=true

MODEL_IDS="Wan-AI/Wan2.1-I2V-14B-${I2V_RES}P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
VACE_GLOB="${WAN_MODEL_DIR}/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors"
PIPELINE="$(python -c 'import idv2v.inference.pipeline as m; print(m.__file__)')"

CMD=()
if [ "$USE_USP" = "true" ]; then
    CMD+=(torchrun --nproc-per-node=$NPROC --master-port=$((29500 + RANDOM % 1000)))
else
    CMD+=(python)
fi
CMD+=("$PIPELINE"
    --prompt "$PROMPT_FILE"
    --input_image "$STYLIZED_FIRST_FRAME"
    --condition_0_path "$SOURCE_VIDEO"      # RELIGHTING: the raw source IS the condition (no SAM3, no gray)
    --original_video_path "$SOURCE_VIDEO"
    --result_save_folder "$OUT_DIR"
    --output_fps $OUTPUT_FPS
    --model_checkpoint "$MODEL_CHECKPOINT"
    --width $WIDTH --height $HEIGHT
    --num_frames_per_clip $NUM_FRAMES_PER_CLIP
    --max_num_frames $MAX_NUM_FRAMES
    --num_inference_steps $NUM_INFERENCE_STEPS
    --cfg_scale $CFG_SCALE
    --vace_scale $VACE_SCALE
    --ref_pad_num $REF_PAD_NUM
    --seed $SEED
    --local_model_path "$WAN_MODEL_DIR"
    --model_id_with_origin_paths "$MODEL_IDS"
    --vace_checkpoint_path "$VACE_GLOB"
)
[ -n "$KEY_FRAME_PATHS" ]   && CMD+=(--key_frame_paths "$KEY_FRAME_PATHS" --key_frame_indices "$KEY_FRAME_INDICES")
[ -n "$NEGATIVE_PROMPT" ]   && CMD+=(--negative_prompt "$NEGATIVE_PROMPT")
[ "$USE_USP" = "true" ]     && CMD+=(--use_usp)
[ "$DEBUG" = "true" ]       && CMD+=(--debug)
[ "$SAVE_VERBOSE" = "false" ] && CMD+=(--no-save_verbose)

echo "Mode:       RELIGHTING (condition_0 = raw source.mp4; no preprocessing)"
echo "Sample:     $SAMPLE_DIR"
echo "Run name:   $RUN_NAME"
echo "Output:     $OUT_DIR"
echo "GPU:        $GPU  (nproc=$NPROC, USP=$USE_USP)"
echo "Keyframes:  $NUM_KEYFRAMES  [$KEY_FRAME_INDICES]"
echo "Checkpoint: $MODEL_CHECKPOINT"
echo ""

CUDA_VISIBLE_DEVICES=$GPU "${CMD[@]}"

echo ""
echo "Done. Output: $OUT_DIR"
