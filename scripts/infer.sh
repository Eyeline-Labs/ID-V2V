#!/bin/bash
# ID-V2V inference: clip-by-clip I2V + a VACE condition
# (foreground-on-gray pixels from preprocessing/orig_pixel.mp4).
#
# Reads:  <SAMPLE_DIR>/{source.mp4, stylized_first_frame.png, prompt.txt, preprocessing/orig_pixel.mp4, keyframes/}
# Writes: <SAMPLE_DIR>/outputs/<RUN_NAME>/
#
# Usage: edit the variables below (or override via env), then run from anywhere:
#     bash scripts/infer.sh
set -e
cd "$(dirname "$0")/.."           # repo root — so relative paths and .venv resolve
source .venv/bin/activate

# ====================================================================
# EDIT THESE 4 VARIABLES BEFORE RUNNING (or override any of the vars below via environment
# variable, e.g. `SAMPLE_DIR=... MAX_NUM_FRAMES=81 bash scripts/infer.sh`). See checkpoints/README.md.
# ====================================================================
SAMPLE_DIR="${SAMPLE_DIR:-test_samples/restylization/two_sitting_woman}"   # input dir (source.mp4 + stylized_first_frame.png + prompt.txt; run preprocess.sh first for orig_pixel.mp4)
# The finetuned idv2v .pth. Defaults to checkpoints/idv2v.pth, which scripts/download_checkpoints.sh
# fetches from Hugging Face; set MODEL_CHECKPOINT to use a local copy.
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-checkpoints/idv2v.pth}"     # from scripts/download_checkpoints.sh; override with a local path
WAN_MODEL_DIR="${WAN_MODEL_DIR:-checkpoints/wan}"                 # default: from scripts/download_checkpoints.sh. Override if Wan weights live elsewhere.
GPU="${GPU:-0,1,2,3,4,5,6,7}"                                     # comma-separated CUDA ids; >1 GPU auto-enables torchrun + USP sequence parallel

RUN_NAME=""   # empty -> auto-derived as r{W}x{H}_f{F}_kf{N}_idv2v

# Inference config (720p defaults)
WIDTH=1280
HEIGHT=720
NUM_FRAMES_PER_CLIP=81
NUM_INFERENCE_STEPS=30
CFG_SCALE=5
VACE_SCALE=1
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-240}"   # caps total output frames (truncates condition videos before scheduling clips)
OUTPUT_FPS=source        # "source" -> encode outputs at the source video's fps (default); or set a number (e.g. 16). Save-layer only.
REF_PAD_NUM=-1            # -1 = full anti-drift (SVI) padding; 0 = original VACE zero padding
SEED=123
I2V_RES=720               # 480 or 720; selects the base Wan I2V variant. Unused when MODEL_CHECKPOINT overrides the DiT.
NEGATIVE_PROMPT=""        # empty -> DEFAULT_NEGATIVE_PROMPT in pipeline.py (Wan 2.1's standard Chinese quality-degradation prompt)

DEBUG=false               # true forces num_inference_steps=2 (smoke test)
SAVE_VERBOSE=true         # false -> only minimal output set (run_config.json, first_frame.png, original_video.mp4, generated_video.mp4, flip_test.mp4)

# Copy the finetuned checkpoint into /dev/shm (a RAM-backed tmpfs) before loading. On slow or
# network-backed storage this makes loading dramatically faster, and on multi-GPU runs all
# ranks then read one shared RAM copy. COST: it needs FREE RAM >= the checkpoint size (tens of
# GB) ON TOP of the model's own memory, and the copy stays in /dev/shm/idv2v_ckpt_cache/ until
# reboot (rm it to reclaim RAM). If you hit CPU out-of-memory, set this to false — the pipeline
# still memory-maps the checkpoint, which is plenty fast from a normal local SSD.
STAGE_CHECKPOINT_TO_SHM=true

SOURCE_VIDEO="${SAMPLE_DIR}/source.mp4"
STYLIZED_FIRST_FRAME="${SAMPLE_DIR}/stylized_first_frame.png"
PROMPT_FILE="${SAMPLE_DIR}/prompt.txt"
PREPROC="${SAMPLE_DIR}/preprocessing"

for f in "$SOURCE_VIDEO" "$STYLIZED_FIRST_FRAME" "$PROMPT_FILE" "$PREPROC/orig_pixel.mp4"; do
    [ -f "$f" ] || { echo "Missing required file: $f (did you run scripts/preprocess.sh?)"; exit 1; }
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

# Auto-discover keyframes from <SAMPLE_DIR>/keyframes/<N>.png. <N> is the 0-based frame
# index where the keyframe is injected. Must be >= 1 (index 0 is reserved for stylized_first_frame.png).
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
    RUN_NAME="r${WIDTH}x${HEIGHT}_f${NUM_FRAMES_PER_CLIP}_kf${NUM_KEYFRAMES}_idv2v"
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
    --condition_0_path "$PREPROC/orig_pixel.mp4"
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
# (pipeline.py writes run_config.json into OUT_DIR itself.)

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
