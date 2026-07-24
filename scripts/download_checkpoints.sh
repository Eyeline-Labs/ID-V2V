#!/bin/bash
# Download public ID-V2V dependencies into checkpoints/.
#
# DEFAULT (for the "idv2v" model — a SINGLE VACE condition, foreground-on-gray pixels): fetches the
# finetuned idv2v.pth (~72 GB) + SAM3 (~6.5 GB) + Wan2.1's T5 + VAE + tokenizer + CLIP (~17 GB) — everything
# needed to run the default model.
#
#   --with-depth : ALSO fetch the alternate "idv2v_with_normal_depth" 3-condition model: its
#                  idv2v_with_normal_depth.pth (~72 GB, same HF repo) + DAViD (surface-normals) +
#                  DepthAnything-V2 (depth). Adds ~75 GB.
#   --skip-idv2v : skip the finetuned idv2v.pth (e.g. you already have it; set MODEL_CHECKPOINT).
#
# Layout produced:
#   checkpoints/
#   ├── idv2v.pth                                      (finetuned default model, ~72 GB)
#   ├── sam3/                                          (gated: requires `hf auth login`)
#   ├── wan/Wan-AI/
#   │   ├── Wan2.1-T2V-14B/{models_t5..., Wan2.1_VAE.pth, google/}
#   │   └── Wan2.1-I2V-14B-480P/models_clip_open-clip-...vit-huge-14.pth
#   └── (only with --with-depth)
#       ├── david/multi-task-model-vitl16_384.onnx
#       └── depthv2/depth_anything_v2_vitl.pth
#
# Usage:  bash scripts/download_checkpoints.sh [--with-depth] [--skip-idv2v] [--skip-sam3] [--skip-wan]
set -e
cd "$(dirname "$0")/.."   # repo root
source .venv/bin/activate

CKPT=checkpoints
mkdir -p "$CKPT/wan"

WITH_DEPTH=0; SKIP_SAM3=0; SKIP_WAN=0; SKIP_IDV2V=0
IDV2V_REPO="${IDV2V_REPO:-Eyeline-Labs/ID-V2V}"   # Hugging Face repo for the finetuned idv2v.pth
for arg in "$@"; do
    case "$arg" in
        --with-depth) WITH_DEPTH=1 ;;
        --skip-idv2v) SKIP_IDV2V=1 ;;
        --skip-sam3)  SKIP_SAM3=1 ;;
        --skip-wan)   SKIP_WAN=1 ;;
        *) echo "Unknown flag: $arg"; echo "Usage: bash scripts/download_checkpoints.sh [--with-depth] [--skip-idv2v] [--skip-sam3] [--skip-wan]"; exit 2 ;;
    esac
done

# Step banners: Wan + SAM3 always run a step (download or "skipped"); idv2v adds one unless
# --skip-idv2v; --with-depth adds three (idv2v_with_normal_depth.pth + DAViD + DepthV2).
TOTAL=2
[ "$SKIP_IDV2V" = "0" ] && TOTAL=$((TOTAL + 1))
[ "$WITH_DEPTH"  = "1" ] && TOTAL=$((TOTAL + 3))
STEP=0

# ---------------------------------------------------------------------------
# Finetuned idv2v.pth (~72 GB) — the default model's weights. Public HF repo (no gate).
# ---------------------------------------------------------------------------
if [ "$SKIP_IDV2V" = "0" ]; then
    STEP=$((STEP + 1))
    if [ ! -f "$CKPT/idv2v.pth" ]; then
        echo "[$STEP/$TOTAL] Downloading finetuned idv2v.pth (~72 GB) from $IDV2V_REPO ..."
        hf download "$IDV2V_REPO" idv2v.pth --local-dir "$CKPT"
    else
        echo "[$STEP/$TOTAL] idv2v.pth already present."
    fi
fi

# ---------------------------------------------------------------------------
# DAViD + DepthAnything-V2 — fetched only with --with-depth (alternate idv2v_with_normal_depth model).
# ---------------------------------------------------------------------------
if [ "$WITH_DEPTH" = "1" ]; then
    mkdir -p "$CKPT/david" "$CKPT/depthv2"

    # idv2v_with_normal_depth.pth (~72 GB) — the alternate 3-condition model's weights (same HF repo, no gate).
    STEP=$((STEP + 1))
    if [ ! -f "$CKPT/idv2v_with_normal_depth.pth" ]; then
        echo "[$STEP/$TOTAL] Downloading idv2v_with_normal_depth.pth (~72 GB) from $IDV2V_REPO ..."
        hf download "$IDV2V_REPO" idv2v_with_normal_depth.pth --local-dir "$CKPT"
    else
        echo "[$STEP/$TOTAL] idv2v_with_normal_depth.pth already present."
    fi

    STEP=$((STEP + 1))
    if [ ! -f "$CKPT/david/multi-task-model-vitl16_384.onnx" ]; then
        echo "[$STEP/$TOTAL] Downloading DAViD multi-task ONNX (~1.4 GB)..."
        curl -L --fail --progress-bar \
            "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/multi-task-model-vitl16_384.onnx" \
            -o "$CKPT/david/multi-task-model-vitl16_384.onnx"
    else
        echo "[$STEP/$TOTAL] DAViD already present."
    fi

    STEP=$((STEP + 1))
    if [ ! -f "$CKPT/depthv2/depth_anything_v2_vitl.pth" ]; then
        echo "[$STEP/$TOTAL] Downloading DepthAnything-V2 ViT-L (~1.3 GB)..."
        hf download depth-anything/Depth-Anything-V2-Large depth_anything_v2_vitl.pth \
            --local-dir "$CKPT/depthv2"
    else
        echo "[$STEP/$TOTAL] DepthAnything-V2 already present."
    fi
fi

# ---------------------------------------------------------------------------
# Wan2.1's T5 + VAE + tokenizer + CLIP (~17 GB) — required by BOTH models.
# Skips the two large DiT/VACE .safetensors (~56 GB) — never read when MODEL_CHECKPOINT is set.
# ---------------------------------------------------------------------------
STEP=$((STEP + 1))
if [ "$SKIP_WAN" = "0" ]; then
    echo "[$STEP/$TOTAL] Downloading Wan 2.1 components (~17 GB)..."
    WAN_T2V="$CKPT/wan/Wan-AI/Wan2.1-T2V-14B"
    WAN_I2V_480="$CKPT/wan/Wan-AI/Wan2.1-I2V-14B-480P"
    mkdir -p "$WAN_T2V" "$WAN_I2V_480"

    # T5 + VAE + tokenizer from T2V-14B
    if [ ! -f "$WAN_T2V/models_t5_umt5-xxl-enc-bf16.pth" ]; then
        hf download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth --local-dir "$WAN_T2V"
    fi
    if [ ! -f "$WAN_T2V/Wan2.1_VAE.pth" ]; then
        hf download Wan-AI/Wan2.1-T2V-14B Wan2.1_VAE.pth --local-dir "$WAN_T2V"
    fi
    if [ ! -d "$WAN_T2V/google" ]; then
        hf download Wan-AI/Wan2.1-T2V-14B --include "google/*" --local-dir "$WAN_T2V"
    fi

    # CLIP from I2V-14B-480P
    if [ ! -f "$WAN_I2V_480/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
        hf download Wan-AI/Wan2.1-I2V-14B-480P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir "$WAN_I2V_480"
    fi
else
    echo "[$STEP/$TOTAL] Wan 2.1 components skipped."
fi

# ---------------------------------------------------------------------------
# SAM3 (~6.5 GB, HuggingFace GATED — requires `hf auth login` first). Required by BOTH models.
# ---------------------------------------------------------------------------
STEP=$((STEP + 1))
if [ "$SKIP_SAM3" = "0" ] && [ ! -f "$CKPT/sam3/config.json" ]; then
    echo "[$STEP/$TOTAL] Downloading SAM3 (~6.5 GB, gated repo)..."
    if ! hf auth whoami >/dev/null 2>&1; then
        cat <<'EOF'
SAM3 (facebook/sam3) is a gated HuggingFace repo. Before this step works:
  1. Visit https://huggingface.co/facebook/sam3 and request access (instant on most days).
  2. Run:  hf auth login   (paste a User Access Token with read scope).
Then re-run this script (or run it with --skip-sam3 to download everything else first).
EOF
        exit 1
    fi
    hf download facebook/sam3 --local-dir "$CKPT/sam3"
else
    echo "[$STEP/$TOTAL] SAM3 already present (or skipped)."
fi

echo ""
echo "Done. Checkpoint layout:"
echo "  $CKPT/idv2v.pth                                (finetuned idv2v model)"
echo "  $CKPT/sam3/                                    (SAM3 model dir)"
echo "  $CKPT/wan/Wan-AI/...                           (Wan 2.1 T5 + VAE + CLIP + tokenizer)"
if [ "$WITH_DEPTH" = "1" ]; then
    echo "  $CKPT/idv2v_with_normal_depth.pth              (idv2v_with_normal_depth model — 3-condition)"
    echo "  $CKPT/david/multi-task-model-vitl16_384.onnx   (DAViD ONNX — idv2v_with_normal_depth only)"
    echo "  $CKPT/depthv2/depth_anything_v2_vitl.pth       (DepthAnything-V2 — idv2v_with_normal_depth only)"
fi
echo ""
echo "Run inference:"
echo "  idv2v                                -> scripts/infer.sh   (uses checkpoints/idv2v.pth fetched above)"
echo "  alternate idv2v_with_normal_depth (3 cond)  -> scripts/idv2v_with_normal_depth/infer_with_depth.sh (uses checkpoints/idv2v_with_normal_depth.pth from --with-depth above)"
