# Checkpoints

This directory is gitignored.

**The easiest path is `bash scripts/download_checkpoints.sh`** — it fetches the default deps (the finetuned `idv2v.pth` + SAM3 + Wan2.1's T5 + VAE + tokenizer + CLIP, ~96 GB) into the expected layout, and the default paths in `scripts/{preprocess,infer}.sh` already point here. SAM3 is gated on Hugging Face, so run `hf auth login` once first. Add `--with-depth` to also fetch the alternate `idv2v_with_normal_depth.pth` + DAViD + DepthAnything-V2.

You can also download manually. Per-file sources are listed below.

## Preprocessing weights

| Weight | Needed by | Download | Shell var |
|---|---|---|---|
| SAM3 HuggingFace model dir (gated; run `hf auth login` first) | **both** models | <https://huggingface.co/facebook/sam3> | `SAM3_CKPT` |
| DAViD multi-task ONNX (`multi-task-model-vitl16_384.onnx`) | `idv2v_with_normal_depth` only | <https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/multi-task-model-vitl16_384.onnx> | `DAVID_CKPT` |
| DepthAnything-V2 ViT-L (`depth_anything_v2_vitl.pth`) | `idv2v_with_normal_depth` only | <https://huggingface.co/depth-anything/Depth-Anything-V2-Large> | `DEPTHV2_CKPT` |

`SAM3_CKPT` is set in both `scripts/preprocess.sh` and `scripts/idv2v_with_normal_depth/preprocess_with_depth.sh`; `DAVID_CKPT` / `DEPTHV2_CKPT` are set only in `scripts/idv2v_with_normal_depth/preprocess_with_depth.sh`.

## ID-V2V checkpoints

There are **two** ID-V2V checkpoints, one per model variant. Each is a single `.pth` containing the finetuned diffusion-transformer (DiT) + VACE control weights, **required** for inference (it replaces the base Wan DiT and VACE-14B safetensors). The default **`idv2v.pth`** is downloaded by `scripts/download_checkpoints.sh` — you don't need to set anything:

| Checkpoint | VACE conditions | Use in | How it's obtained |
|---|---|---|---|
| **`idv2v.pth`** — *default* | 1 (foreground-on-gray pixels) | `scripts/infer.sh` | **downloaded by `download_checkpoints.sh`** from [`Eyeline-Labs/ID-V2V`](https://huggingface.co/Eyeline-Labs/ID-V2V) → `checkpoints/idv2v.pth` (override with `MODEL_CHECKPOINT`) |
| `idv2v_with_normal_depth.pth` — *alternate* | 3 (foreground-on-gray pixels + surface-normals + depth) | `scripts/idv2v_with_normal_depth/infer_with_depth.sh` | **downloaded by `download_checkpoints.sh --with-depth`** from [`Eyeline-Labs/ID-V2V`](https://huggingface.co/Eyeline-Labs/ID-V2V) → `checkpoints/idv2v_with_normal_depth.pth` |

> ⚠️ **The two `.pth` files are different, non-interchangeable weights.** They share the exact same architecture (DiT `in_dim=36`, VACE `vace_in_dim=96` — `vace_in_dim` is per-condition and independent of the number of conditions), so loading the wrong one **does not raise an error** — it just produces degraded/garbage output. Always pair `idv2v.pth` with `infer.sh` and `idv2v_with_normal_depth.pth` with `infer_with_depth.sh`.

**Wan2.1's T5 + VAE + tokenizer + CLIP** (required by both variants) — assign the parent directory to `WAN_MODEL_DIR`. Download each file from <https://huggingface.co/Wan-AI/>:

```
$WAN_MODEL_DIR/
└── Wan-AI/
    ├── Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth          # REQUIRED — T5 text encoder (~11 GB)
    ├── Wan2.1-T2V-14B/Wan2.1_VAE.pth                           # REQUIRED — VAE (~500 MB)
    ├── Wan2.1-T2V-14B/google/                                  # REQUIRED — tokenizer
    └── Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   # REQUIRED — CLIP image encoder (~5 GB)
```

> **What about the big `.safetensors` files under `Wan2.1-I2V-14B-720P/` and `Wan2.1-VACE-14B/` on HuggingFace?**
>
> Skip them. They are referenced in `--model_id_with_origin_paths` (default in the infer scripts) but **never read** at runtime when `MODEL_CHECKPOINT` is set, because the finetuned `.pth` fully replaces both:
> - `Wan2.1-I2V-14B-720P/diffusion_pytorch_model*.safetensors` (~28 GB) — the base DiT
> - `Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors` (~28 GB) — the base VACE control adapter
>
> Saves ~56 GB of download / disk. See `pipeline.py` for the runtime skip logic.

### Glossary
- **DiT** — diffusion transformer (the main video generation backbone)
- **VACE** — Wan's video-control adapter (consumes the VACE condition video(s))
- **T5** — text encoder for the prompt
- **VAE** — encodes/decodes between pixel and latent space
- **CLIP** — image encoder for the stylized first frame
