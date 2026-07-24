# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import random
import time

import torch
import numpy as np

from models.ltx.ltx_vace import LTXVace
from annotators.utils import save_one_video, save_one_image, get_annotator

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total_memory
    return None


def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Load models from separate directories and run the pipeline."
    )

    # Directories
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='models/VACE-LTX-Video-0.9/ltx-video-2b-v0.9.safetensors',
        help="Path to a safetensors file that contains all model parts.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default='models/VACE-LTX-Video-0.9',
        help="Path to a safetensors file that contains all model parts.",
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to the folder to save output video, if None will save in results/ directory.",
    )
    parser.add_argument("--seed", type=int, default="42")

    # Pipeline parameters
    parser.add_argument(
        "--num_inference_steps", type=int, default=40, help="Number of inference steps"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt",
    )
    parser.add_argument(
        "--context_scale",
        type=float,
        default=1.0,
        help="Context scale for the pipeline",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3,
        help="Guidance scale for the pipeline",
    )
    parser.add_argument(
        "--stg_scale",
        type=float,
        default=1,
        help="Spatiotemporal guidance scale for the pipeline. 0 to disable STG.",
    )
    parser.add_argument(
        "--stg_rescale",
        type=float,
        default=0.7,
        help="Spatiotemporal guidance rescaling scale for the pipeline. 1 to disable rescale.",
    )
    parser.add_argument(
        "--stg_mode",
        type=str,
        default="stg_a",
        help="Spatiotemporal guidance mode for the pipeline. Can be either stg_a or stg_r.",
    )
    parser.add_argument(
        "--stg_skip_layers",
        type=str,
        default="19",
        help="Attention layers to skip for spatiotemporal guidance. Comma separated list of integers.",
    )
    parser.add_argument(
        "--image_cond_noise_scale",
        type=float,
        default=0.15,
        help="Amount of noise to add to the conditioned image",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="The height of the output video only if src_video is empty.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="The width of the output video only if src_video is empty.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=97,
        help="The frames of the output video only if src_video is empty.",
    )
    parser.add_argument(
        "--frame_rate", type=int, default=25, help="Frame rate for the output video"
    )

    parser.add_argument(
        "--precision",
        choices=["bfloat16", "mixed_precision"],
        default="bfloat16",
        help="Sets the precision for the transformer and tokenizer. Default is bfloat16. If 'mixed_precision' is enabled, it moves to mixed-precision.",
    )

    # VAE noise augmentation
    parser.add_argument(
        "--decode_timestep",
        type=float,
        default=0.05,
        help="Timestep for decoding noise",
    )
    parser.add_argument(
        "--decode_noise_scale",
        type=float,
        default=0.025,
        help="Noise level for decoding noise",
    )

    # Prompts
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to guide generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt for undesired features",
    )

    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="Offloading unnecessary computations to CPU.",
    )
    parser.add_argument(
        "--use_prompt_extend",
        default='plain',
        choices=['plain', 'ltx_en', 'ltx_en_ds'],
        help="Whether to use prompt extend."
    )
    return parser

def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args

    print(f"Running generation with arguments: {args}")

    seed_everething(args.seed)

    offload_to_cpu = False if not args.offload_to_cpu else get_total_gpu_memory() < 30

    assert os.path.exists(args.ckpt_path) and os.path.exists(args.text_encoder_path)

    ltx_vace = LTXVace(ckpt_path=args.ckpt_path,
                       text_encoder_path=args.text_encoder_path,
                       precision=args.precision,
                       stg_skip_layers=args.stg_skip_layers,
                       stg_mode=args.stg_mode,
                       offload_to_cpu=offload_to_cpu)

    src_ref_images = args.src_ref_images.split(',') if args.src_ref_images is not None else []
    if args.use_prompt_extend and args.use_prompt_extend != 'plain':
        prompt = get_annotator(config_type='prompt', config_task=args.use_prompt_extend, return_dict=False).forward(args.prompt)
        print(f"Prompt extended from '{args.prompt}' to '{prompt}'")
    else:
        prompt = args.prompt

    output = ltx_vace.generate(src_video=args.src_video,
                               src_mask=args.src_mask,
                               src_ref_images=src_ref_images,
                               prompt=prompt,
                               negative_prompt=args.negative_prompt,
                               seed=args.seed,
                               num_inference_steps=args.num_inference_steps,
                               num_images_per_prompt=args.num_images_per_prompt,
                               context_scale=args.context_scale,
                               guidance_scale=args.guidance_scale,
                               stg_scale=args.stg_scale,
                               stg_rescale=args.stg_rescale,
                               frame_rate=args.frame_rate,
                               image_cond_noise_scale=args.image_cond_noise_scale,
                               decode_timestep=args.decode_timestep,
                               decode_noise_scale=args.decode_noise_scale,
                               output_height=args.height,
                               output_width=args.width,
                               num_frames=args.num_frames)


    if args.save_dir is None:
        save_dir = os.path.join('results', 'vace_ltxv', time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frame_rate = output['info']['frame_rate']

    ret_data = {}
    if output['out_video'] is not None:
        save_path = os.path.join(save_dir, 'out_video.mp4')
        out_video = (torch.clamp(output['out_video'] / 2 + 0.5, min=0.0, max=1.0).permute(1, 2, 3, 0) * 255).cpu().numpy().astype(np.uint8)
        save_one_video(save_path, out_video, fps=frame_rate)
        print(f"Save out_video to {save_path}")
        ret_data['out_video'] = save_path
    if output['src_video'] is not None:
        save_path = os.path.join(save_dir, 'src_video.mp4')
        src_video = (torch.clamp(output['src_video'] / 2 + 0.5, min=0.0, max=1.0).permute(1, 2, 3, 0) * 255).cpu().numpy().astype(np.uint8)
        save_one_video(save_path, src_video, fps=frame_rate)
        print(f"Save src_video to {save_path}")
        ret_data['src_video'] = save_path
    if output['src_mask'] is not None:
        save_path = os.path.join(save_dir, 'src_mask.mp4')
        src_mask = (torch.clamp(output['src_mask'], min=0.0, max=1.0).permute(1, 2, 3, 0) * 255).cpu().numpy().astype(np.uint8)
        save_one_video(save_path, src_mask, fps=frame_rate)
        print(f"Save src_mask to {save_path}")
        ret_data['src_mask'] = save_path
    if output['src_ref_images'] is not None:
        for i, ref_img in enumerate(output['src_ref_images']):  # [C, F=1, H, W]
            save_path = os.path.join(save_dir, f'src_ref_image_{i}.png')
            ref_img = (torch.clamp(ref_img.squeeze(1), min=0.0, max=1.0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
            save_one_image(save_path, ref_img, use_type='pil')
            print(f"Save src_ref_image_{i} to {save_path}")
            ret_data[f'src_ref_image_{i}'] = save_path
    return ret_data


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)