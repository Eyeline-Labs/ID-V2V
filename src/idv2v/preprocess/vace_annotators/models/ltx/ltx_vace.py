# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

from .models.transformers.transformer3d import VaceTransformer3DModel
from .pipelines.pipeline_ltx_video import VaceLTXVideoPipeline
from ..utils.preprocessor import VaceImageProcessor, VaceVideoProcessor



class LTXVace():
    def __init__(self, ckpt_path, text_encoder_path, precision='bfloat16', stg_skip_layers="19", stg_mode="stg_a", offload_to_cpu=False):
        self.precision = precision
        self.offload_to_cpu = offload_to_cpu
        ckpt_path = Path(ckpt_path)
        vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
        transformer = VaceTransformer3DModel.from_pretrained(ckpt_path)
        scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)

        text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, subfolder="text_encoder")
        patchifier = SymmetricPatchifier(patch_size=1)
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer")

        if torch.cuda.is_available():
            transformer = transformer.cuda()
            vae = vae.cuda()
            text_encoder = text_encoder.cuda()

        vae = vae.to(torch.bfloat16)
        if precision == "bfloat16" and transformer.dtype != torch.bfloat16:
            transformer = transformer.to(torch.bfloat16)
        text_encoder = text_encoder.to(torch.bfloat16)

        # Set spatiotemporal guidance
        self.skip_block_list = [int(x.strip()) for x in stg_skip_layers.split(",")]
        self.skip_layer_strategy = (
            SkipLayerStrategy.Attention
            if stg_mode.lower() == "stg_a"
            else SkipLayerStrategy.Residual
        )

        # Use submodels for the pipeline
        submodel_dict = {
            "transformer": transformer,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
        }

        self.pipeline = VaceLTXVideoPipeline(**submodel_dict)
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")

        self.img_proc = VaceImageProcessor(downsample=[8,32,32], seq_len=384)

        self.vid_proc = VaceVideoProcessor(downsample=[8,32,32],
                                           min_area=512*768,
                                           max_area=512*768,
                                           min_fps=25,
                                           max_fps=25,
                                           seq_len=4992,
                                           zero_start=True,
                                           keep_last=True)


    def generate(self, src_video=None, src_mask=None, src_ref_images=[], prompt="", negative_prompt="", seed=42,
                 num_inference_steps=40, num_images_per_prompt=1, context_scale=1.0, guidance_scale=3, stg_scale=1, stg_rescale=0.7,
                 frame_rate=25, image_cond_noise_scale=0.15, decode_timestep=0.05, decode_noise_scale=0.025,
                 output_height=512, output_width=768, num_frames=97):
        # src_video: [c, t, h, w] / norm [-1, 1]
        # src_mask : [c, t, h, w] / norm [0, 1]
        # src_ref_images : [[c, h, w], [c, h, w], ...] / norm [-1, 1]
        # image_size: (H, W)
        if (src_video is not None and src_video != "") and (src_mask is not None and src_mask != ""):
            src_video, src_mask, frame_ids, image_size, frame_rate = self.vid_proc.load_video_batch(src_video, src_mask)
            if torch.all(src_mask > 0):
                src_mask = torch.ones_like(src_video[:1, :, :, :])
            else:
                # bool_mask = src_mask > 0
                # bool_mask = bool_mask.expand_as(src_video)
                # src_video[bool_mask] = 0
                src_mask = src_mask[:1, :, :, :]
                src_mask = torch.clamp((src_mask + 1) / 2, min=0, max=1)
        elif (src_video is not None and src_video != "") and (src_mask is None or src_mask == ""):
            src_video, frame_ids, image_size, frame_rate = self.vid_proc.load_video_batch(src_video)
            src_mask = torch.ones_like(src_video[:1, :, :, :])
        else:
            output_height, output_width, frame_rate, num_frames = int(output_height), int(output_width), int(frame_rate), int(num_frames)
            frame_ids = list(range(num_frames))
            image_size = (output_height, output_width)
            src_video = torch.zeros((3, num_frames, output_height, output_width))
            src_mask = torch.ones((1, num_frames, output_height, output_width))

        src_ref_images_prelist = src_ref_images
        src_ref_images = []
        for ref_image in src_ref_images_prelist:
            if ref_image != "" and ref_image is not None:
                src_ref_images.append(self.img_proc.load_image(ref_image)[0])


        # Prepare input for the pipeline
        num_frames = len(frame_ids)
        sample = {
            "src_video": [src_video],
            "src_mask": [src_mask],
            "src_ref_images": [src_ref_images],
            "prompt": [prompt],
            "prompt_attention_mask": None,
            "negative_prompt": [negative_prompt],
            "negative_prompt_attention_mask": None,
        }

        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed)

        output = self.pipeline(
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            context_scale=context_scale,
            guidance_scale=guidance_scale,
            skip_layer_strategy=self.skip_layer_strategy,
            skip_block_list=self.skip_block_list,
            stg_scale=stg_scale,
            do_rescaling=stg_rescale != 1,
            rescaling_scale=stg_rescale,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=image_size[0],
            width=image_size[1],
            num_frames=num_frames,
            frame_rate=frame_rate,
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=ConditioningMethod.UNCONDITIONAL,
            image_cond_noise_scale=image_cond_noise_scale,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            mixed_precision=(self.precision in "mixed_precision"),
            offload_to_cpu=self.offload_to_cpu,
        )
        gen_video = output.images[0]
        gen_video = gen_video.to(torch.float32) if gen_video.dtype == torch.bfloat16 else gen_video
        info = output.info

        ret_data = {
            "out_video": gen_video,
            "src_video": src_video,
            "src_mask": src_mask,
            "src_ref_images": src_ref_images,
            "info": info
        }
        return ret_data