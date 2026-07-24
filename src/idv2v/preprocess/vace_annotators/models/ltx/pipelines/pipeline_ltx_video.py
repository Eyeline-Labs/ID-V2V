# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import html
import inspect
from dataclasses import dataclass

import math
import re
import urllib.parse as ul
import numpy as np
import PIL
from typing import Callable, Dict, List, Optional, Tuple, Union


import torch
from torch import Tensor
import torch.nn.functional as F
from contextlib import nullcontext
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput, BaseOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import Patchifier
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    vae_decode,
    vae_encode,
)
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.schedulers.rf import TimestepShifter
from ltx_video.utils.conditioning_method import ConditioningMethod
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline, retrieve_timesteps

from ...utils.preprocessor import prepare_source



@dataclass
class ImagePipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    info: Optional[Dict] = None


class VaceLTXVideoPipeline(LTXVideoPipeline):

    @torch.no_grad()
    def __call__(
            self,
            src_video: torch.FloatTensor = None,
            src_mask: torch.FloatTensor = None,
            src_ref_images: List[torch.FloatTensor] = None,
            height: int = 512,
            width: int = 768,
            num_frames: int = 97,
            frame_rate: float = 25,
            prompt: Union[str, List[str]] = None,
            negative_prompt: str = "",
            num_inference_steps: int = 20,
            timesteps: List[int] = None,
            guidance_scale: float = 4.5,
            context_scale: float = 1.0,
            skip_layer_strategy: Optional[SkipLayerStrategy] = None,
            skip_block_list: List[int] = None,
            stg_scale: float = 1.0,
            do_rescaling: bool = True,
            rescaling_scale: float = 0.7,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            prompt_attention_mask: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            clean_caption: bool = True,
            media_items: Optional[torch.FloatTensor] = None,
            decode_timestep: Union[List[float], float] = 0.0,
            decode_noise_scale: Optional[List[float]] = None,
            mixed_precision: bool = False,
            offload_to_cpu: bool = False,
            decouple_with_mask: bool = True,
            use_mask: bool = True,
            decode_all_frames: bool = False,
            mask_downsample: [list] = [2, 8, 8],
            **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. This negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        is_video = kwargs.get("is_video", False)
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        do_spatio_temporal_guidance = stg_scale > 0.0

        num_conds = 1
        if do_classifier_free_guidance:
            num_conds += 1
        if do_spatio_temporal_guidance:
            num_conds += 1

        skip_layer_mask = None
        if do_spatio_temporal_guidance:
            skip_layer_mask = self.transformer.create_skip_layer_mask(
                skip_block_list, batch_size, num_conds, 2
            )

        # 3. Encode input prompt
        self.text_encoder = self.text_encoder.to(self._execution_device)

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
        )

        if offload_to_cpu:
            self.text_encoder = self.text_encoder.cpu()

        self.transformer = self.transformer.to(self._execution_device)

        prompt_embeds_batch = prompt_embeds
        prompt_attention_mask_batch = prompt_attention_mask
        if do_classifier_free_guidance:
            prompt_embeds_batch = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            prompt_attention_mask_batch = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )
        if do_spatio_temporal_guidance:
            prompt_embeds_batch = torch.cat([prompt_embeds_batch, prompt_embeds], dim=0)
            prompt_attention_mask_batch = torch.cat(
                [
                    prompt_attention_mask_batch,
                    prompt_attention_mask,
                ],
                dim=0,
            )

        # 3b. Encode and prepare conditioning data
        self.video_scale_factor = self.video_scale_factor if is_video else 1
        conditioning_method = kwargs.get("conditioning_method", None)
        vae_per_channel_normalize = kwargs.get("vae_per_channel_normalize", False)
        image_cond_noise_scale = kwargs.get("image_cond_noise_scale", 0.0)
        init_latents, conditioning_mask = self.prepare_conditioning(
            media_items,
            num_frames,
            height,
            width,
            conditioning_method,
            vae_per_channel_normalize,
        )

        #------------------------ VACE Part ------------------------#
        # 4. Prepare latents.
        image_size = (height, width)
        src_ref_images = [None] * batch_size if src_ref_images is None else src_ref_images
        source_ref_len = max([len(ref_imgs) if ref_imgs is not None else 0 for ref_imgs in src_ref_images])
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_num_frames = num_frames // self.video_scale_factor
        if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
            latent_num_frames += 1
        latent_frame_rate = frame_rate / self.video_scale_factor
        num_latent_patches = latent_height * latent_width * (latent_num_frames + source_ref_len)
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latent_channels=self.transformer.config.in_channels,
            num_patches=num_latent_patches,
            dtype=prompt_embeds_batch.dtype,
            device=device,
            generator=generator,
            latents=init_latents,
            latents_mask=conditioning_mask,
        )
        src_video, src_mask, src_ref_images = prepare_source(src_video, src_mask, src_ref_images, num_frames, image_size, latents.device)

        # Prepare source_latents
        if decouple_with_mask:
            unchanged = [i * (1 - m) + 0 * m for i, m in zip(src_video, src_mask)]
            changed = [i * m + 0 * (1 - m) for i, m in zip(src_video, src_mask)]
            unchanged = torch.stack(unchanged, dim=0).to(self.vae.dtype).to(device) if isinstance(unchanged, list) else unchanged.to(self.vae.dtype).to(device)  # [B, C, F, H, W]
            changed = torch.stack(changed, dim=0).to(self.vae.dtype).to(device) if isinstance(changed, list) else changed.to(self.vae.dtype).to(device)  # [B, C, F, H, W]
            unchanged_latents = vae_encode(unchanged, vae=self.vae, vae_per_channel_normalize=vae_per_channel_normalize).float()
            changed_latents = vae_encode(changed, vae=self.vae, vae_per_channel_normalize=vae_per_channel_normalize).float()
            source_latents = torch.stack([torch.cat((u, c), dim=0) for u, c in zip(unchanged_latents, changed_latents)], dim=0)
        else:
            src_video = torch.stack(src_video, dim=0).to(self.vae.dtype).to(device) if isinstance(src_video, list) else src_video.to(self.vae.dtype).to(device)  # [B, C, F, H, W]
            source_latents = vae_encode(src_video, vae=self.vae, vae_per_channel_normalize=vae_per_channel_normalize).float()

        # Prepare source_ref_latents
        use_ref = all(ref_imgs is not None and len(ref_imgs) > 0 for ref_imgs in src_ref_images)
        if use_ref:
            source_ref_latents = []
            for i, ref_imgs in enumerate(src_ref_images):
                # [(C=3, F=1, H, W), ...]  ->  (N_REF, C'=128, F=1, H', W')
                ref_imgs = torch.stack(ref_imgs, dim=0).to(self.vae.dtype).to(device) if isinstance(ref_imgs, list) else ref_imgs.to(self.vae.dtype).to(device)  # [B, C, F, H, W]
                ref_latents = vae_encode(ref_imgs, vae=self.vae, vae_per_channel_normalize=vae_per_channel_normalize).float()
                # (N_REF, C'=128, F=1, H', W') -> (1, C'=128, N_REF, H', W')
                ref_latents = ref_latents.permute(2, 1, 0, 3, 4)
                if decouple_with_mask:
                    ref_latents = torch.cat([ref_latents, torch.zeros_like(ref_latents)], dim=1)  # [unchanged, changed]
                source_ref_latents.append(ref_latents)
            # (B, C'=128, N_REF, H', W')
            source_ref_latents = torch.cat(source_ref_latents, dim=0)
        else:
            source_ref_latents = None

        # Prepare source_latents
        if source_ref_latents is not None:
            source_latents = torch.cat([source_ref_latents, source_latents], dim=2)
        source_latents = self.patchifier.patchify(latents=source_latents).to(self.transformer.dtype).to(device)

        # Prepare source_mask_latents
        if use_mask and src_mask is not None:
            source_mask_latents = []
            for submask in src_mask:
                submask = F.interpolate(submask.unsqueeze(0),
                                        size=(latent_num_frames * mask_downsample[0],
                                              latent_height * mask_downsample[1],
                                              latent_width * mask_downsample[2]),
                                        mode='trilinear', align_corners=True)
                submask = rearrange(submask, "b c (f p1) (h p2) (w p3) -> b (c p1 p2 p3) f h w", p1=mask_downsample[0], p2=mask_downsample[1], p3=mask_downsample[2]).to(device)
                if source_ref_latents is not None:
                    if decouple_with_mask:
                        submask = torch.cat([torch.zeros_like(source_ref_latents[:, :latents.shape[-1], :]), submask], dim=2)
                    else:
                        submask = torch.cat([torch.zeros_like(source_ref_latents), submask], dim=2)
                submask = self.patchifier.patchify(submask)
                source_mask_latents.append(submask)
            source_mask_latents = torch.cat(source_mask_latents, dim=0).to(self.transformer.dtype).to(device)
        else:
            source_mask_latents = None
        #------------------------ VACE Part ------------------------#

        orig_conditiong_mask = conditioning_mask
        if conditioning_mask is not None and is_video:
            assert num_images_per_prompt == 1
            conditioning_mask = (
                torch.cat([conditioning_mask] * num_conds)
                if num_conds > 1
                else conditioning_mask
            )

        # 5. Prepare timesteps
        retrieve_timesteps_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_timesteps_kwargs["samples"] = latents
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            **retrieve_timesteps_kwargs,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if conditioning_method == ConditioningMethod.FIRST_FRAME:
                    latents = self.image_cond_noise_update(
                        t,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        orig_conditiong_mask,
                        generator,
                    )

                latent_model_input = (
                    torch.cat([latents] * num_conds) if num_conds > 1 else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                latent_frame_rates = (
                        torch.ones(
                            latent_model_input.shape[0], 1, device=latent_model_input.device
                        )
                        * latent_frame_rate
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).unsqueeze(-1)
                scale_grid = (
                    (
                        1 / latent_frame_rates,
                        self.vae_scale_factor,
                        self.vae_scale_factor,
                    )
                    if self.transformer.use_rope
                    else None
                )
                indices_grid = self.patchifier.get_grid(
                    orig_num_frames=latent_num_frames + source_ref_len,
                    orig_height=latent_height,
                    orig_width=latent_width,
                    batch_size=latent_model_input.shape[0],
                    scale_grid=scale_grid,
                    device=latents.device,
                )

                if conditioning_mask is not None:
                    current_timestep = current_timestep * (1 - conditioning_mask)
                # Choose the appropriate context manager based on `mixed_precision`
                if mixed_precision:
                    if "xla" in device.type:
                        raise NotImplementedError(
                            "Mixed precision is not supported yet on XLA devices."
                        )

                    context_manager = torch.autocast(device.type, dtype=torch.bfloat16)
                else:
                    context_manager = nullcontext()  # Dummy context manager

                # predict noise model_output
                with context_manager:
                    noise_pred = self.transformer(
                        latent_model_input.to(self.transformer.dtype),
                        indices_grid,
                        source_latents=source_latents,
                        source_mask_latents=source_mask_latents if use_mask else None,
                        context_scale=context_scale,
                        encoder_hidden_states=prompt_embeds_batch.to(
                            self.transformer.dtype
                        ),
                        encoder_attention_mask=prompt_attention_mask_batch,
                        timestep=current_timestep,
                        skip_layer_mask=skip_layer_mask,
                        skip_layer_strategy=skip_layer_strategy,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_spatio_temporal_guidance:
                    noise_pred_text_perturb = noise_pred[-1:]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[:2].chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )
                if do_spatio_temporal_guidance:
                    noise_pred = noise_pred + stg_scale * (
                            noise_pred_text - noise_pred_text_perturb
                    )
                    if do_rescaling:
                        factor = noise_pred_text.std() / noise_pred.std()
                        factor = rescaling_scale * factor + (1 - rescaling_scale)
                        noise_pred = noise_pred * factor

                current_timestep = current_timestep[:1]
                # learned sigma
                if (
                        self.transformer.config.out_channels // 2
                        == self.transformer.config.in_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    t if current_timestep is None else current_timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, t, {})

        if offload_to_cpu:
            self.transformer = self.transformer.cpu()
            if self._execution_device == "cuda":
                torch.cuda.empty_cache()

        latents = self.patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            output_num_frames=latent_num_frames + source_ref_len,
            out_channels=self.transformer.config.in_channels
                         // math.prod(self.patchifier.patch_size),
        )

        if not decode_all_frames:
            latents = latents[:, :, source_ref_len:]

        if output_type != "latent":
            if self.vae.decoder.timestep_conditioning:
                noise = torch.randn_like(latents)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * latents.shape[0]
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * latents.shape[0]

                decode_timestep = torch.tensor(decode_timestep).to(latents.device)
                decode_noise_scale = torch.tensor(decode_noise_scale).to(
                    latents.device
                )[:, None, None, None, None]
                latents = (
                        latents * (1 - decode_noise_scale) + noise * decode_noise_scale
                )
            else:
                decode_timestep = None
            image = vae_decode(
                latents,
                self.vae,
                is_video,
                vae_per_channel_normalize=kwargs["vae_per_channel_normalize"],
                timestep=decode_timestep,
            )
            # image = self.image_processor.postprocess(image, output_type=output_type)

        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        info = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate
        }

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image, info=info)

