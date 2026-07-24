import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from tqdm.auto import tqdm
from typing import List, Optional
from typing_extensions import Literal
from functools import partial

from .utils import crop_and_resize
from ..models import ModelManager, load_state_dict
from ..models.camera import get_recammaster_embedding, get_plucker_embedding, get_prope_dict
from ..models.latent_encoder import LatentEncoder
from ..models.wan_video_dit import WanModel, RMSNorm, ReferenceCrossAttention, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader


class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype are used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames

    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video

    def load_models_to_device(self, model_names=[]):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)

    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

    def enable_cpu_offload(self):
        warnings.warn("`enable_cpu_offload` will be deprecated. Please use `enable_vram_management`.")
        self.vram_management_enabled = True

    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)

    def freeze_except(self, module_keywords, do_freeze=True):
        for name, module in self.named_modules():
            if any([keyword in name for keyword in module_keywords]):
                module.train()
                module.requires_grad_(True)
            elif do_freeze:
                module.eval()
                module.requires_grad_(False)


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None

    def download_if_necessary(self, local_model_path="./models", skip_download=False, use_usp=False):
        if self.path is None:
            # Check model_id and origin_file_pattern
            if self.model_id is None:
                raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.""")

            # Skip if not in rank 0
            if use_usp:
                import torch.distributed as dist
                skip_download = dist.get_rank() != 0

            # Check whether the origin path is a folder
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                allow_file_pattern = None
                is_folder = True
            elif isinstance(self.origin_file_pattern, str) and self.origin_file_pattern.endswith("/"):
                allow_file_pattern = self.origin_file_pattern + "*"
                is_folder = True
            else:
                allow_file_pattern = self.origin_file_pattern
                is_folder = False

            # Download
            if not skip_download:
                downloaded_files = glob.glob(self.origin_file_pattern, root_dir=os.path.join(local_model_path, self.model_id))
                snapshot_download(
                    self.model_id,
                    local_dir=os.path.join(local_model_path, self.model_id),
                    allow_file_pattern=allow_file_pattern,
                    ignore_file_pattern=downloaded_files,
                    local_files_only=False
                )

            # Let rank 1, 2, ... wait for rank 0
            if use_usp:
                import torch.distributed as dist
                dist.barrier(device_ids=[dist.get_rank()])

            # Return downloaded files
            if is_folder:
                self.path = os.path.join(local_model_path, self.model_id, self.origin_file_pattern)
            else:
                self.path = glob.glob(os.path.join(local_model_path, self.model_id, self.origin_file_pattern))
            if isinstance(self.path, list) and len(self.path) == 1:
                self.path = self.path[0]


class WanVideoWorldRerenderPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_WorldRerenderVideoInput(),
            WanVideoUnit_CameraEmbedder(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video_recammaster
        self.model_size: str = None
        self.camera_encoding: str = None

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    def training_loss(self, **inputs):
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="mean")
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )

    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())

    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        # self.dit.forward = types.MethodType(usp_dit_forward, self.dit)  # usp_dit_forward is not used anywhere
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="google/*"),
        world_rerender_config: dict = None,
        local_model_path: str = "./models",
        checkpoint_path: str = None,
        skip_download: bool = False,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-14B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-14B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]

        # Initialize pipeline
        pipe = WanVideoWorldRerenderPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            pipe.initialize_usp()

        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(local_model_path, skip_download=skip_download, use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )

        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        pipe.dit = model_manager.fetch_model("wan_video_dit")
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary(local_model_path, skip_download=skip_download)
            pipe.prompter.fetch_models(pipe.text_encoder)
            pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        # Add World Rerender modules
        assert world_rerender_config is not None
        pipe.camera_encoding = world_rerender_config["camera_encoding"]
        dim = pipe.dit.blocks[0].self_attn.dim
        num_heads = pipe.dit.blocks[0].self_attn.num_heads
        vae_channels = pipe.dit.patch_embedding.weight.shape[1]  # in_channels of patch_embedding
        patch_kernel_size = tuple(pipe.dit.patch_embedding.weight.shape[2:])

        model_size = {1536: "1.3b", 5120: "14b"}[dim]
        pipe.model_size = model_size
        config_model = world_rerender_config[f"model_{model_size}"]

        pipe.dit.latent_encoder = LatentEncoder(
            in_channels=vae_channels,
            num_inputs=2,
            hidden_channels=config_model["latent_encoder_dim"],
            out_channels=dim,
            patch_kernel_size=patch_kernel_size,
            wan_patch_embedding=pipe.dit.patch_embedding,
            target_reuse_patch_embedding=world_rerender_config["target_reuse_patch_embedding"],
        )

        pipe.dit.reference_injection = world_rerender_config["reference_injection"]

        for i, block in enumerate(pipe.dit.blocks):

            # Camera encoder
            block.camera_encoding = world_rerender_config["camera_encoding"]
            if world_rerender_config["camera_encoding"] == "recammaster":
                block.cam_encoder = torch.nn.Linear(12, dim)
            elif world_rerender_config["camera_encoding"] == "plucker":
                block.cam_encoder = torch.nn.Linear(6, dim)
            elif world_rerender_config["camera_encoding"] in ("prope_nograd", "prope_grad"):
                block.cam_encoder = torch.nn.Linear(dim, dim)  # TODO: Maybe gating since this is expensive?
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()

            # Self-attention projector (after self-attention)
            block.projector = torch.nn.Linear(dim, dim)
            block.projector.weight = torch.nn.Parameter(torch.eye(dim))  # Initialize as identity
            block.projector.bias = torch.nn.Parameter(torch.zeros(dim))

            # Reference injection
            if pipe.dit.reference_injection == "self_attention":
                pass  # No need for additional
            elif (
                pipe.dit.reference_injection == "cross_attention" and i % config_model["cross_attn2_interval"] == 0
            ):
                block.norm4 = torch.nn.LayerNorm(dim)
                block.cross_attn2 = ReferenceCrossAttention(
                    dim=dim,
                    inner_dim=config_model["cross_attn2_dim"],
                    num_heads=config_model["cross_attn2_num_heads"],
                )
                torch.nn.init.zeros_(block.cross_attn2.o.weight)
                torch.nn.init.zeros_(block.cross_attn2.o.bias)

        # Load custom checkpoint (can include World Rerender params), TODO: Set strict=False?
        if checkpoint_path is not None:
            pipe.dit.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print(f"Loaded checkpoint from: {checkpoint_path}")

        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # World Rerender, videos and masks
        source_video: List[Image] = None,
        warped_target_video: List[Image] = None,
        source_color_mask: np.ndarray = None,
        source_geometry_mask: np.ndarray = None,
        source_motion_mask: np.ndarray = None,
        warped_target_color_mask: np.ndarray = None,
        warped_target_geometry_mask: np.ndarray = None,
        warped_target_motion_mask: np.ndarray = None,
        # World Rerender, cameras
        target_cam_c2w: Optional[np.ndarray] = None,  # f 4 4
        target_intrinsics: Optional[np.ndarray] = None,  # f 4
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape and frames
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: Optional[int] = 81,
        frame_interval: Optional[int] = 1,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):  # TODO: Make this work!
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            # World Rerender, videos and masks
            "source_video": source_video,
            "warped_target_video": warped_target_video,
            "source_color_mask": source_color_mask,
            "source_geometry_mask": source_geometry_mask,
            "source_motion_mask": source_motion_mask,
            "warped_target_color_mask": warped_target_color_mask,
            "warped_target_geometry_mask": warped_target_geometry_mask,
            "warped_target_motion_mask": warped_target_motion_mask,
            # World Rerender, cameras
            "cam_c2w": target_cam_c2w, "intrinsics": target_intrinsics,
            # Randomness
            "seed": seed, "rand_device": rand_device,
            # Shape and frames
            "batch_size": len(source_video),
            "height": height, "width": width,
            "num_frames": num_frames, "frame_interval": frame_interval,
            # Classifier-free guidance
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            # Scheduler
            "sigma_shift": sigma_shift,
            # VAE tiling
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            # Augmentation
            "drop_source": False, "drop_warped_target": False, "drop_prompt": False,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega =\
                self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] =\
                self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        # Decode
        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def process(self, pipe: WanVideoWorldRerenderPipeline, inputs: dict, positive=True, **kwargs) -> dict:
        raise NotImplementedError("`process` is not implemented.")


class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: WanVideoWorldRerenderPipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoWorldRerenderPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("batch_size", "height", "width", "num_frames", "seed", "rand_device"))

    def process(self, pipe: WanVideoWorldRerenderPipeline, batch_size, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        noise = pipe.generate_noise(
            (batch_size, 16, length, height // 8, width // 8), seed=seed, rand_device=rand_device,
        )
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoWorldRerenderPipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = torch.cat([pipe.preprocess_video(input_video_) for input_video_ in input_video], dim=0)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive", "drop_prompt": "drop_prompt"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive", "drop_prompt": "drop_prompt"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoWorldRerenderPipeline, prompt, positive, drop_prompt) -> dict:
        if prompt is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if drop_prompt:
            prompt = ""
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoWorldRerenderPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoWorldRerenderPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoWorldRerenderPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_CameraEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("cam_c2w", "intrinsics", "height", "width", "num_frames", "frame_interval"))

    def process(
        self, pipe: WanVideoWorldRerenderPipeline, cam_c2w, intrinsics, height, width, num_frames, frame_interval
    ):
        height_dit, width_dit = height // 8 // 2, width // 8 // 2  # VAE compresses x8, patchify compresses x2

        def check_num_frames_and_convert_torch(t, num_dims, name):
            if len(t.shape) == num_dims - 1:  # Missing batch dimension
                t = t[None]
            assert len(t.shape) == num_dims, f"`{name}` must have {num_dims} dimensions, got {len(t.shape)} dimensions."
            t = t[:, ::frame_interval][:num_frames]
            assert t.shape[1] >= num_frames, (
                f"`{name}` of length={t.shape[1]} (after frame interval skipping) "
                f"is less than the given num_frames={num_frames}."
            )
            t = torch.from_numpy(t).to(dtype=pipe.torch_dtype, device=pipe.device)
            return t

        cam_c2w = check_num_frames_and_convert_torch(cam_c2w, 4, "cam_c2w")

        # `intrinsics` should be b f 4 -> b f [ fx fy cx cy ]
        if pipe.camera_encoding in ("plucker", "prope_nograd", "prope_grad"):
            assert intrinsics is not None, "`intrinsics` should be defined with Plucker and PRoPE."
            intrinsics = check_num_frames_and_convert_torch(intrinsics, 3, "intrinsics")

        if pipe.camera_encoding == "recammaster":
            cam_emb = get_recammaster_embedding(cam_c2w, height_dit, width_dit)  # b f h w 12
            cam_emb = cam_emb[:, ::pipe.time_division_factor]
        elif pipe.camera_encoding == "plucker":
            cam_emb = get_plucker_embedding(
                intrinsics, cam_c2w, height, width, height_dit=height_dit, width_dit=width_dit,
            )  # b f h w 6
            cam_emb = cam_emb[:, ::pipe.time_division_factor]
        elif pipe.camera_encoding in ("prope_nograd", "prope_grad"):
            head_dim = pipe.dit.dim // pipe.dit.num_heads
            cam_emb = get_prope_dict(
                cam_c2w, intrinsics, height, width, height_dit, width_dit,
                time_division_factor=pipe.time_division_factor,
                precompute_coeffs=True, head_dim=head_dim, num_frames_multiplier=2,
            )

        return {"cam_emb": cam_emb}


class WanVideoUnit_WorldRerenderVideoInput(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "source_video", "warped_target_video",
                "source_color_mask", "source_geometry_mask", "source_motion_mask",
                "warped_target_color_mask", "warped_target_geometry_mask", "warped_target_motion_mask",
                "height", "width", "num_frames", "frame_interval",
                "drop_source", "drop_warped_target",
                "seed", "rand_device",
            ),
            onload_model_names=("vae",)
        )

    def encode_video(self, pipe, video_or_masks, height, width, num_frames, frame_interval):
        assert isinstance(video_or_masks, (list, tuple))

        if isinstance(video_or_masks[0][0], Image.Image):  # Video format (batched list of PIL.Image)
            video_or_masks = [video[::frame_interval][:num_frames] for video in video_or_masks]
            resample = "lanczos"

        elif isinstance(video_or_masks[0], np.ndarray):  # Mask format (list of np.ndarray)
            assert 0 < len(video_or_masks) <= 3, f"Given {len(video_or_masks)} masks, should be >0 and <=3"
            assert all([mask.dtype == np.bool_ for mask in video_or_masks]), "Not all masks have type np.bool_"
            video_or_masks = [mask.astype(np.uint8) * 255 for mask in video_or_masks]
            video_or_masks = video_or_masks + [np.zeros_like(video_or_masks[0])] * (3 - len(video_or_masks))
            video_or_masks = np.stack(video_or_masks, axis=-1)  # b f h w 3
            video_or_masks = [[Image.fromarray(frame) for frame in mask] for mask in video_or_masks]
            resample = "bilinear"

        assert len(video_or_masks[0]) >= num_frames, (
            f"Video or masks of length={len(video_or_masks)} (after frame interval skipping) "
            f"is less than the given num_frames={num_frames}."
        )

        video_or_masks = [crop_and_resize(video, height, width, resample=resample) for video in video_or_masks]
        video_or_masks = [pipe.preprocess_video(video) for video in video_or_masks]
        latents = torch.cat([pipe.vae.encode(video, device=pipe.device) for video in video_or_masks], dim=0)
        return latents

    def drop_video_as_noise(self, pipe, tensor, drop):
        batch_size, num_channels_vae, num_frames_vae, height_vae, width_vae = tensor.shape
        tensor = tensor.detach().clone()
        assert batch_size == len(drop)
        for i in range(batch_size):
            if drop[i]:
                tensor[i] = pipe.generate_noise(
                    (num_channels_vae, num_frames_vae, height_vae, width_vae), seed=seed, rand_device=rand_device,
                )
        return tensor

    def drop_masks(self, masks, drop):
        masks_dropped = []
        for mask in masks:
            mask = mask.copy()
            for i in range(len(drop)):
                if drop[i]:
                    mask[i] = False
            masks_dropped.append(mask)
        return masks_dropped

    def process(
        self,
        pipe: WanVideoWorldRerenderPipeline,
        source_video: List[Image],
        warped_target_video: List[Image],
        source_color_mask: np.ndarray,
        source_geometry_mask: np.ndarray,
        source_motion_mask: np.ndarray,
        warped_target_color_mask: np.ndarray,
        warped_target_geometry_mask: np.ndarray,
        warped_target_motion_mask: np.ndarray,
        height: int,
        width: int,
        num_frames: int,
        frame_interval: int,
        drop_source: List[bool],
        drop_warped_target: List[bool],
        seed: int,
        rand_device: torch.device,
    ):
        pipe.load_models_to_device(["vae"])

        encode_video = partial(
            self.encode_video, height=height, width=width, num_frames=num_frames, frame_interval=frame_interval,
        )
        source_latents = encode_video(pipe, source_video)
        warped_target_latents = encode_video(pipe, warped_target_video)
        source_latents = self.drop_video_as_noise(pipe, source_latents, drop_source)
        warped_target_latents = self.drop_video_as_noise(pipe, warped_target_latents, drop_warped_target)

        source_masks = self.drop_masks([source_color_mask, source_geometry_mask, source_motion_mask], drop_source)
        warped_target_masks = self.drop_masks(
            [warped_target_color_mask, warped_target_geometry_mask, warped_target_motion_mask], drop_warped_target,
        )
        source_mask_latents = encode_video(pipe, source_masks)
        warped_target_mask_latents = encode_video(pipe, warped_target_masks)

        return {
            "source_latents": source_latents,
            "warped_target_latents": warped_target_latents,
            "source_mask_latents": source_mask_latents,
            "warped_target_mask_latents": warped_target_mask_latents,
        }


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask

    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def model_fn_wan_video_recammaster(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    use_unified_sequence_parallel: bool = False,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    tea_cache: TeaCache = None,
    # World Rerender
    source_latents: torch.Tensor = None,
    source_mask_latents: torch.Tensor = None,
    warped_target_latents: torch.Tensor = None,
    warped_target_mask_latents: torch.Tensor = None,
    cam_emb: torch.Tensor = None,
    **kwargs,
):
    for arg in (source_latents, source_mask_latents, warped_target_latents, warped_target_mask_latents, cam_emb):
        assert arg is not None

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group
        )

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    assert not dit.has_image_input, "dit.has_image_input=True is untested!"

    x, masked_source_latents, (f, h, w) =\
        dit.latent_encoder(x, source_latents, source_mask_latents, warped_target_latents, warped_target_mask_latents)

    # Reference video (reference_injection=`self_attention`), assuming no temporal patchify
    if masked_source_latents is not None and dit.reference_injection == "self_attention":
        x = torch.concat((x, masked_source_latents), dim=1)
        f *= 2

    # World Rerender: Camera embedding
    if cam_emb is not None:
        # b t h w d -> b 2t h w d -> b (2t h w) d, because World Rerender concats source and target videos
        # TODO: What about input cameras? Should we worry about that???
        if isinstance(cam_emb, torch.Tensor):
            if dit.reference_injection == "self_attention":
                cam_emb = cam_emb.repeat(1, 2, 1, 1, 1)
            cam_emb = rearrange(cam_emb, "b f h w d -> b (f h w) d")
            assert cam_emb.shape[1] == x.shape[1],\
                f"Camera sequence length ({cam_emb.shape[1]}) does not match that of the input latent ({x.shape[1]})"
        elif isinstance(cam_emb, dict):
            if cam_emb["viewmats"].shape[1] != f:  # Not yet duplicated
                cam_emb["viewmats"] = cam_emb["viewmats"].repeat(1, 2, 1, 1)
                cam_emb["Ks"] = cam_emb["Ks"].repeat(1, 2, 1, 1)
            assert cam_emb["viewmats"].shape[1] == f, (
                f"Camera number of frames ({cam_emb['viewmats'].shape[1]}) "
                f"does not match that of the input latent ({f})"
            )

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    # Blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            cam_emb = torch.chunk(cam_emb, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        block_args = [x, context, t_mod, freqs, cam_emb]
        if dit.reference_injection == "cross_attention":
            block_args.append(masked_source_latents)

        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block), *block_args, use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), *block_args, use_reentrant=False,
                )
            else:
                x = block(*block_args)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    # Remove reference video latents
    if masked_source_latents is not None and dit.reference_injection == "self_attention":
        x = x[:, :masked_source_latents.shape[1]]
        f //= 2
    x = dit.unpatchify(x, (f, h, w))
    return x
