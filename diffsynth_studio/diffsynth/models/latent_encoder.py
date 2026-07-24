from typing import Tuple

from einops import rearrange
import torch
from torch import nn

from .wan_video_dit import GateModule


def create_patch_embedding(in_channels, out_channels, patch_kernel_size, wan_patch_embedding=None):
    patch_embedding = nn.Conv3d(in_channels, out_channels, kernel_size=patch_kernel_size, stride=patch_kernel_size)
    if wan_patch_embedding is not None:
        target_out_channels, target_in_channels, kernel_1, kernel_2, kernel_3 = wan_patch_embedding.weight.shape
        target_patch_kernel_size = (kernel_1, kernel_2, kernel_3)
        assert out_channels == target_out_channels,\
            f"`out_channels` ({out_channels}) must match that of `wan_patch_embedding` ({target_out_channels})"
        assert in_channels >= target_in_channels,\
            f"`in_channels` ({in_channels}) must be at least that of `wan_patch_embedding` ({target_in_channels})"
        assert patch_kernel_size == target_patch_kernel_size, (
            f"`patch_kernel_size` ({patch_kernel_size}) must match that of "
            f"`wan_patch_embedding` ({target_patch_kernel_size})"
        )
        dtype = wan_patch_embedding.weight.dtype
        device = wan_patch_embedding.weight.device
        weight_additional = torch.zeros(
            out_channels, in_channels - target_in_channels, *target_patch_kernel_size, dtype=dtype, device=device,
        )
        weight = torch.cat([wan_patch_embedding.weight.data.clone(), weight_additional], dim=1)
        assert weight.shape == patch_embedding.weight.shape,\
            f"Shape of new weight {weight.shape} must match that of original weight {patch_embedding.weight.shape}"
        patch_embedding.weight.data = nn.Parameter(weight)
        patch_embedding.bias.data = nn.Parameter(wan_patch_embedding.bias.data.clone())
    else:
        nn.init.zeros_(patch_embedding.weight)
        nn.init.zeros_(patch_embedding.bias)
    return patch_embedding


def patchify(x, patch_embedding):
    x = patch_embedding(x)
    grid_size = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    return x, grid_size  # x, grid_size: (f, h, w)


class LatentEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_inputs: int = 2,
        hidden_channels: int = 1024,
        out_channels: int = 5120,
        patch_kernel_size: Tuple[int] = (1, 2, 2),
        wan_patch_embedding: nn.Module = None,
        target_reuse_patch_embedding: bool = False,
    ):
        super().__init__()
        assert wan_patch_embedding is not None

        self.patch_embedding = lambda x: wan_patch_embedding(x)
        self.target_reuse_patch_embedding = target_reuse_patch_embedding

        dtype = wan_patch_embedding.weight.dtype
        device = wan_patch_embedding.weight.device

        # Reference (masked source trajectory)
        self.reference_patch_embedding = create_patch_embedding(
            in_channels * num_inputs, out_channels, patch_kernel_size, wan_patch_embedding=wan_patch_embedding,
        )

        # Target (masked target trajectory)
        if target_reuse_patch_embedding:
            self.target_patch_embedding = create_patch_embedding(
                in_channels * num_inputs, out_channels, patch_kernel_size,
                wan_patch_embedding=wan_patch_embedding,
            )
            self.gate = GateModule()
            self.target_gate = nn.Parameter(torch.zeros(out_channels, 1, 1, 1, dtype=dtype, device=device))
        else:
            self.target_encoder = nn.Sequential(  # From EX-4D
                nn.Linear(in_channels * num_inputs, hidden_channels, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels, bias=True),
            )
            self.target_patch_embedding = nn.Conv3d(
                hidden_channels, out_channels, kernel_size=patch_kernel_size, stride=patch_kernel_size,
            )
            nn.init.zeros_(self.target_patch_embedding.weight)
            nn.init.zeros_(self.target_patch_embedding.bias)

    def forward(self, x, reference_video_latents, reference_mask_latents, target_video_latents, target_mask_latents):
        x, (f, h, w) = patchify(x, self.patch_embedding)

        masked_reference = None
        if reference_video_latents is not None and reference_mask_latents is not None:
            masked_reference = torch.cat([reference_video_latents, reference_mask_latents], dim=-4)
            masked_reference, (f_, h_, w_) = patchify(masked_reference, self.reference_patch_embedding)
            assert (f, h, w) == (f_, h_, w_)

        if target_video_latents is not None and target_mask_latents is not None:
            masked_target = torch.cat([target_video_latents, target_mask_latents], dim=-4)
            if self.target_reuse_patch_embedding:
                masked_target, _ = patchify(masked_target, self.target_patch_embedding)
                x = self.target_gate_apply(x, self.target_gate, masked_target)
            else:
                masked_target = masked_target.moveaxis(-4, -1)  # b c f h w -> b f h w c
                masked_target = self.target_encoder(masked_target)
                masked_target = masked_target.moveaxis(-1, -4)  # b f h w c -> b c f h w
                masked_target, (f_, h_, w_) = patchify(masked_target, self.target_patch_embedding)
                assert (f, h, w) == (f_, h_, w_)
                x = x + masked_target

        return x, masked_reference, (f, h, w)
