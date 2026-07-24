# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from torch import nn

from diffusers.utils.torch_utils import maybe_allow_in_graph

from ltx_video.models.transformers.attention import BasicTransformerBlock


@maybe_allow_in_graph
class BasicTransformerMainBlock(BasicTransformerBlock):
    def __init__(self, *args, **kwargs):
        self.block_id = kwargs.pop('block_id')
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.FloatTensor:
        context_hints = kwargs.pop('context_hints')
        context_scale = kwargs.pop('context_scale')
        hidden_states = super().forward(*args, **kwargs)
        if self.block_id < len(context_hints) and context_hints[self.block_id] is not None:
            hidden_states = hidden_states + context_hints[self.block_id] * context_scale
        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBypassBlock(BasicTransformerBlock):
    def __init__(self, *args, **kwargs):
        self.dim = args[0]
        self.block_id = kwargs.pop('block_id')
        super().__init__(*args, **kwargs)
        if self.block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, *args, **kwargs):
        hidden_states = kwargs.pop('hidden_states')
        context_hidden_states = kwargs.pop('context_hidden_states')
        if self.block_id == 0:
            context_hidden_states = self.before_proj(context_hidden_states) + hidden_states

        kwargs['hidden_states'] = context_hidden_states
        bypass_context_hidden_states = super().forward(*args, **kwargs)
        main_context_hidden_states = self.after_proj(bypass_context_hidden_states)
        return (main_context_hidden_states, bypass_context_hidden_states)
