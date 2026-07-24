# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_t2v_1_3B import t2v_1_3B
from .wan_t2v_14B import t2v_14B

WAN_CONFIGS = {
    'vace-1.3B': t2v_1_3B,
    'vace-14B': t2v_14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
    '720p': (1280, 720),
    '480p': (480, 832)
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '720p': 1280 * 720,
    '480p': 480 * 832
}

SUPPORTED_SIZES = {
    'vace-1.3B': ('480*832', '832*480', '480p'),
    'vace-14B': ('720*1280', '1280*720', '480*832', '832*480', '480p', '720p')
}
