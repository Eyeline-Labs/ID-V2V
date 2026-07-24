"""
Preprocessing package for IDv2v condition-video generation.

Modules build the per-frame conditioning signals fed to the model:
  - sam3.py         SAM3 segmentation + Secret Panda mask cleanup (per-frame masks/crops)
  - secret_panda.py binary-mask cleanup (hole-fill, morphological close, gap bridging)
  - orig_pixel.py   foreground-on-gray condition video from the SAM3 masks
  - david.py        DAViD surface-normal condition video composited on a gray canvas
  - depth.py        DepthAnything-V2 dense-depth condition video (idv2v_with_normal_depth model only)
"""
