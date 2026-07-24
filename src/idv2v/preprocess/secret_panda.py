"""
Secret Panda - Binary mask cleanup algorithm.
Standalone, copy-paste ready.
"""

import numpy as np
from scipy import ndimage


def secret_panda(mask: np.ndarray,
                 fill_holes_first: bool = True,
                 close_kernel: int = 10,
                 bridge_distance: int = 15) -> np.ndarray:
    """
    Clean up a binary mask by closing rivers/gaps between foreground
    regions and filling internal holes.

    Args:
        mask: Binary mask (0/1 or bool)
        fill_holes_first: Fill internal holes before the morphological close
        close_kernel: Square kernel size for the morphological close
        bridge_distance: Square kernel size for bridging wider gaps between regions

    Returns:
        Cleaned binary mask (same dtype as input)
    """
    result = mask.copy()

    # Step 1: fill internal holes
    if fill_holes_first:
        result = ndimage.binary_fill_holes(result).astype(result.dtype)

    # Step 2: morphological close (dilate then erode) to seal small gaps
    if close_kernel > 0:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        dilated = ndimage.binary_dilation(result, structure=kernel, iterations=1)
        result = ndimage.binary_erosion(dilated, structure=kernel, iterations=1).astype(result.dtype)

    # Step 3: same close with a larger kernel to bridge wider gaps between regions
    if bridge_distance > 0:
        kernel = np.ones((bridge_distance, bridge_distance), dtype=np.uint8)
        dilated = ndimage.binary_dilation(result, structure=kernel, iterations=1)
        result = ndimage.binary_erosion(dilated, structure=kernel, iterations=1).astype(result.dtype)

    # Step 4: final hole fill
    result = ndimage.binary_fill_holes(result).astype(result.dtype)

    return result