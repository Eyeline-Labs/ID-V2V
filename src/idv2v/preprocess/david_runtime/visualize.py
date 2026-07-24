"""Visualization utilities for images, masks, and depth maps.

Copyright (c) Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Optional

import cv2
import numpy as np


def visualize_foreground(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: Optional[tuple[int, int, int]] = (0, 255, 0),
) -> np.ndarray:
    """Visualizes a foreground mask on top of an image.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The foreground mask. It can be a binary mask or a soft mask.
        background_color (tuple): The color of the background.

    Returns:
        np.ndarray: The composite image with the binary mask visualized.
    """
    mask = np.expand_dims(mask, -1)
    mask = np.clip(mask, 0, 1)
    background = np.full(image.shape, background_color)

    # Create the composite image
    composite_image = (image.astype(np.float32) * mask).astype(np.uint8) + (
        background.astype(np.float32) * (1 - mask)
    ).astype(np.uint8)

    return composite_image


def visualize_normal_maps(frame: np.ndarray, normals: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Visualize normal map and overlay it on the original image if soft mask is provided."""
    # check that dimensions of frame and normals and mask if exists are the same
    if frame.shape[0:2] != normals.shape[0:2]:
        raise ValueError("The dimensions of 'frame' and 'normals' must match.")
    if mask is not None and frame.shape[0:2] != mask.shape[0:2]:
        raise ValueError("The dimensions of 'frame' and 'mask' must match.")
    vis_normals = ((normals / 2.0 + 0.5) * 255)[:, :, ::-1].astype(np.uint8)

    if mask is not None:
        vis_normals[mask == 0] = 0
        mask = np.expand_dims(mask, -1)
        mask = np.clip(mask, 0, 1)
        vis_normals = (vis_normals.astype(np.float32) * mask).astype(np.uint8) + (
            frame.astype(np.float32) * (1 - mask)
        ).astype(np.uint8)

    return vis_normals


def visualize_relative_depth_map(
    frame: np.ndarray,
    depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    alpha_threshold: float = 0.0,
) -> np.ndarray:
    """Visualize relative depth map and overlay it on the original image if soft mask is provided."""
    processed_depth = np.full((frame.shape[0], frame.shape[1], 3), 0, dtype=np.uint8)
    if frame.shape[0:2] != depth.shape[0:2]:
        raise ValueError("The dimensions of 'frame' and 'depth' must match.")
    if mask is not None and frame.shape[0:2] != mask.shape[0:2]:
        raise ValueError("The dimensions of 'frame' and 'mask' must match.")
    if mask is not None:
        foreground = np.logical_and(
            mask > alpha_threshold, depth != 65504
        )  # account for invalid depth values in GT images
        if not np.any(foreground):
            return processed_depth
        depth_foreground = depth[foreground]
        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)

        depth_normalized_foreground = 1 - (
            (depth_foreground - min_val) / (max_val - min_val if max_val != min_val else 1e-8)
        )
        depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)
        depth_colored_foreground = cv2.applyColorMap(depth_normalized_foreground, cv2.COLORMAP_INFERNO)
        processed_depth[foreground] = depth_colored_foreground.reshape(-1, 3)
        mask = np.clip(mask[..., None], 0, 1).astype(np.float32)
        mask = np.repeat(mask, 3, axis=-1)
        processed_depth[mask == 0] = 0
        vis_depth = (processed_depth.astype(np.float32) * mask).astype(np.uint8) + (
            frame.astype(np.float32) * (1 - mask)
        ).astype(np.uint8)
    else:
        min_val, max_val = np.min(depth), np.max(depth)
        depth_normalized = 1 - ((depth - min_val) / (max_val - min_val if max_val != min_val else 1e-8))
        depth_normalized = (depth_normalized * 255.0).astype(np.uint8)
        vis_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    return vis_depth


def create_concatenated_display(visualizations: list[np.ndarray], labels: list[str], downscale: int = 1):
    """Create a horizontally concatenated display with labels."""
    assert len(visualizations) == len(labels), "Number of visualizations must match number of labels"
    # Resize all images to same height for concatenation
    target_height = visualizations[0].shape[0] // downscale  # Make smaller for display

    resized_vis = []
    for vis in visualizations:
        aspect_ratio = vis.shape[1] / vis.shape[0]
        target_width = int(target_height * aspect_ratio)
        resized = cv2.resize(vis, (target_width, target_height))
        resized_vis.append(resized)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2

    for vis, label in zip(resized_vis, labels):
        cv2.putText(vis, label, (10, 30), font, font_scale, color, thickness)

    return cv2.hconcat(resized_vis)
