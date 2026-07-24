"""This module provides a Multi-Task Estimator for depth, foreground, and normal estimation.

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

from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
from .pixelwise_estimator import PixelwiseEstimator
from .utils import composite_model_output_to_image


class MultiTaskEstimator(PixelwiseEstimator):
    """Estimates depth, foreground, and surface normals from a single multi-task model."""

    def __init__(
        self,
        onnx_model: Union[str, Path],
        providers: Optional[list[str]] = None,
        is_inverse_depth: bool = True,
        binarization_threshold: Optional[float] = None,
    ):
        """Creates a multi-task estimator.

        Arguments:
            onnx_model: A path to an ONNX multi-task model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].
            is_inverse_depth: If True, the depth map is inverted (i.e., closer objects have higher values).
            binarization_threshold: Optional threshold for binarizing foreground mask.

        Raises:
            TypeError: if onnx_model is not a string or Path.
            ModelNotFoundError: if the model file does not exist.
        """
        super().__init__(
            onnx_model,
            providers=providers,
        )
        self.is_inverse_depth = is_inverse_depth
        self.binarization_threshold = binarization_threshold

    def estimate_all_tasks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict depth, foreground, and surface normals from input image.

        Args:
            image: Input RGB image as numpy array

        Returns:
            Dictionary containing:
            - 'depth': Relative depth map
            - 'foreground': Soft foreground segmentation mask
            - 'normal': Surface normal map
        """
        # Run inference on the multi-task model
        outputs, metadata = self._estimate_dense_map(image)

        results = {}

        # Parse outputs based on expected model structure
        # This assumes outputs are in order: [depth, normal, foreground]
        if len(outputs) >= 3:
            depth_raw = outputs[0][0]  # First output
            normal_raw = outputs[1][0]  # Second output
            foreground_raw = outputs[2][0]  # Third output

            # Process depth
            depth_map = composite_model_output_to_image(depth_raw, metadata, interp_mode=cv2.INTER_CUBIC)
            if self.is_inverse_depth:
                depth_map = depth_map * -1
            results["depth"] = depth_map

            # Process normals
            normal_transposed = np.transpose(normal_raw, (1, 2, 0))
            normal_map = composite_model_output_to_image(normal_transposed, metadata, interp_mode=cv2.INTER_CUBIC)
            # Normalize normals
            normal_map /= np.linalg.norm(normal_map, axis=-1, keepdims=True) + 1e-8
            results["normal"] = normal_map

            # Process foreground
            foreground_transposed = np.transpose(foreground_raw, (1, 2, 0))
            foreground_map = composite_model_output_to_image(
                foreground_transposed, metadata, interp_mode=cv2.INTER_CUBIC
            )
            # Clip to [0, 1] and apply threshold if set
            foreground_map = np.clip(foreground_map, 0, 1)
            if self.binarization_threshold:
                foreground_map = ((foreground_map > self.binarization_threshold) * 1).astype(np.uint8)
            results["foreground"] = foreground_map

        return results

    def estimate_relative_depth(self, image: np.ndarray) -> np.ndarray:
        """Predict the relative depth map given input image."""
        results = self.estimate_all_tasks(image)
        return results.get("depth", np.zeros(image.shape[:2], dtype=np.float32))

    def estimate_normal(self, image: np.ndarray) -> np.ndarray:
        """Predict the normal map given input image."""
        results = self.estimate_all_tasks(image)
        return results.get("normal", np.zeros((*image.shape[:2], 3), dtype=np.float32))

    def estimate_foreground_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Predict the soft foreground/background segmentation given input image."""
        results = self.estimate_all_tasks(image)
        return results.get("foreground", np.zeros(image.shape[:2], dtype=np.float32))
