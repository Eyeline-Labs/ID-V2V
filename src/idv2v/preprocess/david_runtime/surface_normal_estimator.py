"""This module provides a Surface Normal Estimator which estimates the surface normal map of human in an image.

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
from typing import Optional, Union

import cv2
import numpy as np
from .pixelwise_estimator import PixelwiseEstimator
from .utils import composite_model_output_to_image


class SurfaceNormalEstimator(PixelwiseEstimator):
    """Estimates the surface normal map of human in an image."""

    def __init__(
        self,
        onnx_model: Union[str, Path],
        providers: Optional[list[str]] = None,
    ):
        """Creates a surface normal estimator.

        Arguments:
            onnx_model: A path to an ONNX model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].

        Raises:
            TypeError: if onnx_model is not a string or Path.
            ModelNotFoundError: if the model file does not exist.
        """
        super().__init__(
            onnx_model,
            providers=providers,
        )

    def estimate_normal(self, image: np.ndarray) -> np.ndarray:
        """Predict the normal map given input image."""
        normal, metadata = self._estimate_dense_map(image)
        normal = normal[0][0]
        normal = np.transpose(normal, (1, 2, 0))

        normal_map = composite_model_output_to_image(normal, metadata, interp_mode=cv2.INTER_CUBIC)
        normal_map /= np.linalg.norm(normal_map, axis=-1, keepdims=True) + 1e-8
        return normal_map
