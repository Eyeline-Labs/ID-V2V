"""This module provides a SoftForegroundSegmenter which segments the foreground human subjects from the background.

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


class SoftForegroundSegmenter(PixelwiseEstimator):
    """Estimates the soft foreground segmentation mask of human in an image."""

    def __init__(
        self,
        onnx_model: Union[str, Path],
        providers: Optional[list[str]] = None,
        binarization_threshold: Optional[float] = None,
    ):
        """Creates a soft foreground segmenter to segment the foreground human subjects in an image.

        Arguments:
            onnx_model: A path to an ONNX model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].
            binarization_threshold: Threshold above which the mask is considered foreground. When None, the mask is returned as is.

        Raises:
            TypeError: if onnx_model is not a string or Path.
            ModelNotFoundError: if the model file does not exist.
        """
        super().__init__(
            onnx_model,
            providers=providers,
        )
        self.binarization_threshold = binarization_threshold

    def estimate_foreground_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Predict the soft foreground/background segmentation given input image."""
        mask, metadata = self._estimate_dense_map(image)
        mask = mask[0][0]
        mask = np.transpose(mask, (1, 2, 0))

        # post_process to get the final segmentation mask and composite it onto the original size
        segmented_image = composite_model_output_to_image(mask, metadata, interp_mode=cv2.INTER_CUBIC)

        # clip the mask to [0, 1]
        segmented_image = np.clip(segmented_image, 0, 1)

        # Apply threshold if binarization_threshold is set
        if self.binarization_threshold:
            return ((segmented_image > self.binarization_threshold) * 1).astype(np.uint8)

        return segmented_image
