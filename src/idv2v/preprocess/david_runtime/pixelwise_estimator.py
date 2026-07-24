"""Runtime core for pixelwise estimators.

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

import numpy as np
from onnxruntime import InferenceSession
from .utils import ONNX_EP, ModelNotFoundError, prepare_image_for_model, preprocess_img


class RuntimeSession(InferenceSession):
    """The runtime session."""

    def __init__(self, onnx_model: Union[str, Path], providers: Optional[list[str]] = None) -> None:
        """Create a runtime session.

        Args:
            onnx_model: The path to the onnx model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].
        """
        super().__init__(str(onnx_model), providers=providers or ONNX_EP)
        self.onnx_model_path: Path = Path(onnx_model)

    @property
    def input_name(self) -> str:
        """Get the name of the input tensor."""
        return self.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        """Run the model on the input tensor."""
        x = x.astype(np.float32)
        return self.run(None, {self.input_name: x})


class PixelwiseEstimator:
    """Given an input image, estimates the pixelwise (dense) output (e.g., normal map, depth map, etc.)."""

    def __init__(self, onnx_model: Union[str, Path], providers: Optional[list[str]] = None):
        """Creates a pixelwise estimator.

        Arguments:
            onnx_model: Path to an ONNX model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].

        Raises:
            TypeError: If onnx_model is not a string or Path.
            ModelNotFoundError: If the model file does not exist.
            ModelError: If the provided model has an undeclared or incorrect roi type.
        """
        if not isinstance(onnx_model, (str, Path)):
            raise TypeError(f"onnx_model should be a string or Path, got {type(onnx_model)}")
        onnx_model = Path(onnx_model)
        if not onnx_model.exists():
            raise ModelNotFoundError(f"model {onnx_model} does not exist")

        self.onnx_model = onnx_model

        self.roi_size = 512

        self.onnx_sess = RuntimeSession(str(onnx_model), providers=providers)

    @staticmethod
    def inference(input_img: np.ndarray, onnx_sess: RuntimeSession) -> np.ndarray:
        """Predict the pixelwise (dense) map given an input image.

        Args:
            input_img: Input image.
            onnx_sess: ONNX inference session.

        Returns:
            Predicted output map.
        """
        input_tensor = onnx_sess.get_inputs()[0]
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        input_img = np.transpose(input_img, (2, 0, 1)).reshape(1, *input_shape[1:])  # HWC to BCHW
        pred_onnx = onnx_sess.run(None, {input_name: input_img.astype(np.float32)})

        return pred_onnx

    def _estimate_dense_map(self, image: np.ndarray) -> tuple[np.ndarray]:
        """Estimating dense maps from image input."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image should be a numpy array, got {type(image)}")

        image_bgr = preprocess_img(image)
        processed_image, metadata = prepare_image_for_model(image_bgr, self.roi_size)
        output = self.inference(processed_image, self.onnx_sess)

        return output, metadata
