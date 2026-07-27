import numpy as np

from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    """Rectified Linear Unit activation, ``f(x) = max(0, x)``.

    Parameter-free. The forward input is cached so the backward pass can mask
    out the gradient wherever the input was non-positive.
    """

    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        """Apply the rectifier elementwise.

        Args:
            input_tensor: Batch with shape ``[b, n]``.

        Returns:
            Tensor of the same shape with negative entries clamped to zero.
        """
        self.input_tensor = input_tensor
        output_tensor = np.maximum(0, input_tensor)
        return output_tensor

    def backward(self, error_tensor):
        """Pass the error through only where the cached input was positive.

        Args:
            error_tensor: Gradient w.r.t. this layer's output, ``[b, n]``.

        Returns:
            Gradient w.r.t. this layer's input, same shape.
        """
        derivative_tensor = np.where(self.input_tensor > 0, error_tensor, 0)
        return derivative_tensor
