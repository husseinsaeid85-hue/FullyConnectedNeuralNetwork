import numpy as np

from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    """Affine (dense) layer computing ``output = [input, 1] @ weights``.

    The bias is folded into the weight matrix by appending a constant column
    of ones to the input, so ``weights`` has shape
    ``(input_size + 1, output_size)`` and the last row holds the bias.

    Args:
        input_size: Number of input features per sample.
        output_size: Number of output features per sample.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (self.input_size+1, self.output_size))
        self._gradient_weights = None
        self._optimizer = None
        self._last_input = None

    def forward(self, input_tensor):
        """Compute the affine transform of a batch.

        Args:
            input_tensor: Batch with shape ``[b, input_size]``.

        Returns:
            Tensor with shape ``[b, output_size]``.
        """
        one_vec = np.ones((input_tensor.shape[0], 1))  # dim: [b,1] for bias
        self._last_input = np.concatenate((input_tensor, one_vec), axis=1)
        output = np.dot(self._last_input, self.weights)
        return output

    def backward(self, error_tensor):
        """Propagate the error backwards and update the weights.

        Stores the weight gradient in :attr:`gradient_weights` and, if an
        optimizer has been assigned, applies the update in place.

        Args:
            error_tensor: Gradient w.r.t. this layer's output, ``[b, output_size]``.

        Returns:
            Gradient w.r.t. this layer's input, ``[b, input_size]``. The bias
            column is dropped since the previous layer has no bias input.
        """
        self._gradient_weights = np.dot(self._last_input.T, error_tensor)  # dim=[n+1, m]
        error_tensor = np.dot(error_tensor, self.weights.T[:, :-1])  # dim = [b, n] error tensor for the previous layer
        if self._optimizer:  # update weights
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return error_tensor

    @property
    def optimizer(self):
        """The optimizer used to update this layer's weights."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val

    @property
    def gradient_weights(self):
        """Weight gradient from the most recent backward pass."""
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val
