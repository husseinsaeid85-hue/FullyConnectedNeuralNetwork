import numpy as np

from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    """SoftMax activation turning logits into a per-row probability distribution.

    Parameter-free. Used as the final layer for classification, paired with
    :class:`Optimization.Loss.CrossEntropyLoss`.
    """

    def __init__(self):
        super().__init__()
        self.softmax_forward = None

    def forward(self, input_tensor):
        """Normalise each row of logits into probabilities.

        Each row is shifted by its own maximum before exponentiating, which
        leaves the result unchanged mathematically but avoids overflow.

        Args:
            input_tensor: Logits with shape ``[b, k]``.

        Returns:
            Probabilities with shape ``[b, k]``; every row sums to one.
        """
        shifted_exp = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.softmax_forward = shifted_exp / np.sum(shifted_exp, axis=1, keepdims=True)
        return self.softmax_forward

    def backward(self, error_tensor):
        """Apply the SoftMax Jacobian to the incoming error.

        Uses the identity ``J^T e = y * (e - sum(e * y))``, which avoids ever
        materialising the full ``[k, k]`` Jacobian per sample.

        Args:
            error_tensor: Gradient w.r.t. the probabilities, ``[b, k]``.

        Returns:
            Gradient w.r.t. the logits, same shape.
        """
        centered_error = error_tensor - np.sum(error_tensor * self.softmax_forward, axis=1, keepdims=True)
        softmax_backward = self.softmax_forward * centered_error
        return softmax_backward
