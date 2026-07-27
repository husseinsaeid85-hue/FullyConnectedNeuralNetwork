import numpy as np


class CrossEntropyLoss:
    """Categorical cross-entropy for one-hot labels.

    Expects probabilities as input, so it is normally placed directly after a
    :class:`Layers.SoftMax.SoftMax` layer. The loss is summed over the batch
    rather than averaged.
    """

    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        """Compute the loss for one batch.

        Args:
            prediction_tensor: Predicted probabilities, ``[b, k]``.
            label_tensor: One-hot ground truth, ``[b, k]``.

        Returns:
            Scalar loss summed over the batch, ``-sum(log p_true)``.
        """
        self.prediction_tensor = prediction_tensor
        return np.where(label_tensor == 1, -np.log(self.prediction_tensor+np.finfo(float).eps), 0).sum()

    def backward(self, label_tensor):
        """Start the backward pass.

        Args:
            label_tensor: One-hot ground truth, ``[b, k]``.

        Returns:
            Gradient w.r.t. the predictions from the last :meth:`forward`,
            non-zero only at the true class.
        """
        # The epsilon belongs inside the denominator, matching the forward pass's
        # -log(p + eps). Outside it, the division is unguarded and blows up to
        # inf/nan whenever a predicted probability reaches exactly zero.
        return np.where(label_tensor == 1, -1 / (self.prediction_tensor + np.finfo(float).eps), 0)
