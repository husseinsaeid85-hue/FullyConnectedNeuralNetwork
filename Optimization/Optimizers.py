class Sgd:
    """Plain stochastic gradient descent.

    Every trainable layer receives its own copy of the optimizer (see
    :meth:`NeuralNetwork.NeuralNetwork.append_layer`), so stateful optimizers
    added later do not share state across layers.

    Args:
        learning_rate: Step size applied to the gradient.
    """

    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        """Take one descent step.

        Args:
            weight_tensor: Current parameters.
            gradient_tensor: Gradient of the loss w.r.t. those parameters.

        Returns:
            The updated parameters, ``w - learning_rate * grad``.
        """
        weight_tensor -= self.learning_rate * gradient_tensor
        return weight_tensor
