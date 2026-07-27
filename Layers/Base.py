class BaseLayer:
    """Common interface shared by every layer in the framework.

    Subclasses are expected to implement ``forward(input_tensor)`` and
    ``backward(error_tensor)``. Layers that own parameters set ``trainable``
    to ``True`` so that :class:`NeuralNetwork` knows to hand them their own
    optimizer instance.

    Attributes:
        trainable: Whether the layer has parameters updated during training.
        weights: The parameter tensor, or ``None`` for parameter-free layers.
    """

    def __init__(self):
        self.trainable = False
        self.weights = None
