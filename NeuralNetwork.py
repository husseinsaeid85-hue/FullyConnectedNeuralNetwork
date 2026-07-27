from copy import deepcopy


class NeuralNetwork:
    """Orchestrates a stack of layers, a loss layer and a data source.

    Assemble a network by setting :attr:`data_layer` and :attr:`loss_layer`,
    then appending layers with :meth:`append_layer`. Call :meth:`train` to fit
    and :meth:`test` to run inference.

    Args:
        optimizer: Prototype optimizer. Each trainable layer is given a deep
            copy of it, so per-layer optimizer state stays independent.

    Attributes:
        loss: Loss value recorded once per training iteration.
        layers: The layer stack, in forward order.
        data_layer: Object exposing ``next() -> (input_tensor, label_tensor)``.
        loss_layer: Object exposing ``forward(prediction, labels)`` and
            ``backward(labels)``.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []  # loss value after calling train
        self.layers = []
        self.data_layer = None  # contains input data and labels
        self.loss_layer = None  # contains loss and prediction
        self.label_tensor = None  # labels of the current batch, kept for backward

    def forward(self):
        """Pull one batch and run it through the stack and the loss layer.

        The batch labels are cached so :meth:`backward` can reuse them.

        Returns:
            The scalar loss for the batch.
        """
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.forward(input_tensor, label_tensor)  # calculate the output of the last layer (loss)

    def backward(self):
        """Propagate the error from the loss layer back through the stack.

        Trainable layers update their own weights inside their ``backward``,
        using the optimizer assigned by :meth:`append_layer`.

        Returns:
            The error tensor arriving at the first layer's input.
        """
        error = self.loss_layer.backward(self.label_tensor)
        backward_layers = self.layers[::-1]  # reverse layers
        for layer in backward_layers:
            error = layer.backward(error)
        return error

    def append_layer(self, layer):
        """Add a layer to the end of the stack.

        Trainable layers receive a deep copy of the network's optimizer so that
        each layer keeps its own optimization state.

        Args:
            layer: A layer exposing ``forward`` and ``backward``.
        """
        if layer.trainable:
            optimizer_copy = deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        """Run forward/backward for a fixed number of batches.

        Appends the loss of every iteration to :attr:`loss`.

        Args:
            iterations: Number of batches to train on.
        """
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        """Run a forward pass only, without touching the loss layer.

        Args:
            input_tensor: Batch of inputs, ``[b, input_size]``.

        Returns:
            The output of the final layer, ``[b, output_size]``.
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
