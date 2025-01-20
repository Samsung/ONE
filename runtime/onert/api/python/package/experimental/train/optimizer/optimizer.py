from onert.native.libnnfw_api_pybind import trainable_ops


class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, learning_rate=0.001, nums_trainable_ops=trainable_ops.ALL):
        """
        Initialize the optimizer.
        Args:
            learning_rate (float): The learning rate for optimization.
        """
        self.learning_rate = learning_rate
        self.nums_trainable_ops = nums_trainable_ops
