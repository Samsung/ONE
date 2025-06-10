from onert.native.libnnfw_api_pybind import trainable_ops


class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 nums_trainable_ops: int = trainable_ops.ALL) -> None:
        """
        Initialize the optimizer.

        Args:
            learning_rate (float): The learning rate for optimization.
            nums_trainable_ops (int or enum): Number of trainable ops or enum mask.
        """
        self.learning_rate: float = learning_rate
        self.nums_trainable_ops: int = nums_trainable_ops
