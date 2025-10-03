from typing import Literal
from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-7) -> None:
        """
        Initialize the Adam optimizer.

        Args:
            learning_rate (float): The learning rate for optimization.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant to prevent division by zero.
        """
        super().__init__(learning_rate)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
