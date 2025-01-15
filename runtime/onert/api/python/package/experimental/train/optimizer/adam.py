from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        """
        Initialize the Adam optimizer.
        Args:
            learning_rate (float): The learning rate for optimization.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant to prevent division by zero.
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
