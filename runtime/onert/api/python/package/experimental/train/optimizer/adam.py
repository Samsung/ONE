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
        self.m = None
        self.v = None
        self.t = 0

    def step(self, gradients, parameters):
        """
        Update parameters using Adam optimization.
        Args:
            gradients (list): List of gradients for each parameter.
            parameters (list): List of parameters to be updated.
        """
        if self.m is None:
            self.m = [0] * len(parameters)
        if self.v is None:
            self.v = [0] * len(parameters)

        self.t += 1
        for i, (grad, param) in enumerate(zip(gradients, parameters)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            param -= self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)
