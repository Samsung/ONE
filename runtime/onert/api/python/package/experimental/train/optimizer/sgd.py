from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, learning_rate=0.001, momentum=0.0):
        """
        Initialize the SGD optimizer.

        Args:
            learning_rate (float): The learning rate for optimization.
            momentum (float): Momentum factor (default: 0.0).
        """
        super().__init__(learning_rate)

        if momentum != 0.0:
            raise NotImplementedError(
                "Momentum is not supported in the current version of SGD.")
        self.momentum = momentum
        self.velocity = None

    def step(self, gradients, parameters):
        """
        Update parameters using SGD with optional momentum.

        Args:
            gradients (list): List of gradients for each parameter.
            parameters (list): List of parameters to be updated.
        """
        if self.velocity is None:
            self.velocity = [0] * len(parameters)

        for i, (grad, param) in enumerate(zip(gradients, parameters)):
            self.velocity[
                i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            parameters[i] += self.velocity[i]
