import numpy as np
from .loss import LossFunction


class CategoricalCrossentropy(LossFunction):
    """
    Categorical Cross-Entropy Loss Function with reduction type.
    """
    def __init__(self, reduction="mean"):
        """
        Initialize the Categorical Cross-Entropy loss function.
        Args:
            reduction (str): Reduction type ('mean', 'sum').
        """
        super().__init__(reduction)

    def __call__(self, y_true, y_pred):
        """
        Compute the Categorical Cross-Entropy loss.
        Args:
            y_true (np.ndarray): One-hot encoded ground truth values.
            y_pred (np.ndarray): Predicted probabilities.
        Returns:
            float or np.ndarray: Computed loss value(s).
        """
        epsilon = 1e-7  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)

        if self.reduction == "mean":
            return np.mean(loss)
        elif self.reduction == "sum":
            return np.sum(loss)
