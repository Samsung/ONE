import numpy as np
from .loss import LossFunction


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE) Loss Function with reduction type.
    """
    def __init__(self, reduction="mean"):
        """
        Initialize the MSE loss function.
        Args:
            reduction (str): Reduction type ('mean', 'sum').
        """
        super().__init__(reduction)

    def __call__(self, y_true, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss.
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float or np.ndarray: Computed MSE loss value(s).
        """
        loss = (y_true - y_pred)**2
        if self.reduction == "mean":
            return np.mean(loss)
        elif self.reduction == "sum":
            return np.sum(loss)
