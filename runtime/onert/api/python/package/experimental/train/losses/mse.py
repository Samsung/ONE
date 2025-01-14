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
