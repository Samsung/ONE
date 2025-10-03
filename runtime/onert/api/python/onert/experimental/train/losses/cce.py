from typing import Literal
import numpy as np
from .loss import LossFunction


class CategoricalCrossentropy(LossFunction):
    """
    Categorical Cross-Entropy Loss Function with reduction type.
    """
    def __init__(self, reduction: Literal["mean", "sum"] = "mean") -> None:
        """
        Initialize the Categorical Cross-Entropy loss function.
        Args:
            reduction (str): Reduction type ('mean', 'sum').
        """
        super().__init__(reduction)
