from .cce import CategoricalCrossentropy
from .mse import MeanSquaredError


class LossRegistry:
    """
    Registry for creating losses by name.
    """
    _losses = {
        "categorical_crossentropy": CategoricalCrossentropy,
        "mean_squred_error": MeanSquaredError
    }

    @staticmethod
    def create_loss(name):
        """
        Create a loss instance by name.
        Args:
            name (str): Name of the loss.
        Returns:
            BaseLoss: Loss instance.
        """
        if name not in LossRegistry._losses:
            raise ValueError(f"Unknown Loss: {name}. Custom loss is not supported yet")
        return LossRegistry._losses[name]()
