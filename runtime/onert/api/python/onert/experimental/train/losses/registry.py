from typing import Type, Dict
from .loss import LossFunction
from .cce import CategoricalCrossentropy
from .mse import MeanSquaredError
from onert.native.libnnfw_api_pybind import loss as loss_type


class LossRegistry:
    """
    Registry for creating and mapping losses by name or instance.
    """
    _losses: Dict[str, Type[LossFunction]] = {
        "categorical_crossentropy": CategoricalCrossentropy,
        "mean_squared_error": MeanSquaredError
    }

    @staticmethod
    def create_loss(name: str) -> LossFunction:
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

    @staticmethod
    def map_loss_function_to_enum(loss_instance: LossFunction) -> loss_type:
        """
        Maps a LossFunction instance to the appropriate enum value.
        Args:
            loss_instance (BaseLoss): An instance of a loss function.
        Returns:
            loss_type: Corresponding enum value for the loss function.
        Raises:
            TypeError: If the loss_instance is not a recognized LossFunction type.
        """
        loss_to_enum: Dict[Type[LossFunction], loss_type] = {
            CategoricalCrossentropy: loss_type.CATEGORICAL_CROSSENTROPY,
            MeanSquaredError: loss_type.MEAN_SQUARED_ERROR
        }
        for loss_class, enum_value in loss_to_enum.items():
            if isinstance(loss_instance, loss_class):
                return enum_value
        raise TypeError(
            f"Unsupported loss function type: {type(loss_instance).__name__}. "
            f"Supported types are: {list(loss_to_enum.keys())}.")
