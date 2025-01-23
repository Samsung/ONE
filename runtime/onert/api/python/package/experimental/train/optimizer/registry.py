from onert.native.libnnfw_api_pybind import optimizer as optimizer_type
from .adam import Adam
from .sgd import SGD


class OptimizerRegistry:
    """
    Registry for creating optimizers by name.
    """
    _optimizers = {"adam": Adam, "sgd": SGD}

    @staticmethod
    def create_optimizer(name):
        """
        Create an optimizer instance by name.
        Args:
            name (str): Name of the optimizer.
        Returns:
            BaseOptimizer: Optimizer instance.
        """
        if name not in OptimizerRegistry._optimizers:
            raise ValueError(
                f"Unknown Optimizer: {name}. Custom optimizer is not supported yet")
        return OptimizerRegistry._optimizers[name]()

    @staticmethod
    def map_optimizer_to_enum(optimizer_instance):
        """
        Maps an optimizer instance to the appropriate enum value.
        Args:
            optimizer_instance (Optimizer): An instance of an optimizer.
        Returns:
            optimizer_type: Corresponding enum value for the optimizer.
        Raises:
            TypeError: If the optimizer_instance is not a recognized optimizer type.
        """
        # Optimizer to Enum mapping
        optimizer_to_enum = {SGD: optimizer_type.SGD, Adam: optimizer_type.ADAM}
        for optimizer_class, enum_value in optimizer_to_enum.items():
            if isinstance(optimizer_instance, optimizer_class):
                return enum_value
        raise TypeError(
            f"Unsupported optimizer type: {type(optimizer_instance).__name__}. "
            f"Supported types are: {list(optimizer_to_enum.keys())}.")
