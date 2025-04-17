from typing import Type, Dict
from .metric import Metric
from .categorical_accuracy import CategoricalAccuracy


class MetricsRegistry:
    """
    Registry for creating metrics by name.
    """
    _metrics: Dict[str, Type[Metric]] = {
        "categorical_accuracy": CategoricalAccuracy,
    }

    @staticmethod
    def create_metric(name: str) -> Metric:
        """
        Create a metric instance by name.

        Args:
            name (str): Name of the metric.

        Returns:
            Metric: Metric instance.
        """
        if name not in MetricsRegistry._metrics:
            raise ValueError(
                f"Unknown Metric: {name}. Custom metric is not supported yet")
        return MetricsRegistry._metrics[name]()
