from .categorical_accuracy import CategoricalAccuracy


class MetricsRegistry:
    """
    Registry for creating metrics by name.
    """
    _metrics = {
        "categorical_accuracy": CategoricalAccuracy,
    }

    @staticmethod
    def create_metric(name):
        """
        Create a metric instance by name.
        Args:
            name (str): Name of the metric.
        Returns:
            BaseMetric: Metric instance.
        """
        if name not in MetricsRegistry._metrics:
            raise ValueError(
                f"Unknown Metric: {name}. Custom metric is not supported yet")
        return MetricsRegistry._metrics[name]()
