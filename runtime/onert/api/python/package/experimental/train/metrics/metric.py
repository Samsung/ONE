from typing import Any


class Metric:
    """
    Abstract base class for all metrics.
    """
    def reset_state(self) -> None:
        """
        Reset the metric's state.
        """
        raise NotImplementedError

    def update_state(self, outputs: Any, expecteds: Any) -> None:
        """
        Update the metric's state based on the outputs and expecteds.

        Args:
            outputs (Any): Model outputs.
            expecteds (Any): Expected ground truth values.
        """
        raise NotImplementedError

    def result(self) -> float:
        """
        Compute and return the final metric value.

        Returns:
            float: Metric value.
        """
        raise NotImplementedError
