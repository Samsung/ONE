import numpy as np
from typing import List
from .metric import Metric


class CategoricalAccuracy(Metric):
    """
    Metric for computing categorical accuracy.
    """
    def __init__(self) -> None:
        """
        Initialize internal counters and axis.
        """
        self.correct: int = 0
        self.total: int = 0
        self.axis: int = 0

    def reset_state(self) -> None:
        """
        Reset the metric's state.
        """
        self.correct = 0
        self.total = 0

    def update_state(self, outputs: List[np.ndarray],
                     expecteds: List[np.ndarray]) -> None:
        """
        Update the metric's state based on the outputs and expecteds.

        Args:
            outputs (list of np.ndarray): List of model outputs for each output layer.
            expecteds (list of np.ndarray): List of expected ground truth values for each output layer.
        """
        if len(outputs) != len(expecteds):
            raise ValueError(
                "The number of outputs and expecteds must match. "
                f"Got {len(outputs)} outputs and {len(expecteds)} expecteds.")

        for output, expected in zip(outputs, expecteds):
            if output.shape[self.axis] != expected.shape[self.axis]:
                raise ValueError(
                    f"Output and expected shapes must match along the specified axis {self.axis}. "
                    f"Got output shape {output.shape} and expected shape {expected.shape}."
                )

            batch_size = output.shape[self.axis]
            for b in range(batch_size):
                output_idx = np.argmax(output[b])
                expected_idx = np.argmax(expected[b])
                if output_idx == expected_idx:
                    self.correct += 1
            self.total += batch_size

    def result(self) -> float:
        """
        Compute and return the final metric value.

        Returns:
            float: Metric value.
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total
