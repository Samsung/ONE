class Metric:
    """
    Abstract base class for all metrics.
    """
    def reset_state(self):
        """
        Reset the metric's state.
        """
        raise NotImplementedError

    def update_state(self, outputs, expecteds):
        """
        Update the metric's state based on the outputs and expecteds.
        Args:
            outputs (np.ndarray): Model outputs.
            expecteds (np.ndarray): Expected ground truth values.
        """
        raise NotImplementedError

    def result(self):
        """
        Compute and return the final metric value.
        Returns:
            float: Metric value.
        """
        raise NotImplementedError
