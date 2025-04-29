from typing import List, Union, Tuple, Dict
import numpy as np
import time
import warnings
from contextlib import contextmanager

from ..native.libnnfw_api_pybind import infer, tensorinfo
from ..common.basesession import BaseSession


class session(BaseSession):
    """
    Class for inference using nnfw_session.
    """
    def __init__(self, path: str, backends: str = "cpu") -> None:
        """
        Initialize the inference session.

        Args:
            path (str): Path to the model file or nnpackage directory.
            backends (str): Backends to use, default is "cpu".
        """
        super().__init__(infer.nnfw_session(path, backends))
        self._prepared: bool = False

    def update_inputs_tensorinfo(self, new_infos: List[tensorinfo]) -> None:
        """
        Update all input tensors' tensorinfo at once.

        Args:
            new_infos (list[tensorinfo]): A list of updated tensorinfo objects for the inputs.

        Raises:
            ValueError: If the number of new_infos does not match the session's input size,
                        or if any tensorinfo contains a negative dimension.
        """
        num_inputs: int = self.session.input_size()
        if len(new_infos) != num_inputs:
            raise ValueError(
                f"Expected {num_inputs} input tensorinfo(s), but got {len(new_infos)}.")

        for i, info in enumerate(new_infos):
            # Check for any negative dimension in the specified rank
            if any(d < 0 for d in info.dims[:info.rank]):
                raise ValueError(
                    f"Input tensorinfo at index {i} contains negative dimension(s): "
                    f"{info.dims[:info.rank]}")
            self.session.set_input_tensorinfo(i, info)

    def infer(
        self,
        inputs_array: List[np.ndarray],
        *,
        measure: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict[str, float]]]:
        """
        Run a complete inference cycle:
         - If the session has not been prepared or outputs have not been set, call prepare() and set_outputs().
         - Automatically configure input buffers based on the provided numpy arrays.
         - Execute the inference session.
         - Return the output tensors with proper multi-dimensional shapes.

        This method supports both static and dynamic shape modification:
         - If update_inputs_tensorinfo() has been called before running inference, the model is compiled
           with the fixed static input shape.
         - Otherwise, the input shapes can be adjusted dynamically.

        Args:
            inputs_array (list[np.ndarray]): List of numpy arrays representing the input data.
            measure (bool): If True, measure prepare/io/run latencies (ms).

        Returns:
            list[np.ndarray]: A list containing the output numpy arrays.
            OR
            (outputs, metrics): Tuple where metrics is a dict with keys
                'prepare_time_ms', 'io_time_ms', 'run_time_ms'
        """
        metrics: Dict[str, float] = {}

        # Verify that the number of provided inputs matches the session's expected input count.
        expected_input_size: int = self.session.input_size()
        if len(inputs_array) != expected_input_size:
            raise ValueError(
                f"Expected {expected_input_size} input(s), but received {len(inputs_array)}."
            )

        # Check if the session is prepared. If not, call prepare() and set_outputs() once.
        if not self._prepared:
            with self._time_block(metrics, 'prepare_time_ms', measure):
                # On first call, fix any -1 dims to real input shapes and validate
                original_infos = self.get_inputs_tensorinfo()
                fixed_infos = []
                for idx, info in enumerate(original_infos):
                    input_shape = inputs_array[idx].shape
                    new_dims = []
                    static_dim_changed = False
                    # only the first `info.rank` entries matter
                    for j, d in enumerate(info.dims[:info.rank]):
                        if d == -1:
                            # replace dynamic dim with actual incoming shape
                            new_dims.append(input_shape[j])
                        elif d == input_shape[j]:
                            # static dim must match the provided array
                            new_dims.append(d)
                        else:
                            static_dim_changed = True

                    if static_dim_changed:
                        warnings.warn(
                            f"infer() called with input {idx}'s shape={input_shape}, "
                            f"which differs from model’s expected shape={tuple(info.dims)}. "
                            "Ensure this is intended.", UserWarning)

                    info.dims = new_dims
                    fixed_infos.append(info)

                # Update tensorinfo to optimize using it
                self.update_inputs_tensorinfo(fixed_infos)

                self.session.prepare()
                self.set_outputs(self.session.output_size())
                self._prepared = True

        # Configure input buffers using the current session's input size and provided data.
        with self._time_block(metrics, 'io_time_ms', measure):
            self.set_inputs(expected_input_size, inputs_array)

        # Execute the inference.
        with self._time_block(metrics, 'run_time_ms', measure):
            self.session.run()

        # TODO: Support dynamic shapes for outputs.

        # Return the output buffers.
        return (self.outputs.copy(), metrics) if measure else self.outputs.copy()

    @contextmanager
    def _time_block(self, metrics: Dict[str, float], key: str, mesure: bool):
        if mesure:
            start = time.perf_counter()
            yield
            metrics[key] = (time.perf_counter() - start) * 1000
        else:
            yield
