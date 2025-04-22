import time
import numpy as np
from typing import Any, List, Tuple, Dict, Union, Optional

from onert.native import libnnfw_api_pybind
from onert.native.libnnfw_api_pybind import traininfo
from onert.common.basesession import BaseSession
from .dataloader import DataLoader
from .losses.loss import LossFunction
from .losses.registry import LossRegistry
from .metrics.metric import Metric
from .metrics.registry import MetricsRegistry
from .optimizer.optimizer import Optimizer
from .optimizer.registry import OptimizerRegistry


# TODO: Support import checkpoint
class TrainSession(BaseSession):
    """
    Class for training and inference using nnfw_session.
    """
    def __init__(self, nnpackage_path: str, backends: str = "train") -> None:
        """
        Initialize the train session.

        Args:
            nnpackage_path (str): Path to the nnpackage file or directory.
            backends (str): Backends to use, default is "train".
        """
        load_start: float = time.perf_counter()
        super().__init__(
            libnnfw_api_pybind.experimental.nnfw_session(nnpackage_path, backends))
        load_end: float = time.perf_counter()

        self.total_time: Dict[str, Union[float, List[float]]] = {
            'MODEL_LOAD': (load_end - load_start) * 1000
        }
        self.train_info: traininfo = self.session.train_get_traininfo()
        self.optimizer: Optional[Optimizer] = None
        self.loss: Optional[LossFunction] = None
        self.metrics: List[Metric] = []

    def compile(self,
                optimizer: Union[str, Optimizer],
                loss: Union[str, LossFunction],
                metrics: List[Union[str, Metric]] = [],
                batch_size: int = 16) -> None:
        """
        Compile the session with optimizer, loss, and metrics.

        Args:
            optimizer (str or Optimizer): Optimizer instance or name.
            loss (str or LossFunction): Loss instance or name.
            metrics (list of str or Metric): Metrics to evaluate during training.
            batch_size (int): Number of samples per batch.
        """
        self.optimizer = (OptimizerRegistry.create_optimizer(optimizer) if isinstance(
            optimizer, str) else optimizer)
        self.loss = (LossRegistry.create_loss(loss) if isinstance(loss, str) else loss)
        self.metrics = [
            MetricsRegistry.create_metric(m) if isinstance(m, str) else m for m in metrics
        ]

        # Validate that all elements in self.metrics are instances of Metric
        for m in self.metrics:
            if not isinstance(m, Metric):
                raise TypeError(f"Invalid metric type: {type(m).__name__}. "
                                "All metrics must inherit from the Metric base class.")

        # Check if the number of metrics matches the number of outputs
        num_outputs: int = self.session.output_size()
        if 0 < len(self.metrics) != num_outputs:
            raise ValueError(
                f"Number of metrics ({len(self.metrics)}) does not match outputs ({num_outputs})"
                "Please ensure one metric is provided for each model output.")

        # Set training info
        self.train_info.learning_rate = self.optimizer.learning_rate
        self.train_info.batch_size = batch_size
        self.train_info.loss_info.loss = LossRegistry.map_loss_function_to_enum(self.loss)
        self.train_info.loss_info.reduction_type = self.loss.reduction
        self.train_info.opt = OptimizerRegistry.map_optimizer_to_enum(self.optimizer)
        self.train_info.num_of_trainable_ops = self.optimizer.nums_trainable_ops
        self.session.train_set_traininfo(self.train_info)

        # Print training parameters
        self._print_training_parameters()

        # Prepare session for training
        compile_start: float = time.perf_counter()
        self.session.train_prepare()
        compile_end: float = time.perf_counter()
        self.total_time["COMPILE"] = (compile_end - compile_start) * 1000

    def _print_training_parameters(self) -> None:
        """
        Print the training parameters in a formatted way.
        """
        loss_name: str = self.loss.__class__.__name__ if self.loss else "Unknown Loss"
        reduction_name: str = (
            self.train_info.loss_info.reduction_type.name.lower().replace("_", " "))
        opt_name: str = self.optimizer.__class__.__name__ if self.optimizer else "Unknown Optimizer"

        print("== training parameter ==")
        print(f"- learning_rate = {self.train_info.learning_rate:.4f}".rstrip('0').rstrip(
            '.'))
        print(f"- batch_size = {self.train_info.batch_size}")
        print(f"- loss_info = {{loss = {loss_name}, reduction = {reduction_name}}}")
        print(f"- optimizer = {opt_name}")
        print(f"- num_of_trainable_ops = {self.train_info.num_of_trainable_ops}")
        print("========================")

    def train(
            self,
            data_loader: DataLoader,
            epochs: int,
            validation_split: float = 0.0,
            checkpoint_path: Optional[str] = None
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Train the model using the given data loader.

        Args:
            data_loader: Data loader providing input and expected data.
            epochs (int): Number of epochs to train.
            validation_split (float): Ratio of validation data. Default is 0.0.
            checkpoint_path (str, optional): Path to save training checkpoints.

        Returns:
            dict: Timing and performance metrics.
        """
        if self.optimizer is None or self.loss is None:
            raise RuntimeError("Call compile() before train().")

        train_data, val_data = data_loader.split(validation_split)
        epoch_times: List[float] = []

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            train_loss, io_ms, train_ms = self._run_phase(train_data, train=True)
            msg = [
                f"Epoch {epoch+1}/{epochs}", f"Train time: {train_ms:.3f}ms/step",
                f"IO time: {io_ms:.3f}ms/step", f"Train Loss: {train_loss:.4f}"
            ]

            if validation_split > 0.0:
                val_loss, _, _ = self._run_phase(val_data, train=False)
                msg.append(f"Validation Loss: {val_loss:.4f}")
                for m in self.metrics:
                    msg.append(f"{m.__class__.__name__}: {m.result():.4f}")
                    m.reset_state()

            epoch_times.append((time.perf_counter() - epoch_start) * 1000)
            print(" - ".join(msg))

            if checkpoint_path:
                self.session.train_export_checkpoint(checkpoint_path)

        self.total_time["EXECUTE"] = sum(epoch_times)
        self.total_time["EPOCH_TIMES"] = epoch_times
        return self.total_time

    def _run_phase(self,
                   data: Tuple[List[np.ndarray], List[np.ndarray]],
                   train: bool = True) -> Tuple[float, float, float]:
        """
        Run a training or validation phase.

        Args:
            data: Data generator.
            train (bool): Whether to update weights.

        Returns:
            (avg_loss, avg_io_ms, avg_train_ms)
        """
        total_loss: float = 0.0
        num_batches: int = 0
        io_time: float = 0.0
        train_time: float = 0.0

        for inputs, expecteds in data:
            # Validate batch sizes
            self._check_batch_size(inputs, self.train_info.batch_size, "input")
            self._check_batch_size(expecteds, self.train_info.batch_size, "expected")

            # Set inputs
            io_start = time.perf_counter()
            for i, input_data in enumerate(inputs):
                self.session.train_set_input(i, input_data)

            # Set expected outputs
            outputs: List[np.ndarray] = []
            for i, expected_data in enumerate(expecteds):
                expected = np.array(expected_data,
                                    dtype=self.session.output_tensorinfo(i).dtype)
                self.session.train_set_expected(i, expected)
                output = np.zeros(expected.shape,
                                  dtype=self.session.output_tensorinfo(i).dtype)
                self.session.train_set_output(i, output)
                outputs.append(output)
            io_end = time.perf_counter()

            # Run training or validation
            t_start = time.perf_counter()
            self.session.train(update_weights=train)
            t_end = time.perf_counter()

            # Accumulate loss
            batch_loss = sum(
                self.session.train_get_loss(i) for i in range(len(expecteds)))
            total_loss += batch_loss
            num_batches += 1

            # Update metrics
            if not train:
                for m in self.metrics:
                    m.update_state(outputs, expecteds)

            # Calculate times
            io_time += (io_end - io_start)
            train_time += (t_end - t_start)

        if num_batches:
            return (total_loss / num_batches, (io_time * 1000) / num_batches,
                    (train_time * 1000) / num_batches)
        return (0.0, 0.0, 0.0)

    def _check_batch_size(self,
                          data: List[np.ndarray],
                          batch_size: int,
                          data_type: str = "input") -> None:
        """
        Validate that the batch size of the data matches the configured training batch size.

        Args:
            data (list of np.ndarray): The data to validate.
            batch_size (int): The expected batch size.
            data_type (str): 'input' or 'expected'.

        Raises:
            ValueError: If the batch size does not match the expected value.
        """
        for idx, arr in enumerate(data):
            if arr.shape[0] > batch_size:
                raise ValueError(
                    f"{data_type} batch size mismatch at index {idx}: "
                    f"shape[0] = {arr.shape[0]} vs batch size = {batch_size}")

    def train_step(self, inputs: List[np.ndarray],
                   expecteds: List[np.ndarray]) -> Dict[str, Any]:
        """
        Train the model for a single batch.

        Args:
            inputs (list of np.ndarray): List of input arrays for the batch.
            expecteds (list of np.ndarray): List of expected output arrays for the batch.

        Returns:
            dict: Loss and metrics values, and train_time in ms.
        """
        if self.optimizer is None or self.loss is None:
            raise RuntimeError("Call `compile()` before `train_step()`")
        result = self._batch_step(inputs, expecteds, update_weights=True)
        result["train_time"] = result.pop("time_ms")
        return result

    def eval_step(self, inputs: List[np.ndarray],
                  expecteds: List[np.ndarray]) -> Dict[str, Any]:
        """
        Run one evaluation batch: forward only (no weight update).

        Args:
            inputs (list of np.ndarray): Inputs for this batch.
            expecteds (list of np.ndarray): Ground‑truth outputs for this batch.

        Returns:
            dict: {
                "loss": list of float losses,
                "metrics": dict of metric_name -> value,
                "eval_time": float (ms)
            }
        """
        if self.optimizer is None or self.loss is None:
            raise RuntimeError("Call `compile()` before `eval_step()`")
        result = self._batch_step(inputs, expecteds, update_weights=False)
        result["eval_time"] = result.pop("time_ms")
        return result

    def _batch_step(self, inputs: List[np.ndarray], expecteds: List[np.ndarray],
                    update_weights: bool) -> Dict[str, Any]:
        """
        Common logic for one batch: bind data, run, collect loss & metrics.
        Returns a dict with keys "loss", "metrics", "time_ms".
        """
        # Validate batch sizes
        self._check_batch_size(inputs, self.train_info.batch_size, "input")
        self._check_batch_size(expecteds, self.train_info.batch_size, "expected")

        # Set inputs
        for i, input_data in enumerate(inputs):
            self.session.train_set_input(i, input_data)

        # Set expected outputs
        outputs: List[np.ndarray] = []
        for i, expected_data in enumerate(expecteds):
            self.session.train_set_expected(i, expected_data)
            output = np.zeros(expected_data.shape,
                              dtype=self.session.output_tensorinfo(i).dtype)
            self.session.train_set_output(i, output)
            outputs.append(output)

        # Run a single training step
        t_start: float = time.perf_counter()
        self.session.train(update_weights=update_weights)
        t_end: float = time.perf_counter()

        # Update loss
        losses = [self.session.train_get_loss(i) for i in range(len(expecteds))]

        # Update metrics
        metrics: Dict[str, float] = {}
        for m in self.metrics:
            m.update_state(outputs, expecteds)
            metrics[m.__class__.__name__] = m.result()
            m.reset_state()

        return {"loss": losses, "metrics": metrics, "time_ms": (t_end - t_start) * 1000}
