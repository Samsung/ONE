import numpy as np

from ..native import libnnfw_api_pybind
from ..native.libnnfw_api_pybind import optimizer as optimizer_type
from ..native.libnnfw_api_pybind import loss as loss_type
from ..common.basesession import BaseSession
from .metrics.registry import MetricsRegistry
from .losses import CategoricalCrossentropy, MeanSquaredError
from .optimizer import Adam, SGD
import time


# TODO: Support import checkpoint
class TrainSession(BaseSession):
    """
    Class for training and inference using nnfw_session.
    """
    def __init__(self, nnpackage_path, backends="train"):
        """
        Initialize the train session.

        Args:
            nnpackage_path (str): Path to the nnpackage file or directory.
            backends (str): Backends to use, default is "train".
        """
        load_start = time.perf_counter()
        super().__init__(
            libnnfw_api_pybind.experimental.nnfw_session(nnpackage_path, backends))
        load_end = time.perf_counter()
        self.total_time = {'MODEL_LOAD': (load_end - load_start) * 1000}
        self.train_info = self.session.train_get_traininfo()
        self.optimizer = None
        self.loss = None
        self.metrics = []

    def compile(self, optimizer, loss, metrics=[], batch_size=16):
        """
        Compile the session with optimizer, loss, and metrics.

        Args:
            optimizer (Optimizer): Optimizer instance.
            loss (Loss): Loss instance.
            metrics (list): List of metrics to evaluate during training.
            batch_size (int): Number of samples per batch.

        Raises:
            ValueError: If the number of metrics does not match the number of model outputs.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = [
            MetricsRegistry.create_metric(m) if isinstance(m, str) else m for m in metrics
        ]

        # Check if the number of metrics matches the number of outputs
        num_model_outputs = self.session.output_size()
        if 0 < len(self.metrics) != num_model_outputs:
            raise ValueError(
                f"Number of metrics ({len(self.metrics)}) does not match the number of model outputs ({num_model_outputs}). "
                "Please ensure one metric is provided for each model output.")

        # Set training information
        self.train_info.learning_rate = optimizer.learning_rate
        self.train_info.batch_size = batch_size
        self.train_info.loss_info.loss = self._map_loss_function_to_enum(loss)
        self.train_info.loss_info.reduction_type = loss.reduction
        self.train_info.opt = self._map_optimizer_to_enum(optimizer)
        self.train_info.num_of_trainable_ops = optimizer.nums_trainable_ops
        self.session.train_set_traininfo(self.train_info)

        # Print training parameters
        self._print_training_parameters()

        # Prepare session for training
        compile_start = time.perf_counter()
        self.session.train_prepare()
        compile_end = time.perf_counter()
        self.total_time["COMPILE"] = (compile_end - compile_start) * 1000

    def _map_loss_function_to_enum(self, loss_instance):
        """
        Maps a LossFunction instance to the appropriate enum value.

        Args:
            loss_instance (LossFunction): An instance of a loss function.

        Returns:
            train_loss: Corresponding enum value for the loss function.

        Raises:
            TypeError: If the loss_instance is not a recognized LossFunction type.
        """
        if isinstance(loss_instance, CategoricalCrossentropy):
            return loss_type.CATEGORICAL_CROSSENTROPY
        elif isinstance(loss_instance, MeanSquaredError):
            return loss_type.MEAN_SQUARED_ERROR
        else:
            raise TypeError(
                f"Unsupported loss function type: {type(loss_instance).__name__}. "
                "Supported types are CategoricalCrossentropy and MeanSquaredError.")

    def _map_optimizer_to_enum(self, optimizer_instance):
        """
        Maps an Optimizer instance to the appropriate enum value.

        Args:
            optimizer_instance (Optimizer): An instance of an optimizer.

        Returns:
            train_optimizer: Corresponding enum value for the optimizer.

        Raises:
            TypeError: If the optimizer_instance is not a recognized Optimizer type.
        """
        if isinstance(optimizer_instance, SGD):
            return optimizer_type.SGD
        elif isinstance(optimizer_instance, Adam):
            return optimizer_type.ADAM
        else:
            raise TypeError(
                f"Unsupported optimizer type: {type(optimizer_instance).__name__}. "
                "Supported types are SGD and Adam.")

    def _print_training_parameters(self):
        """
        Print the training parameters in a formatted way.
        """
        # Get loss function name
        loss_name = self.loss.__class__.__name__ if self.loss else "Unknown Loss"

        # Get reduction type name from enum value
        reduction_name = self.train_info.loss_info.reduction_type.name.lower().replace(
            "_", " ")

        # Get optimizer name
        optimizer_name = self.optimizer.__class__.__name__ if self.optimizer else "Unknown Optimizer"

        print("== training parameter ==")
        print(
            f"- learning_rate        = {f'{self.train_info.learning_rate:.4f}'.rstrip('0').rstrip('.')}"
        )
        print(f"- batch_size           = {self.train_info.batch_size}")
        print(
            f"- loss_info            = {{loss = {loss_name}, reduction = {reduction_name}}}"
        )
        print(f"- optimizer            = {optimizer_name}")
        print(f"- num_of_trainable_ops = {self.train_info.num_of_trainable_ops}")
        print("========================")

    def train(self, data_loader, epochs, validation_split=0.0, checkpoint_path=None):
        """
        Train the model using the given data loader.

        Args:
            data_loader: A data loader providing input and expected data.
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs to train.
            validation_split (float): Ratio of validation data. Default is 0.0 (no validation).
            checkpoint_path (str): Path to save or load the training checkpoint.
        """
        if self.optimizer is None or self.loss is None:
            raise RuntimeError(
                "The training session is not properly configured. "
                "Please call `compile(optimizer, loss)` before calling `train()`.")

        # Split data into training and validation
        train_data, val_data = data_loader.split(validation_split)

        # Timings for summary
        epoch_times = []

        # Training loop
        for epoch in range(epochs):
            message = [f"Epoch {epoch + 1}/{epochs}"]

            epoch_start_time = time.perf_counter()
            # Training phase
            train_loss, avg_io_time, avg_train_time = self._run_phase(train_data,
                                                                      train=True)
            message.append(f"Train time: {avg_train_time:.3f}ms/step")
            message.append(f"IO time: {avg_io_time:.3f}ms/step")
            message.append(f"Train Loss: {train_loss:.4f}")

            # Validation phase
            if validation_split > 0.0:
                val_loss, _, _ = self._run_phase(val_data, train=False)
                message.append(f"Validation Loss: {val_loss:.4f}")

                # Print metrics
                for metric in self.metrics:
                    message.append(f"{metric.__class__.__name__}: {metric.result():.4f}")
                    metric.reset_state()

            epoch_time = (time.perf_counter() - epoch_start_time) * 1000
            epoch_times.append(epoch_time)

            print(" - ".join(message))

            # Save checkpoint
            if checkpoint_path is not None:
                self.session.train_export_checkpoint(checkpoint_path)

        self.total_time["EXECUTE"] = sum(epoch_times)
        self.total_time["EPOCH_TIMES"] = epoch_times

        return self.total_time

    def _run_phase(self, data, train=True):
        """
        Run a training or validation phase.

        Args:
            data: Data generator providing input and expected data.
            train (bool): Whether to perform training or validation.

        Returns:
            float: Average loss for the phase.
        """
        total_loss = 0.0
        num_batches = 0

        io_time = 0
        train_time = 0

        for inputs, expecteds in data:
            # Validate batch sizes
            self._check_batch_size(inputs, self.train_info.batch_size, data_type="input")
            self._check_batch_size(expecteds,
                                   self.train_info.batch_size,
                                   data_type="expected")

            set_io_start = time.perf_counter()
            # Set inputs
            for i, input_data in enumerate(inputs):
                self.session.train_set_input(i, input_data)

            # Set expected outputs
            outputs = []
            for i, expected_data in enumerate(expecteds):
                expected = np.array(expected_data,
                                    dtype=self.session.output_tensorinfo(i).dtype)
                self.session.train_set_expected(i, expected)

                output = np.zeros(expected.shape,
                                  dtype=self.session.output_tensorinfo(i).dtype)
                self.session.train_set_output(i, output)
                assert i == len(outputs)
                outputs.append(output)

            set_io_end = time.perf_counter()

            # Run training or validation
            train_start = time.perf_counter()
            self.session.train(update_weights=train)
            train_end = time.perf_counter()

            # Accumulate loss
            batch_loss = sum(
                self.session.train_get_loss(i) for i in range(len(expecteds)))
            total_loss += batch_loss
            num_batches += 1

            # Update metrics
            if not train:
                for metric in self.metrics:
                    metric.update_state(outputs, expecteds)

            # Calculate times
            io_time += (set_io_end - set_io_start)
            train_time += (train_end - train_start)

        if num_batches > 0:
            return (total_loss / num_batches, (io_time * 1000) / num_batches,
                    (train_time * 1000) / num_batches)
        else:
            return (0.0, 0.0, 0.0)

    def _check_batch_size(self, data, batch_size, data_type="input"):
        """
        Validate that the batch size of the data matches the configured training batch size.

        Args:
            data (list of np.ndarray): The data to validate.
            batch_size (int): The expected batch size.
            data_type (str): A string to indicate whether the data is 'input' or 'expected'.

        Raises:
            ValueError: If the batch size does not match the expected value.
        """
        for i, array in enumerate(data):
            if array.shape[0] > batch_size:
                raise ValueError(
                    f"Batch size mismatch for {data_type} data at index {i}: "
                    f"batch size ({array.shape[0]}) does not match the configured "
                    f"training batch size ({batch_size}).")

    def train_step(self, inputs, expecteds):
        """
        Train the model for a single batch.

        Args:
            inputs (list of np.ndarray): List of input arrays for the batch.
            expecteds (list of np.ndarray): List of expected output arrays for the batch.

        Returns:
            dict: A dictionary containing loss and metrics values.
        """
        if self.optimizer is None or self.loss is None:
            raise RuntimeError(
                "The training session is not properly configured. "
                "Please call `compile(optimizer, loss)` before calling `train_step()`.")

        # Validate batch sizes
        self._check_batch_size(inputs, self.train_info.batch_size, data_type="input")
        self._check_batch_size(expecteds,
                               self.train_info.batch_size,
                               data_type="expected")

        # Set inputs
        for i, input_data in enumerate(inputs):
            self.session.train_set_input(i, input_data)

        # Set expected outputs
        outputs = []
        for i, expected_data in enumerate(expecteds):
            self.session.train_set_expected(i, expected_data)
            output = np.zeros(expected_data.shape,
                              dtype=self.session.output_tensorinfo(i).dtype)
            self.session.train_set_output(i, output)
            outputs.append(output)

        # Run a single training step
        train_start = time.perf_counter()
        self.session.train(update_weights=True)
        train_end = time.perf_counter()

        # Calculate loss
        losses = [self.session.train_get_loss(i) for i in range(len(expecteds))]

        # Update metrics
        metric_results = {}
        for metric in self.metrics:
            metric.update_state(outputs, expecteds)
            metric_results[metric.__class__.__name__] = metric.result()

        return {
            "loss": losses,
            "metrics": metric_results,
            "train_time": (train_end - train_start) * 1000
        }
