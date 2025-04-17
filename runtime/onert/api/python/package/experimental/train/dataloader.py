import os
import numpy as np
from typing import List, Tuple, Union, Optional, Any, Iterator


class DataLoader:
    """
    A flexible DataLoader to manage training and validation data.
    Automatically detects whether inputs are paths or NumPy arrays.
    """
    def __init__(self,
                 input_dataset: Union[List[np.ndarray], np.ndarray, str],
                 expected_dataset: Union[List[np.ndarray], np.ndarray, str],
                 batch_size: int,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 expected_shape: Optional[Tuple[int, ...]] = None,
                 dtype: Any = np.float32) -> None:
        """
        Initialize the DataLoader.

        Args:
            input_dataset (list of np.ndarray | np.ndarray | str):
                List of input arrays where each array's first dimension is the batch dimension,
                or a single NumPy array, or a file path.
            expected_dataset (list of np.ndarray | np.ndarray | str):
                List of expected arrays where each array's first dimension is the batch dimension,
                or a single NumPy array, or a file path.
            batch_size (int): Number of samples per batch.
            input_shape (tuple[int, ...], optional): Shape of the input data if raw format is used.
            expected_shape (tuple[int, ...], optional): Shape of the expected data if raw format is used.
            dtype (type, optional): Data type of the raw file (default: np.float32).
        """
        self.batch_size: int = batch_size
        self.inputs: List[np.ndarray] = self._process_dataset(input_dataset, input_shape,
                                                              dtype)
        self.expecteds: List[np.ndarray] = self._process_dataset(
            expected_dataset, expected_shape, dtype)
        self.batched_inputs: List[List[np.ndarray]] = []

        # Verify data consistency
        self.num_samples: int = self.inputs[0].shape[0]  # Batch dimension
        if self.num_samples != self.expecteds[0].shape[0]:
            raise ValueError(
                "Input data and expected data must have the same number of samples.")

        # Precompute batches
        self.batched_inputs, self.batched_expecteds = self._create_batches()

    def _process_dataset(self,
                         data: Union[List[np.ndarray], np.ndarray, str],
                         shape: Optional[Tuple[int, ...]],
                         dtype: Any = np.float32) -> List[np.ndarray]:
        """
        Process a dataset or file path.

        Args:
            data (str | np.ndarray | list[np.ndarray]): Path to file or NumPy arrays.
            shape (tuple[int, ...], optional): Shape of the data if raw format is used.
            dtype (type, optional): Data type for raw files.

        Returns:
            list[np.ndarray]: Loaded or passed data as NumPy arrays.
        """
        if isinstance(data, list):
            # Check if all elements in the list are NumPy arrays
            if all(isinstance(item, np.ndarray) for item in data):
                return data
            raise ValueError("All elements in the list must be NumPy arrays.")
        if isinstance(data, np.ndarray):
            # If it's already a NumPy array and is not a list of arrays
            if data.ndim > 1:
                # If the array has multiple dimensions, split it into a list of arrays
                return [data[i] for i in range(data.shape[0])]
            else:
                # If it's a single array, wrap it into a list
                return [data]
        elif isinstance(data, str):
            # If it's a string, assume it's a file path
            return [self._load_data(data, shape, dtype)]
        else:
            raise ValueError("Data must be a NumPy array or a valid file path.")

    def _load_data(self,
                   file_path: str,
                   shape: Optional[Tuple[int, ...]],
                   dtype: Any = np.float32) -> np.ndarray:
        """
        Load data from a file, supporting both .npy and raw formats.

        Args:
            file_path (str): Path to the file to load.
            shape (tuple[int, ...], optional): Shape of the data if raw format is used.
            dtype (type, optional): Data type of the raw file (default: np.float32).

        Returns:
            np.ndarray: Loaded data as a NumPy array.
        """
        _, ext = os.path.splitext(file_path)

        if ext == ".npy":
            # Load .npy file
            return np.load(file_path)
        elif ext in [".bin", ".raw"]:
            # Load raw binary file
            if shape is None:
                raise ValueError(f"Shape must be provided for raw file: {file_path}")
            return self._load_raw(file_path, shape, dtype)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_raw(self, file_path: str, shape: Tuple[int, ...], dtype: Any) -> np.ndarray:
        """
        Load raw binary data.

        Args:
            file_path (str): Path to the raw binary file.
            shape (tuple[int, ...]): Shape of the data to reshape into.
            dtype (type): Data type of the binary file.

        Returns:
            np.ndarray: Loaded data as a NumPy array.
        """
        # Calculate the expected number of elements based on the provided shape
        expected_elements: int = int(np.prod(shape))

        # Calculate the expected size of the raw file in bytes
        expected_size: int = expected_elements * np.dtype(dtype).itemsize

        # Get the actual size of the raw file
        actual_size: int = os.path.getsize(file_path)

        # Check if the sizes match
        if actual_size != expected_size:
            raise ValueError(
                f"Raw file size ({actual_size} bytes) does not match the expected size "
                f"({expected_size} bytes) based on the provided shape {shape} and dtype {dtype}."
            )

        # Read and load the raw data
        with open(file_path, "rb") as f:
            data = f.read()
        array = np.frombuffer(data, dtype=dtype)
        if array.size != expected_elements:
            raise ValueError(
                f"Raw data size does not match the expected shape: {shape}. "
                f"Expected {expected_elements} elements, got {array.size} elements.")
        return array.reshape(shape)

    def _create_batches(self) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        Precompute batches for inputs and expected outputs.

        Returns:
            tuple: Lists of batched inputs and batched expecteds.
        """
        batched_inputs: List[List[np.ndarray]] = []
        batched_expecteds: List[List[np.ndarray]] = []

        for batch_start in range(0, self.num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_samples)

            # Collect batched inputs
            inputs_batch = [
                input_array[batch_start:batch_end] for input_array in self.inputs
            ]
            if batch_end - batch_start < self.batch_size:
                # Resize the last batch to match batch_size
                inputs_batch = [
                    np.resize(batch, (self.batch_size, *batch.shape[1:]))
                    for batch in inputs_batch
                ]

            batched_inputs.append(inputs_batch)

            # Collect batched expecteds
            expecteds_batch = [
                expected_array[batch_start:batch_end] for expected_array in self.expecteds
            ]
            if batch_end - batch_start < self.batch_size:
                # Resize the last batch to match batch_size
                expecteds_batch = [
                    np.resize(batch, (self.batch_size, *batch.shape[1:]))
                    for batch in expecteds_batch
                ]

            batched_expecteds.append(expecteds_batch)

        return batched_inputs, batched_expecteds

    def __iter__(self) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Make the DataLoader iterable.

        Returns:
            self
        """
        self.index = 0
        return self

    def __next__(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return the next batch of data.

        Returns:
            tuple: (inputs, expecteds) for the next batch.
        """
        if self.index >= len(self.batched_inputs):
            raise StopIteration

        # Retrieve precomputed batch
        input_batch = self.batched_inputs[self.index]
        expected_batch = self.batched_expecteds[self.index]

        self.index += 1
        return input_batch, expected_batch

    def split(self, validation_split: float) -> Tuple["DataLoader", "DataLoader"]:
        """
        Split the data into training and validation sets.

        Args:
            validation_split (float): Ratio of validation data. Must be between 0.0 and 1.0.

        Returns:
            tuple: Two DataLoader instances, one for training and one for validation.
        """
        if not (0.0 <= validation_split <= 1.0):
            raise ValueError("Validation split must be between 0.0 and 1.0.")

        split_index = int(len(self.inputs[0]) * (1.0 - validation_split))

        train_inputs = [input_array[:split_index] for input_array in self.inputs]
        val_inputs = [input_array[split_index:] for input_array in self.inputs]
        train_expecteds = [
            expected_array[:split_index] for expected_array in self.expecteds
        ]
        val_expecteds = [
            expected_array[split_index:] for expected_array in self.expecteds
        ]

        train_loader = DataLoader(train_inputs, train_expecteds, self.batch_size)
        val_loader = DataLoader(val_inputs, val_expecteds, self.batch_size)

        return train_loader, val_loader
