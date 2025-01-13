# How to Use NNFW PYTHON API

## CAUTION

This Python API is experimental yet. It can be changed later.

## Prepare nnpackage

### Use nnpackage examples

Use the [nnpackage examples](https://github.com/Samsung/ONE/tree/master/nnpackage/examples/v1.0.0) to run tutorial code.

## Install nnfw python API

Please see [nnfw python api](https://github.com/SAMSUNG/ONE/tree/master/infra/nnfw/python) for installing nnfw python api.

1. Initialize nnfw_session

```python
import onert

# Create session and load nnpackage
# The default value of backends is "cpu".
session = onert.infer.session(nnpackage_path, backends)
```

2. Prepare Input

```python
# Prepare input. Here we just allocate dummy input arrays.
input_size = session.input_size()
session.set_inputs(input_size)
```

3. Inference

```python
# Do inference
outputs = session.inference()
```

## Run Inference with app on the target devices

reference app : [minimal-python app](https://github.com/Samsung/ONE/blob/master/runtime/onert/sample/minimal-python/infer)

```
$ python3 minimal.py path_to_nnpackage_directory
```

## Experimental API

### Train with dataset

1. Import the Module and Initialize TrainSession

```python
import onert

# Create a training session and load the nnpackage
# Default backends is set to "train".
session = onert.experimental.train.session(nnpackage_path, backends="train")
```

2. Prepare Input and Output Data

```python
# Create a DataLoader

from onert.experimental.train import DataLoader

# Define the paths for input and expected output data
input_path = "path/to/input_data.npy"
expected_path = "path/to/expected_data.npy"

# Define batch size
batch_size = 16

# Initialize DataLoader
data_loader = DataLoader(input_dataset=input_path,
                         expected_dataset=expected_path,
                         batch_size=batch_size)
```

3. Compile the Session

```python
# Set Optimizer, Loss, and Metrics

from onert.experimental.train import optimizer, losses, metrics

# Define optimizer
optimizer_fn = optimizer.Adam(learning_rate=0.01)

# Define loss function
loss_fn = losses.CategoricalCrossentropy()

# Define metrics
metric_list = [metrics.CategoricalAccuracy()]

# Compile the training session
session.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metric_list, batch_size=batch_size)
```

4. Train the Model

```python
# Train and Validate

# Train the model
session.train(data_loader=data_loader,
              epochs=5,
              validation_split=0.2,
              checkpoint_path="checkpoint.ckpt")
```

5. Train one step with data loader (Optional)

```python
for batch_idx, (inputs, expecteds) in enumerate(data_loader):
    # Train on a single step
    results = sess.train_step(inputs, expecteds)
```

### Custom Metric

You can use custom metrics instread of provided metrics

```python
from onert.experimental.train import metrics

class CustomMeanAbsoluteError(Metric):
    """
    Custom metric to calculate the mean absolute error (MAE) between predictions and ground truth.
    """
    def __init__(self):
        self.total_absolute_error = 0.0
        self.total_samples = 0

    def update_state(self, outputs, expecteds):
        """
        Update the metric's state based on the outputs and expected values.

        Args:
            outputs (list of np.ndarray): List of model outputs.
            expecteds (list of np.ndarray): List of expected (ground truth) values.
        """
        for output, expected in zip(outputs, expecteds):
            self.total_absolute_error += np.sum(np.abs(output - expected))
            self.total_samples += expected.size

    def result(self):
        """
        Calculate and return the current mean absolute error.

        Returns:
            float: The mean absolute error.
        """
        return self.total_absolute_error / self.total_samples if self.total_samples > 0 else 0.0

    def reset_state(self):
        """
        Reset the metric's state for the next epoch.
        """
        self.total_absolute_error = 0.0
        self.total_samples = 0

# Add the custom metric to the list
metric_list = [
    CustomMeanAbsoluteError()
]

# Compile the session with the custom metric
session.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metric_list, batch_size=batch_size)
```

### Run Train with dataset on the target devices
reference app : [minimal-python app](https://github.com/Samsung/ONE/blob/master/runtime/onert/sample/minimal-python/experimental/)
