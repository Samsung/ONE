# train_step_with_dataset.py - Single Step Training Example

## Purpose

- Load an `.nnpackage` model
- Load training dataset from `.npy` file
- Load corresponding labels from `.npy` file
- Create optimizer (Stochastic Gradient Descent or Adam)
- Create loss function (Mean Squared Error or Categorical Cross-Entropy)
- Run `session.train_step(...)` and report progress

## Example Dataset Preparation

Input and output (label) shapes should match the model's expected input and
output shapes. In this example, the model expects input shape `(1, 224, 224, 3)`
and output shape `(1, 112, 112, 32)`. To prepare a sample dataset and labels,
one can use the following code snippet to generate random data and save them as
`.npy` files:

```python
import numpy as np

# Create random dataset
data = np.random.rand(16, 224, 224, 3).astype(np.float32)
np.save('dataset.npy', data)

# Create random labels
labels = np.random.rand(16, 112, 112, 32).astype(np.float32)
np.save('labels.npy', labels)
```

## Running the Example

To run the training step example with the prepared dataset and labels, use the
following command:

```bash
python train_step_with_dataset.py \
    --nnpkg=<ONE-ROOT>/nnpackage/examples/v1.3.0/two_tflites/mv1.0.tflite \
    --input=dataset.npy \
    --label=labels.npy \
    --batch_size=16 \
    --data_length=1
```
