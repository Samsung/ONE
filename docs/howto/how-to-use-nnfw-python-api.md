# How to Use NNFW PYTHON API

## Prepare nnpackage

### Use nnpackage examples

Use the [nnpackage examples](https://github.com/Samsung/ONE/tree/master/nnpackage/examples/v1.0.0) to run tutorial code.

## Install nnfw python API

Please see [nnfw python api](https://github.com/SAMSUNG/ONE/tree/master/infra/nnfw/python) for installing nnfw python api.

1. Initialize nnfw_session

```python
# Create session and load nnpackage
# operations is optional to assign a specific backends to each operation.
# The default value of backends is "cpu".
if operations:
    session = nnfw_session(nnpackage_path, backends, operations)
else:
    # NOTE: This Python API is experimental yet. It can be changed later.
    session = nnfw_session(nnpackage_path, backends)
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

reference app : [minimal-python app](https://github.com/Samsung/ONE/blob/master/runtime/onert/sample/minimal-python)

```
$ python3 minimal.py path_to_nnpackage_directory
```
