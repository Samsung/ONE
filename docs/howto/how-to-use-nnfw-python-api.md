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
    session = nnfw_session(nnpackage_path, backends)
```

2. Prepare Input/Output

```python
# Prepare input. Here we just allocate dummy input arrays.
input_size = session.input_size()
inputs = []

for i in range(input_size):
    # Get i-th input's info
    input_tensorinfo = session.input_tensorinfo(i)
    ti_dtype = input_tensorinfo.dtype

    input_array = [0.] * num_elems(input_tensorinfo)
    input_array = np.array(input_array, dtype=ti_dtype)
    # TODO: Please add initialization for your input.
    session.set_input(i, input_array)

    inputs.append(input_array)

# Prepare output
output_size = session.output_size()
outputs = []

for i in range(output_size):
    # Get i-th output's info
    output_tensorinfo = session.output_tensorinfo(i)
    ti_dtype = output_tensorinfo.dtype

    output_array = [0.] * num_elems(output_tensorinfo)
    output_array = np.array(output_array, dtype=ti_dtype)
    session.set_output(i, output_array)

    outputs.append(output_array)
```

3. Inference

```python
# Do inference
session.run()
```

## Run Inference with app on the target devices

reference app : [minimal-python app](https://github.com/Samsung/ONE/blob/master/runtime/onert/sample/minimal-python)

```
$ python3 minimal.py path_to_nnpackage_directory
```
