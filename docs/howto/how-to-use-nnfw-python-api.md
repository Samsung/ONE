# How to Use NNFW PYTHON API

## Prepare nnpackage

### Convert tensorflow pb file to nnpackage

Follow the [compiler guide](https://github.com/Samsung/ONE/blob/master/docs/nncc/v1.0.0/tutorial.md) to generate nnpackge from tensorflow pb file

### Convert tflite file to nnpackage

Please see [model2nnpkg](https://github.com/Samsung/ONE/tree/master/tools/nnpackage_tool/model2nnpkg) for converting from tflite model file.

## Build app with NNFW PYTHON API

Here are basic steps to build app with [NNFW PYTHON API](https://github.com/Samsung/ONE/blob/master/runtime/onert/python/api)

1. Initialize nnfw_session

```python
# Create session and Loading nnpackage
# OPERATION is optional for assigning a specific backend to operations.
# "cpu" is default value of BACKEND.
if OPERATION: session = nnfw_session(NNPACKAGE_PATH, BACKEND, OPERATION)
else: session = nnfw_session(NNPACKAGE_PATH, BACKEND)
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

reference app : [minimal app](https://github.com/Samsung/ONE/blob/master/runtime/onert/sample/minimal)

```
$ python3 minimal.py path_to_nnpackage_directory
```
