# How to Use onert Python API

## CAUTION

This Python API is experimental yet. It can be changed later.

## Prepare nnpackage

### Use nnpackage examples

Use the [nnpackage examples](https://github.com/Samsung/ONE/tree/master/nnpackage/examples/v1.0.0) to run tutorial code.

## Install onert Python API

You can install the package as follows:

```
$ pip install onert
```

By specifying the version, you can use a specific version of the package.

```
$ pip install onert==0.1.0
```

You can install latest developing package as follows:

```
$ pip install onert --pre
```

This definition has to be set at the top of the script using onert Python API.

```
import onert
```

Or you can import the onert module directly.

```
from onert.infer import *
```

This can be use onert session directly.


## Example Code

1. Initialize session

```python
import onert

# Create session and load nnpackage or modelfile
# The default value of backends is "cpu".
session = onert.infer.session(path, backends)
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
