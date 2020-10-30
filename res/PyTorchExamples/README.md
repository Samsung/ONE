# PyTorch examples

Python example to convert PyTorch to ONNX/TF/tflite models

## Package versions

- Python 3.X
- torch==1.7.0
- onnx=1.7.0
- onnx-tf==1.6.0 (see note)
- tensorflow-cpu==2.3.0
- tensorflow-addons

note: please install `onnx-tf` from master branch to use TensorFlow 2.x
downloaded from https://github.com/onnx/onnx-tensorflow

## Directory Layout

```
tpem.py     <- PyThorch Example Manager
examples/
  [EXAMPLE NAME]/
    __init__.py
```

## Folder naming convention

Follow python API name

## HOWTO: Generate a tflite from examples

```
$ python3 ptem.py [EXAMPLE NAME 1] [EXANMPE NAME 2] ...
```

## HOWTO: Add a new example

- create a folder name same as python API name
- add `__init__.py` file
- set `_model_` variable holding model of the network containing the operator
- set `_dummy_` variable holding a dummy input for generating ONNX file
