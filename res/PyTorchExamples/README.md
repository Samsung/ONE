# PyTorch examples

Python example to convert PyTorch/Caffe to ONNX/TF/tflite models

## Package versions

- Python 3.X
- PyTorch 1.7
- ONNX 1.7.0
- ONNX-TensorFlow 1.6.0
- TensorFlow 2.3.0

## Directory Layout

```
c2em.py     <- Caffe2 Example Manager
tpem.py     <- PyThorch Example Manager
pthdump.py  <- .pth file dumper
examples/
  [EXAMPLE NAME]/
    __init__.py
c2examples/
  [EXAMPLE NAME]/
    __init__.py
```

## Folder naming convention

Follow python API name

## Install PyThorch nightly

Visit https://pytorch.org/get-started/locally/

## HOWTO: Generate a tflite from examples

```
$ python3 ptem.py [EXAMPLE NAME 1] [EXANMPE NAME 2] ...
```

for Caffe2 examples;
```
$ python3 c2em.py [EXAMPLE NAME 1] [EXANMPE NAME 2] ...
```


## HOWTO: Dump pth file

```
$ python3 pthdump.py [.pth NAME 1] [.pth NAME 2] ...
```

## HOWTO: Add a new example

For Pytorch examples;
- create a folder name same as python API name
- add `__init__.py` file
- set `_model_` variable holding model of the network containing the operator
- set `_dummy_` variable holding a dummy input for generating ONNX file

For Caffe2 examples;
- create a folder name same as python API name
- add `__init__.py` file
- set `_model_` variable holding model of the network containing the operator
- set `_model_init_` variable holding initial network
- set `_dummy_` variable holding a dummy input for generating ONNX file
