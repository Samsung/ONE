# PyTorch examples

Python example to produce PyTorch and ONNX files

## Prerequisite

- Python 3.X
- PyTorch nightly

## Directory Layout

```
tpem.py <- PyThorch Example Manager
pthdump.py <- .pth file dumper
examples/
  [EXAMPLE NAME]/
    __init__.py
```

## Folder naming convention

Follow python API name

## Install PyThorch nightly

Visit https://pytorch.org/get-started/locally/

## HOWTO: Generate a pbtxt from examples

```
$ python3 ptem.py [EXAMPLE NAME 1] [EXANMPE NAME 2] ...
```

## HOWTO: Dump pth file

```
$ python3 pthdump.py [.pth NAME 1] [.pth NAME 2] ...
```

## HOWTO: Add a new example

TBA
