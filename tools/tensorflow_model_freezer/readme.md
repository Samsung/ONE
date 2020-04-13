## Prerequisites

The scripts here use TensorFlow's tools, so you need an environment to run TensorFlow. Running the scripts within this tutorial requires:

* Install [TensorFlow](https://www.tensorflow.org/install/) v1.12 or later
    * Use pip
    ```
    $ pip install tensorflow==1.12
    ```

## What this tool is about

This tool generaes the following files:
1. __Tensorflow model__ files in *.pb, *.pbtxt
1. Tensorflow model after __freezing__ in *.pb, *.pbtxt
1. __Tensorboard__ log file to visually see the above 1 and 2.
1. __TFLITE__ file after running TOCO

By define `Test Cases`, you can easily and quickly generate files for various ranks of operands.

## How to use

- Copy `MUL_gen.py` or `TOPK_gen.py` and modify for your taste.
  - Note that `TOPK_gen.py` fails while generating TFLITE file since TOCO does not support `TOPK` oeration.

- Run `~/nnfw$ PYTHONPATH=$PYTHONPATH:./tools/tensorflow_model_freezer/ python tools/tensorflow_model_freezer/sample/MUL_gen.py  ~/temp`
  - Files will be generated under `~/temp`

## How to run
```
$ chmod +x tools/tensorflow_model_freezer/sample/name_of_this_file.py
$ PYTHONPATH=$PYTHONPATH:./tools/tensorflow_model_freezer/ \
      tools/tensorflow_model_freezer/sample/name_of_this_file.py \
      ~/temp  # directory where model files are saved
```

## Note
- This tool is tested with Python 2.7 and 3
