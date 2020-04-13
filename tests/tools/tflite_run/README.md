# tflite_run

A simple Tensorflow Lite runner. It measures the elapsed time and optionally dump the input/output tensors or verify them.

## Usage

### Simple run

This will run with random input data

```
$ ./tflite_run model.tflite
```

Output would look like:

```
input tensor indices = [0,]
Input image size is smaller than the size required by the model. Input will not be set.
output tensor indices = [308(max:984),]
Prepare takes 0.00126718 seconds
Invoke takes 7.09527 seconds
```

### Specifying input feature map

We can specify input feature map, but it only accepts preprocessed data which means that the image files must be converted.

TODO : Add input image preprocessing instruction

```
$ ./tflite_run model.tflite -i binary_input_file
```

### Dump the input and output tensors

Dump the input and output tensors to a file.
```
$ ./tflite_run model.tflite --dump golden
```

Why we do this is usually for later verification. The tensors are written to name "golden".

### Compare with the saved outputs

The result from `tflite_run` and binary file are compared with `--compare` option.

```
$ ls golden
golden
$ ./tflite_run model.tflite --compare golden
```

The output would look like:

```
input tensor indices = [0,]
Input image size is smaller than the size required by the model. Input will not be set.
output tensor indices = [308(max:984),]
Prepare takes 0.00126718 seconds
Invoke takes 7.09527 seconds
========================================
Comparing the results with "golden2".
========================================
  Tensor #308: UNMATCHED
    1 diffs are detected
    Max absolute diff at [0, 0]
       expected: 99
       obtained: 0.000139008
       absolute diff: 98.9999
    Max relative diff at [0, 1007]
       expected: 7.01825e-33
       obtained: 0.000139011
       relative diff: 1
         (tolerance level = 8.38861e+06)
```

If `--compare` option is on, the exit code will be depend on its compare result. 0 for matched, other number for unmatched.

## How Verification Works

For verification, we may follow these steps:

1. Generate and store the verfication data (run with option `--dump`)
    1. Input Tensor does not matter as we will keep inputs along with outputs
    1. Interpreter.Invoke()
    1. Dump input tensors and output tensors to a file
1. Give the dumped file for other runtime that we want to verify (run with option `--compare`)
    1. Set interpreter's input to input tensor data from the file
    1. Interpreter.Invoke()
    1. Compare the results with output tensor data from the file
