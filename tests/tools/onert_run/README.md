# onert_run

A simple ONERT runner. It measures the elapsed time and optionally dump the input/output tensors or verify them.

## Usage

### Simple run

This will run with random input data

```
$ ./onert_run model.tflite
```

Output would look like:

```
===================================
MODEL_LOAD   takes 201.058 ms
PREPARE      takes 40.116 ms
EXECUTE      takes 17.543 ms
- MEAN     :  17.543 ms
- MAX      :  17.543 ms
- MIN      :  17.543 ms
- GEOMEAN  :  17.543 ms
===================================
```

### Specifying input feature map (TBD)

We can specify input feature map, but it only accepts preprocessed data which means that the image files must be converted.

```
$ ./onert_run model.tflite -i binary_input_file
```

### Dump the input and output tensors (TBD)

Dump the input and output tensors to a file.
```
$ ./onert_run model.tflite --dump golden
```

Why we do this is usually for later verification. The tensors are written to name "golden".

### Compare with the saved outputs (TBD)

The result from `onert_run` and binary file are compared with `--compare` option.

```
$ ls golden
golden
$ ./onert_run model.tflite --compare golden
```

TBD

## How Verification Works (TBD)

For verification, we may follow these steps:

1. Generate and store the verfication data (run with option `--dump`)
    1. Input Tensor does not matter as we will keep inputs along with outputs
    1. Run onert
    1. Dump input tensors and output tensors to a file
1. Give the dumped file for other runtime that we want to verify (run with option `--compare`)
    1. Set input to input tensor data from the file
    1. Run onert
    1. Compare the results with output tensor data from the file
