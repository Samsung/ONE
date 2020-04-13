# nnkit

`nnkit` is collection of neural networks tools for our _nncc_ project.
This tool is mostly used for testing.

# Purpose
For testing, we need to have
- a tool to run existing framework such as Tensorflow for expected tensor result --- (1)
- a tool to run our implementation for actual tensor result --- (2)

`nnkit` provides a flexible framework to get expected and actual result.

# Design

## Requirements to address:

- Input
  - Same randomized input is used for both of (1) and (2)
  - Expect tensor layout (e.g., NHWC) could be different for (1) and (2)
- Input and output format
  - Results of (1) and (2) have same file format and data format

For (1), `nnkit` designed to enable the following:
- Input of `nnkit` is randomized and saved into a file in a specific format
- Existing framework such as Tensorflow can run with input tensors that is properly translated
- Result is written into a file in a specific format

For (2), `nnkit` designed to enable the following:
- Data of `nnkit` in a file by (1) is used as input
- Our implementation can run with input tensors that is properly translated
- Result is written into a file in a specific format

## `nnkit-run`

`nnkit-run` is a command line interface to interact with existing inference engines
or compiled artifacts.

## How `nnkit-run` works

`nnkit-run` first dynamically loads `backend` and multiple pre/post `action`
specified by command-line. After loading  backend and actions, `nnkit-run` requests
`backend` to prepare itself. When backend is prepared, `backend` exposes its
internal state to `nnkit-run` (as `nnkit::TensorContext`).
`nnkit-run` takes this state, and passes it to registered pre `action`(s).
Each action may read tensor(s) (e.g. dump the content into a file),
or manipulate their value (e.g. fill random values).
`nnkit-run` then invokes `backend` through `run()` method.
After successful running the `backend`, post `action`(s) are called same like
pre `action`(s) as a teardown step.

## Backends

In 2019 there will be the following backends as of writing this document

- Backends for the existing framework:
  - Caffe as `libnnkit_caffe_backend.so`
  - Tensorflow Lite as `libnnkit_tflite_backend.so`
  - Tensorflow as `libnnkit_tf_backend.so`
  - Onnx as `libnnkit_onnx_backend.so`

- Backends for our implementation:
  - Moco Tensorflow (TBD)
  - Moco Onnx (TBD)

# How to use

## How to run inference with nnkit-run

To run `nnkit-run`, we need to provide a backend module and argument(s) if required
and optional `pre-` or `post-` action module(s)

## How to pass arguments

Syntax is `--argument` with `value` form. Existing arguments are as follows.
- `--backend` [Backend module path]. Only one is needed.
- `--backend-arg` [Backend argument]. Argument(s) for the backend.
- `--pre` [Pre-Action module path]. Multiple Pre-Action can be given.
- `--pre-arg` [Pre-Action argument]. Set argument(s) for the pre-action just before.
- `--post` [Post-Action module path]. Multiple Post-Action can be given.
- `--post-arg` [Post-Action argument]. Set argument(s) for the post-action just before.

For example,
```
nnkit-run \
--backend ./path/to/backend --backend-arg arg1 --backend-arg arg2 \
--pre ./path/to/preA --pre-arg arg1preA --pre-arg arg2preA \
--pre ./path/to/preB --pre-arg arg1preB --pre-arg arg2preB \
--post ./path/to/postA --post-arg arg1postA
```

This will run
- backend `./path/to/backend` with arguments `arg1 arg2` with
   - pre-action `./path/to/preA` with arguments `arg1preA arg2preA`,
   - pre-action `./path/to/preB` with arguments `arg1preB arg2preB` and
   - post-action `./path/to/postA` with an argument `arg1postA`

## Example : Running with Tensorflow backend

To run Tensorflow backend, you need two parameters: model file in protobuf format (`pb` file) and input/output tensor information such as tensor name, data type, shape. Please refer to `test.info` files under `moco/test/tf`.

```
cd build

compiler/nnkit/tools/run/nnkit-run \
--backend ./compiler/nnkit-tf/backend/libnnkit_tf_backend.so \
--backend-arg inceptionv3_non_slim_2015.pb \
--backend-arg inceptionv3_non_slim_2015.info
```

## Example: Running with Onnx backend
TBD

## Example : Running with tflite backend

```
cd build

compiler/nnkit/tools/run/nnkit-run \
--backend ./compiler/nnkit-tflite/backend/libnnkit_tflite_backend.so \
--backend-arg inceptionv3_non_slim_2015.tflite
```

## Example: Running with Caffe backend

Running with caffe backend is similar to running with tflite, except that you need to provide `prototxt` file, `caffemodel` is not necessary, unless you want to use specific weights (weights are random if `caffemodel` is not provided and `prototxt` is not filled with specific weights):

```
cd build

compiler/nnkit/tools/run/nnkit-run \
--backend ./compiler/nnkit-caffe/backend/libnnkit_caffe_backend.so \
--backend-arg inception_v3.prototxt
```

## Running with pre & post actions

The above command for the tflite backend shows nothing except `nnapi error: unable to open library libneuralnetworks.so` warning even though running correctly. The following command displays inferenced values.
```
cd build

compiler/nnkit/tools/run/nnkit-run \
--backend ./compiler/nnkit-tflite/backend/libnnkit_tflite_backend.so \
--backend-arg inceptionv3_non_slim_2015.tflite \
--post ./compiler/nnkit/actions/builtin/libnnkit_show_action.so
```

The following command initializes input tensors with random values generated by `RandomizeAction` pre-action.
```
compiler/nnkit/tools/run/nnkit-run \
--backend ./compiler/nnkit-tflite/backend/libnnkit_tflite_backend.so \
--backend-arg inceptionv3_non_slim_2015.tflite \
--pre ./compiler/nnkit/actions/builtin/libnnkit_randomize_action.so \
--post ./compiler/nnkit/actions/builtin/libnnkit_show_action.so
```

## Example: Dump HDF5

You can drop a HDF5 file of inputs and outputs with `HDF5_export_action` action module.

```
cd build

compiler/nnkit/tools/run/nnkit-run \
--backend ./compiler/nnkit-tflite/backend/libnnkit_tflite_backend.so \
--backend-arg inceptionv3_non_slim_2015.tflite \
--pre ./compiler/nnkit/actions/builtin/libnnkit_randomize_action.so  \ # randomize first
--pre ./compiler/nnkit/actions/HDF5/libnnkit_HDF5_export_action.so \   # then drop input in HDF5 format
--pre-arg ./pre.hdf5 \
--post ./compiler/nnkit/actions/HDF5/libnnkit_HDF5_export_action.so \  # drop output in HDF5 format
--post-arg ./post.hdf5
```

This will drop `pre.hdf5` and `post.hdf5` files containing input and output
tensor of inceptionv3_non_slim_2015.tflite model.

# To do
- `nnkit` backend for `moco` Tensorflow frontend
- `nnkit` backend for `moco` Onnx frontend
- `nnkit` backend for Onnx frontend
