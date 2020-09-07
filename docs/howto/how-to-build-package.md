# How to Build Package

## Overview

This document describes how to build a Package to run the model in our runtime
_onert_ that consists of model and additional file(s). Users can build a
package through command line tools.

Steps of building a Package:
1. Import model and convert to circle
1. Optionally, optimize and quantize circle
3. Create package from circle

NOTE: Examples and options of each commands shown are from the version of
writing this document. They may differ from latest version of commands.
Please fire an issue or post a PR to correct them if anything needs update.

## Import model

Currently TensorFlow and TensorFlow lite models are supported as of writing
this document.

To import a model, use `one-import` with a model framework key and arguments.
```
$ one-import FRAMEWORK [arguments]
```

Execute `one-import` without any key will show possible frameworks.

Example of `one-import` command:
```
$ one-import
Usage: one-import [FRAMEWORK] ...
Available FRAMEWORK drivers:
  bcq
  tf
  tflite
```

### Example for TensorFlow

This is an example to import TensorFlow model:
```
$ one-import tf --input_path mymodel.pb --output_path mymode.circle \
--input_arrays input1,input2 --input_shapes "1,224,224,3:1000" \
--output_arrays output
```

Run with `--help` will show current required/optional arguments:
```
$ one-import tf --help
Convert TensorFlow model to circle.
Usage: one-import-tf
    --version Show version information and exit
    --input_path <path/to/tfmodel>
    --output_path <path/to/circle>
    --input_arrays <names of the input arrays, comma-separated>
    --input_shapes <input shapes, colon-separated>
    --output_arrays <names of the output arrays, comma-separated>
    --v2 Use TensorFlow 2.x interface (default is 1.x interface)
```

### Example for TensorFlow lite

This is an example to import TensorFlow lite model:
```
$ one-import tflite --input_path mymodel.tflite --output_path mymodel.circle
```

As like, run with `--help` will show current required/optional arguments:
```
$ one-import tflite --help
Convert TensorFlow lite model to circle.
Usage: one-import-tflite
    --version Show version information and exit
    --input_path <path/to/tflitemodel>
    --output_path <path/to/circle
```

### Example for TensorFlow BCQ model

TBD

## Optimize circle model

circle model can be optimized to run faster and make the model smaller.
Typical optimization algorithm for this is to fuse some patterns of operators
to one fused operator.

This is an example to optimize circle model:
```
$ one-optimize --all --input_path mymodel.circle --output_path optmodel.circle
```

Run with `--help` will show current optimization options:
```
$ one-optimize --help
Optimize circle model.
Usage: one-optimize
    --version       Show version information and exit
    --all           Enable all optimization algorithms
    --fuse_bcq      Enable FuseBCQ Pass
    --fuse_instnorm Enable FuseInstanceNormalization Pass
    --resolve_customop_add
                    Enable ResolveCustomOpAddPass Pass
    --resolve_customop_batchmatmul
                    Enable ResolveCustomOpBatchMatMulPass Pass
    --resolve_customop_matmul
                    Enable ResolveCustomOpMatMulPass Pass
    --input_path <path/to/input/circle>
    --output_path <path/to/output/circle>
```

## Quantize circle model

circle model can be quantized to run faster and make smaller, by reducing data
bits representing weight values to 8 or 16bits.

This is an example to quantize circle model:
```
$ one-quantize --input_path mymodel.circle --output_path quantmodel.circle
```

Like wise, `--help` will show current quantization options:
```
$ one-quantize --help
Quantize circle model.
Usage: one-quantize
    --version         Show version information and exit
    --input_dtype     Input data type (supported: float32, default=float32)
    --quantized_dtype Output quantized data type (supported: uint8, default=uint8)
    --granularity     Quantize granularity (supported: layer, channel, default=layer)
    --min_percentile  Minimum percentile (0.0~100.0, default=1.0)
    --max_percentile  Maximum percentile (0.0~100.0, default=99.0)
    --mode            Record mode (supported: percentile/moving_average, default=percentile)
    --input_path <path/to/input/circle>
    --input_data <path/to/input/data>
    --output_path <path/to/output/circle>
```

## Pack circle model

Use `one-pack` command to create package.

```
$ one-pack -i mymodel.circle -o nnpackage
```

`nnpackage` is a folder containing circle model and addition file(s)

```
$ tree nnpackage
nnpackage
└── mymodel
    ├── metadata
    │   └── MANIFEST
    └── mymodel.circless
```

Likewise, `--help` will show current package options:

```
$ one-pack --help
Package circle to nnpkg
Usage: one-pack
    -v, --version Show version information and exit
    -i <path/to/circle>
    -o <path/to/nnpackage/folder>
```
