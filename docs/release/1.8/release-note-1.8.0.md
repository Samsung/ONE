# Release Note 1.8.0

## Feature Highlights

- **ONE** Compiler
    - Support new command line interface

- **ONE** Runtime
    - CPU backend supports 7 more operations
    - CPU backend supports 9 more quant8 operations

## ONE Compiler

### New command line interface for user interface consistancy

- `one-import-bcq` : import BCQ(Binary coding quantized) TensorFlow model
- Commands now support `--version` option to show version number

### Changes

- Experimental support for TensorFlow 2.x has updated to 2.3.0 (TensorFlow 1.3.2 is our official support version)
- Support more operators in luci-interpreter
- Enhancing one-quantizer

## ONE Runtime

### Rename headers

- Rename `nnfw_dev.h` to `nnfw_experimental.h`

### Optimization

- Remove copies for model input/outputs whenever possible

### Support CPU backend operation

- BatchToSpaceND, L2Normalization, ReLU6, ResizeBilinear, SpaceToDepth, SplitV, StatelessRandomUniform

### Support CPU backend quant8 operation

- BatchToSpaceND, L2Normalization, Pad, PadV2, ResizeBilinear, Slice, Quantize, SpaceToDepth, Sum

