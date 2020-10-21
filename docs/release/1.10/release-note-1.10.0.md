# Release Note 1.10.0

## ONE Compiler

### Compiler supports more operations

- Dequantize,  UnidirectionalSequenceLSTM

### Changes

- New `--fold_dequantize` option in `one-optimize`
- New `--fuse_add_with_tconv` option in `one-optimize`
- Support `int16` quantization in `one-quantize`
- Test scripts are added for basic testing of one-cmds command line tools
- Bug fixes for one-cmds command line tools


## ONE Runtime

### Runtime backend operation support
  - ACL-CL backend: OneHot
  - CPU backend: FullyConnected for Float32 16x1 Block Sparsity

### Optimization
  - Speed up for ReduceSum, StrideSlice in CPU backend
