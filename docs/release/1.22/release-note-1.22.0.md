# Release Note 1.22.0

## ONE Compiler

- Introduce new optimization options: `unroll_unidirseqlstm`, `forward_transpose_op`, `fold_fully_connected`, `fuse_prelu`
- Support more Ops for fake quantization: `Depth2Space`, `Space2Depth`, `Pack`, `Unpack`, `Abs`
- Support more Ops for quantization: `Abs`, `ReduceProd`
- Introduce _visq_ tool for quantization error visualization
- Introduce _Environment_ section into configuration file
- Improve speed of `convert_nchw_to_nhwc` option
- Support `Add`, `Mul` of index-type (int32, int64) tensors in _one-quantize_
- Support ubuntu 20.04
