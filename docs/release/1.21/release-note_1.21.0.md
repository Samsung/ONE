# Release Note 1.21.0

## ONE Compiler

- Support unrolling of LSTM and RNN Ops in `one-import-onnx` tool
- Introduced new tools `one-infer`, `circle-operator`, `circle-interpreter`
- Introduced `Workflow`(WIP) in `one-cmds`
- New option `quant_config` in `one-quantize`
- New option `fake_quantize` in `one-quantize`
- More Ops supported: Densify
- More Ops for quantization: ReduceMax
- More Ops for mixed-precision quantization (MPQ): LeakyRelu, Neg, Relu6, Squeeze
- More Ops for `convert_nchw_to_nhwc` option: LogSoftmax, ReduceMax, SplitV, Softmax
- New optimization options in `one-optimize`: `replace_non_const_fc_with_bmm`, `resolve_customop_splitv`, `fold_densify`
- Improved reshape elimination in `convert_nchw_to_nhwc` option.
- Support fusion of Channel-wise Add + Relu with TConv
- Support negative axis in ArgMin/Max
- Show errors for unrecognized options in `one-optimize`
- Fix shape inference for `StridedSlice`
- Fix FuseBatchNormWithTConvPass to support TConv with bias
- Deprecate `--O1` option in `circle2circle`
- Support gcc-11
- Support limited Float16 for kernels constants with dequantization to Float32

## ONE Runtime

### Basic Multimodel nnpackage
- Runtime supports to run nnpackage with two models

### Channel Wise Quantization on Conv2D and Depthwise Conv2D
- Conv2D and Depthwise Conv2D supports per-channel quantization of uint8 type.

### Batch Execution with TRIX backend
- TRIX backend supports batch execution which run in parallel with multicore
