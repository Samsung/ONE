# Release Note 1.11.0

## ONE Compiler

### Compiler supports more operations
- MaxPoolWithArgMax by CustomOp

### Changes 
- `one-build` command added as representative command
- one-cmds are now revised to python script and supports configuration file as input parameters
- added `rawdata2hdf5` tool to prepare quantization data
- added more optimization passes in `one-optimize`; `fuse_preactivation_batchnorm`, `make_batchnorm_gamma_positive` and `fuse_activation_function`

## ONE Runtime

### Runtime backend operation supports more operations and types

- CPU backend
  - float: AddN, Floor, UniDirectionalSequenceLSTM
  - uint8: Dequantize, Rank
  - int8: Dequantize, Rank, Shape
