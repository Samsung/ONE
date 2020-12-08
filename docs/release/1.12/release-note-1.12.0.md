# Release Note 1.12.0

## ONE Compiler

### Compiler Frontend

- Add optimization pass: ReplaceMulAddWithDepthwiseConvPass, SubstitutePackToReshape, RemoveRedundantTranspose, ShuffleWeightTo16x1Float32Pass
- Add quantization for InstanceNorm.
- Fix bug of `one-import-bcq` command for `--v1`, `--v2` arguments.
- Fix FuseBCQPass to work with inter-subgraphs in the model file and minor BCQ related optimizations.

## ONE Runtime

### Runtime backend operation supports more operations and types

- CPU backend
  - Concat: int8
  - DepthToSpace: float, uint8, int8
  - LeakyRelu: float
- ACL-CL backend
  - ArgMin: float, uint8, int8
- ACL-NEON backend
  - ArgMax: int8
  - ArgMin: float, uint8, int8

### nnpackage defines configuration file

- Allow users to set configuration variable via conf file. For more information, See [nnpackage spec](../../../nnpackage/spec)
