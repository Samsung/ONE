# Release Note 1.27.0

## ONE Compiler

- Support more Op(s): CircleGRU, CircleRelu0To1
- Support more optimization option(s): `resolve_former_customop`, `--forward_transpose_op`,
    `fold_shape`, `remove_gather_guard`, `fuse_add_with_conv`, `fold_squeeze`, `fuse_rsqrt`
- Support INT4, UINT4 data types
- Support 4bit quantization of ONNX fake quantize model
- Introduce global configuration target feature
- Introduce command schema feature
- Use C++17
