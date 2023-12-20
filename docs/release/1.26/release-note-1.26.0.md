# Release Note 1.26.0

## ONE Compiler

- Support more Op(s): HardSwish, CumSum, BroadcastTo
- Support more optimization option(s): `decompose_softmax`, `decompose_hardswish`, `fuse_slice_with_tconv`,
    `fuse_mul_with_conv`, `remove_unnecessary_add`, `fuse_horizontal_fc_layers`, `common_subexpression_elimination`,
    `remove_unnecessary_transpose`
- _one-quantize_ supports more option
  - Requantization option to convert TF2-quantized int8 model to uint8 model (--requantize)
  - A new option to automatically find mixed-precision configuration (--ampq)
  - A new option to save calibrated min/max values (--save_min_max)
  - Add new parameters for moving average calibration (--moving_avg_batch, --moving_avg_const)
- Introduce _q-implant_ that writes quantization parameters and weights into the circle model
- Introduce _minmax-embedder_ that embeds min/max values into the circle model
