# Release Note 1.15.0

## ONE Compiler

### Compiler Frontend

- Support more Ops for quantization
- Fix `record-minmax` tool for bool type, NaN values
- Fix `one-cmds` test scripts
- Remove `stdex` module
- `arser` supports short option


## ONE Runtime

### Runtime backend supports more operations and types

- CPU backend
  - Add: int8
  - AvgPool2d: int8
  - Conv2D: int8
  - DepthwiseConv2D: int8
  - Div: uint8
  - Elu: float
  - ExpandDims: int8
  - LogicalAnd: boolean
  - Maximum: uint8
  - MaxPool2D: int8
  - Minimum: uint8
  - Mul: int8 
  - Pad: int8
  - PadV2: int8
  - Quantize: uint8, int8
  - Reshape: int8
  - Resizebiliear: int8
  - Softmax: int8
  - Squeeze: int8
  - Sub: int8

### ARM Compute Library Update 

- ONERT uses Compute Library v21.02
