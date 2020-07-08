# Release Note 1.6.0

## Feature Highlights

- **ONE** Compiler
    - Compiler supports 22 more operations including 1 custom operation.

- **ONE** Runtime
    - CPU backend supports 8 more operations
    - Support dynamically shaped tensors
    - Support Control Flow operations
    - API updates

- Add **ONE** Runtime package for `x86_64` linux (Experimental)

## ONE Compiler

### Compiler supports more operations

- AddN, ArgMin, Custom(BatchMatmul), Ceil, DepthToSpace, Floor, InstanceNormalize,
L2Normalization, L2Pool, LessEqual, Log, LogSoftmax, PReLU, Rank, ReduceMin(Min),
ResizeBilinear, Round, ScatterND, Sqrt, TransposeConv, BCQGather,
BCQFullyConnected

## ONE Runtime

### CPU backend supports more operations

- BatchMatMul, BroadcastTo, Einsum, FusedBatchNorm, MatrixBandPart, Range,  ReduceAll, Add(quant8)

### Support dynamically shaped tensors

- Support static shape inference (input resizing)
- Support dynamic shape inference (general resizing)

### Support Control Flow operations

- IF and WHILE
- Fully support static and dynamic tensors

### API updates

- Introduce `nnfw_set_input_tensorinfo` for input resizing
- `nnfw_input_tensorinfo` and `nnfw_output_tensorinfo` behavior have changed to return tensorinfo according to the session state
