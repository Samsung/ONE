# Release Note 1.5.0

## Feature Highlights

- `ONE Compiler`
    - Compiler supports more operations

- `ONE Runtime`
    - CPU backend supports more operations

## ONE Compiler

### Compiler supports more operations

The following operations are supported on CPU backend :

- Abs, Add, ArgMax, AvgPool2D, BatchToSpaceND, Cast, Concat, Const, Conv2D, Cos,
Custom, DepthwiseConv2D, Div, Elu, Equal, Exp, ExpandDims, Fill, FloorDiv,
FloorMod, FullyConnected, Gather, GatherNd, Greater, GreaterEqual, If,
LeakyRelu, Less, LocalResponseNormalize, LogicalAnd, LogicalNot, LogicalOr,
Logistic, Maximum, MaxPool2D, Mean, Minimum, MirrorPad, Mul, Neg, NotEqual,
OneHot, Pack, Pad, Pow, Range, ReduceAny(Any), ReduceMax(Max), ReduceProd,
ReduceSum(Sum), ReLU, RELU_N1_TO_1, Reshape, ResizeNearestneighbor, Rsqrt,
Select, Shape, Sin, Slice, Softmax, SpaceToBatchND, SpaceToDepth, Split, SplitV,
Square, SquaredDifference, Squeeze, StridedSlice, Sub, Tanh, Tile, TopKV2,
Transpose, Unpack(Unstack), While, , ZerosLike

## ONE Runtime

### CPU backend supports more operations

The following operations are supported on CPU backend :

- ArgMax, Cos, ExpandDims, Fill, Log, LogicalNot, LogicalOr, Mean, Neg, Pow,
  ReLU, ReduceAny, ReduceProd, Reverse, Round, Select, SquaredDifference, Tile,
  ZerosLike
