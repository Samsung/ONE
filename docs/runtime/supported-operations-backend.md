# Supported Operations and backend

As of 2021-03-08

### Raw-data format (float32, int32, boolean, etc)

Operation | CPU | ACL-CL | ACL-NEON
-- | -- | -- | --
Abs | O | O | O
Add | O | O | O
AddN | O |   |
ArgMax | O | O | O
ArgMin | O | O | O
AvgPool2D | O | O | O
BatchMatmul | O |   |
BatchToSpaceND | O | O | O
BroadcastTo | O |   |
Cast | O | O | O
Concat | O | O | O
Conv2D | O | O | O
Cos | O |   |
Custom | O |   |
DepthToSpace | O | O | O
DepthwiseConv2D | O | O | O
Div | O | O | O
Einsum | O |   |
Elu | O |   |
EmbeddingLookup |   | O | O
Equal | O | O | O
Exp | O | O | O
ExpandDims | O | O | O
Fill | O |   |
Floor | O | O | O
FullyConnected | O | O | O
FusedBatchNorm | O |   |
Gather | O | O | O
Greater | O | O | O
GreaterEqual | O | O | O
HashtableLookup |   | O | O
If | O |   |
InstanceNormalize |   | O | O
L2Normalization | O | O | O
L2Pool |   | O | O
LeakyRelu | O | O | O
Less | O | O | O
LessEqual | O | O | O
LocalResponseNormalize |   | O | O
Log | O |   |
LogicalAnd | O | O | O
LogicalNot | O | O | O
LogicalOr | O | O | O
Logistic | O | O | O
LogSoftmax | O |   |
LSTM |   | O | O
MatrixBandPart | O |   |
Maximum | O | O | O
MaxPool2D | O | O | O
Mean | O | O | O
Minimum | O | O | O
Mul | O | O | O
Neg | O | O | O
NotEqual | O | O | O
OneHot | O | O |
Pack | O | O | O
Pad | O | O | O
PadV2 | O | O | O
Pow | O |   |
PReLU |   | O | O
Quantize | O |   |
Range | O |   |
Rank | O |   |
ReduceAny(All) | O |   |
ReduceAny(Any) | O |   |
ReduceMax(Max) | O | O | O
ReduceMin(Min) | O | O | O
ReduceProd | O |   |
ReduceSum(Sum) | O | O | O
ReLU | O | O | O
ReLU6 | O | O | O
Reshape | O | O | O
ResizeBilinear | O | O | O
ResizeNearestNeighbor |   | O | O
ReverseV2 | O | O | O
RNN |   | O | O
Round | O |   |
Rsqrt | O | O | O
Select | O |   |
SelectV2 | O |   |
Shape | O |   |
Sin | O |   |
Slice | O | O | O
Softmax | O | O | O
SpaceToBatchND | O | O | O
SpaceToDepth | O | O | O
Split | O | O | O
SplitV | O | O |
Sqrt | O | O | O
Square | O |   |   |
SquaredDifference | O | O | O
Squeeze | O | O | O
StridedSlice | O | O | O
Sub | O | O | O
Tanh | O | O | O
Tile | O |   |
TopKV2 |   |   | O
Transpose | O | O | O
TransposeConv |   | O | O
Unpack(Unstack) | O | O | O
UniDirectionalSequenceLSTM | O |   |
While | O |   |
ZerosLike | O |   |

### Quantization format (uint8 asymmetric)

Operation | CPU | ACL-CL | ACL-NEON
-- | -- | -- | --
Add | O | O | O
ArgMax | O | O | O
ArgMin | O | O | O
AvgPool2D | O | O | O
BatchToSpaceND | O | O | O
Cast | O | O |
Concat | O | O | O
Conv2D | O | O | O
Custom | O |   |
DepthToSpace | O | O | O
DepthwiseConv2D | O | O | O
Dequantize | O | O | O
Div | O |   |
EmbeddingLookup |   | O | O
Equal | O | O | O
Erf | O |   |
ExpandDims | O | O | O
FullyConnected | O | O | O
Gather | O | O | O
Greater | O | O | O
GreaterEqual | O | O | O
HashtableLookup |   | O | O
L2Normalization | O |   |
Less | O | O | O
LessEqual | O | O | O
Logistic | O | O | O
LogSoftmax | O |   |
Maximum | O | O | O
MaxPool2D | O | O | O
Mean | O | O | O
Minimum | O | O | O
Mul | O | O |
NotEqual | O | O | O
Pack |   | O | O
Pad | O | O | O
PadV2 | O | O | O
PReLU |   | O | O
Quantize | O |   |
Rank | O |   |
ReduceMax(Max) |   | O |
ReduceMin(Min) |   | O |
ReduceSum(Sum) | O | O |
ReLU |   | O | O
ReLU6 |   | O | O
Reshape | O | O | O
ResizeBilinear | O | O | O
ResizeNearestNeighbor |   | O | O
Shape | O |   |
Slice | O | O | O
Softmax | O | O | O
SpaceToBatchND | O | O | O
SpaceToDepth | O | O | O
Split | O | O | O
SplitV | O | O |
Squeeze | O | O | O
StatelessRandomUniform | O |   |
StridedSlice |   | O | O
Sub | O | O | O
Tanh | O | O | O
Tile | O |   |
Transpose | O | O | O
TransposeConv |   | O | O
Unpack(Unstack) |   | O | O

### Quantization format (int8)

Operation | CPU | ACL-CL | ACL-NEON
-- | -- | -- | --
Add | O | O | O
ArgMax | O | O | O
ArgMin | O | O | O
AvgPool2D | O |   |
Concat | O | O | O
Conv2D | O |   |
DepthToSpace | O |   |
DepthwiseConv2D | O |   |
Dequantize | O | O | O
ExpandDims | O | O | O
MaxPool2D | O |   |
Mul | O | O | O
Pad | O | O | O
PadV2 | O |   |
PReLU |   | O | O
Quantize | O |   |
Rank | O |   |
Reshape | O | O | O
ResizeBilinear | O | O | O
ResizeNearestNeighbor |   | O | O
Shape | O |   |
Softmax | O | O | O
Squeeze | O | O | O
Sub | O | O | O
