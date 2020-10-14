# Supported Operations and backend

As of 2020-10-14

### Raw-data format (float32, int32, boolean, etc)

Operation | CPU | ACL-CL | ACL-NEON
-- | -- | -- | --
Abs | O | O | O
Add | O | O | O
ArgMax | O | O | O
ArgMin | O |   |
AvgPool2D | O | O | O
BatchMatmul | O |   |
BatchToSpaceND | O | O | O
Cast | O | O | O
Concat | O | O | O
Conv2D | O | O | O
Cos | O |   |
Custom | O |   |
DepthToSpace |   | O | O
DepthwiseConv2D | O | O | O
Div | O | O | O
EmbeddingLookup |   | O | O
Equal | O | O | O
Exp | O | O | O
ExpandDims | O |   |
Fill | O |   |
Floor |   | O | O
FullyConnected | O | O | O
Gather | O | O | O
Greater | O | O | O
GreaterEqual | O | O | O
HashtableLookup |   | O | O
If | O |   |
InstanceNormalize |   | O | O
L2Normalization | O | O | O
L2Pool |   | O | O
LeakyRelu |   | O | O
Less | O | O | O
LessEqual | O | O | O
LocalResponseNormalize |   | O | O
Log | O |   |
LogicalAnd |   | O | O
LogicalNot | O | O | O
LogicalOr | O | O | O
Logistic | O | O | O
LogSoftmax | O |   |
LSHProjection |   |   |
LSTM |   | O | O
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
ReduceAny(Any) | O |   |
ReduceMax(Max) | O | O | O
ReduceMin(Min) | O | O | O
ReduceProd | O |   |
ReduceSum(Sum) | O | O | O
ReLU | O | O | O
ReLU6 |   | O | O
Reshape | O | O | O
ResizeBilinear | O | O | O
ReverseV2 | O |   | O
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
SplitV | O |   |
Sqrt | O | O | O
SquaredDifference | O | O | O
Squeeze | O | O | O
StridedSlice | O | O | O
Sub | O | O | O
Svdf |   |   |
Tanh | O | O | O
Tile | O |   |
TopKV2 |   |   | O
Transpose | O | O | O
TransposeConv |   | O | O
Unpack(Unstack) | O | O | O
While | O |   |
ZerosLike | O |   |

### Quantization format (uint8 asymmetric)

Operation | CPU | ACL-CL | ACL-NEON
-- | -- | -- | --
Add | O | O | O
ArgMax | O | O | O
ArgMin | O |   |
AvgPool2D | O | O | O
BatchToSpaceND | O | O | O
Cast | O | O |
Concat | O | O | O
Conv2D | O | O | O
Custom | O |   |
DepthToSpace |   | O | O
DepthwiseConv2D | O | O | O
Dequantize |   | O | O
EmbeddingLookup |   | O | O
Equal | O | O | O
ExpandDims | O |   |
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
Maximum |   | O | O
MaxPool2D | O | O | O
Mean | O | O | O
Minimum |   | O | O
Mul | O | O |
NotEqual | O | O | O
OneHot |   | O |
Pack |   | O | O
Pad | O | O | O
PadV2 | O | O | O
PReLU |   | O | O
ReduceMax(Max) |   | O |
ReduceMin(Min) |   | O |
ReduceSum(Sum) | O | O |
ReLU |   | O | O
ReLU6 |   | O | O
Reshape | O | O | O
ResizeBilinear | O |   | O
Shape | O |   |
Slice | O | O | O
Softmax | O | O | O
SpaceToBatchND | O | O | O
SpaceToDepth | O | O | O
Split | O | O | O
SplitV | O |   |
Squeeze | O | O | O
StridedSlice |   | O | O
Sub | O | O | O
Tanh | O | O | O
Tile | O |   |
Transpose | O | O | O
TransposeConv |   | O | O
Unpack(Unstack) |   | O | O
