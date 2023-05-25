/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file has no ifdef guard intentionally

#include "ir/operation/AddN.h"
#include "ir/operation/ArgMinMax.h"
#include "ir/operation/BatchMatMul.h"
#include "ir/operation/BatchToSpaceND.h"
#include "ir/operation/BCQFullyConnected.h"
#include "ir/operation/BCQGather.h"
#include "ir/operation/BinaryArithmetic.h"
#include "ir/operation/BroadcastTo.h"
#include "ir/operation/Bulk.h"
#include "ir/operation/Comparison.h"
#include "ir/operation/Concat.h"
#include "ir/operation/Conv2D.h"
#include "ir/operation/ConvertFp16ToFp32.h"
#include "ir/operation/ConvertFp32ToFp16.h"
#include "ir/operation/Custom.h"
#include "ir/operation/DepthToSpace.h"
#include "ir/operation/DepthwiseConv2D.h"
#include "ir/operation/Einsum.h"
#include "ir/operation/ElementwiseActivation.h"
#include "ir/operation/ElementwiseBinary.h"
#include "ir/operation/ElementwiseUnary.h"
#include "ir/operation/EmbeddingLookup.h"
#include "ir/operation/ExpandDims.h"
#include "ir/operation/Fill.h"
#include "ir/operation/FullyConnected.h"
#include "ir/operation/FusedBatchNorm.h"
#include "ir/operation/Gather.h"
#include "ir/operation/HashtableLookup.h"
#include "ir/operation/If.h"
#include "ir/operation/InstanceNorm.h"
#include "ir/operation/L2Normalization.h"
#include "ir/operation/LocalResponseNormalization.h"
#include "ir/operation/LogSoftmax.h"
#include "ir/operation/Loss.h"
#include "ir/operation/LSTM.h"
#include "ir/operation/MatrixBandPart.h"
#include "ir/operation/DetectionPostProcess.h"
#include "ir/operation/OneHot.h"
#include "ir/operation/Pack.h"
#include "ir/operation/Pad.h"
#include "ir/operation/Permute.h"
#include "ir/operation/Pool2D.h"
#include "ir/operation/Pow.h"
#include "ir/operation/PReLU.h"
#include "ir/operation/Range.h"
#include "ir/operation/Rank.h"
#include "ir/operation/Reduce.h"
#include "ir/operation/Reshape.h"
#include "ir/operation/ResizeBilinear.h"
#include "ir/operation/ResizeNearestNeighbor.h"
#include "ir/operation/Reverse.h"
#include "ir/operation/RNN.h"
#include "ir/operation/Select.h"
#include "ir/operation/Shape.h"
#include "ir/operation/Slice.h"
#include "ir/operation/Softmax.h"
#include "ir/operation/SpaceToBatchND.h"
#include "ir/operation/SpaceToDepth.h"
#include "ir/operation/Split.h"
#include "ir/operation/SplitV.h"
#include "ir/operation/SquaredDifference.h"
#include "ir/operation/Squeeze.h"
#include "ir/operation/StatelessRandomUniform.h"
#include "ir/operation/StridedSlice.h"
#include "ir/operation/Tile.h"
#include "ir/operation/TopKV2.h"
#include "ir/operation/Transpose.h"
#include "ir/operation/TransposeConv.h"
#include "ir/operation/Unpack.h"
#include "ir/operation/While.h"
