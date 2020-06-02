/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __OP_CHEFS_H__
#define __OP_CHEFS_H__

#include "Op/Abs.h"
#include "Op/Add.h"
#include "Op/ArgMax.h"
#include "Op/AveragePool2D.h"
#include "Op/BatchToSpaceND.h"
#include "Op/Cast.h"
#include "Op/Concatenation.h"
#include "Op/Conv2D.h"
#include "Op/Cos.h"
#include "Op/DepthToSpace.h"
#include "Op/DepthwiseConv2D.h"
#include "Op/Div.h"
#include "Op/ELU.h"
#include "Op/Equal.h"
#include "Op/Exp.h"
#include "Op/ExpandDims.h"
#include "Op/Fill.h"
#include "Op/Floor.h"
#include "Op/FloorDiv.h"
#include "Op/FloorMod.h"
#include "Op/FullyConnected.h"
#include "Op/Gather.h"
#include "Op/GatherNd.h"
#include "Op/Greater.h"
#include "Op/GreaterEqual.h"
#include "Op/If.h"
#include "Op/L2Normalize.h"
#include "Op/L2Pool2D.h"
#include "Op/LeakyRelu.h"
#include "Op/Less.h"
#include "Op/LocalResponseNormalization.h"
#include "Op/Log.h"
#include "Op/LogicalAnd.h"
#include "Op/LogicalNot.h"
#include "Op/LogicalOr.h"
#include "Op/Logistic.h"
#include "Op/Maximum.h"
#include "Op/MaxPool2D.h"
#include "Op/Mean.h"
#include "Op/Minimum.h"
#include "Op/MirrorPad.h"
#include "Op/Mul.h"
#include "Op/Neg.h"
#include "Op/NotEqual.h"
#include "Op/OneHot.h"
#include "Op/Pack.h"
#include "Op/Pad.h"
#include "Op/Pow.h"
#include "Op/PRelu.h"
#include "Op/Range.h"
#include "Op/ReduceAny.h"
#include "Op/ReduceMax.h"
#include "Op/ReduceProd.h"
#include "Op/ReLU.h"
#include "Op/ReLU6.h"
#include "Op/ReLUN1To1.h"
#include "Op/Reshape.h"
#include "Op/ResizeBilinear.h"
#include "Op/ResizeNearestNeighbor.h"
#include "Op/Rsqrt.h"
#include "Op/Select.h"
#include "Op/Shape.h"
#include "Op/Sin.h"
#include "Op/Slice.h"
#include "Op/Softmax.h"
#include "Op/SpaceToBatchND.h"
#include "Op/SpaceToDepth.h"
#include "Op/Split.h"
#include "Op/SplitV.h"
#include "Op/Sqrt.h"
#include "Op/Square.h"
#include "Op/SquaredDifference.h"
#include "Op/Squeeze.h"
#include "Op/StridedSlice.h"
#include "Op/Sub.h"
#include "Op/Sum.h"
#include "Op/Tanh.h"
#include "Op/Tile.h"
#include "Op/TopKV2.h"
#include "Op/Transpose.h"
#include "Op/Unpack.h"
#include "Op/While.h"
#include "Op/ZerosLike.h"

#include "CustomOp/All.h"
#include "CustomOp/BatchMatMulV2.h"
#include "CustomOp/MatrixBandPart.h"

#endif // __OP_CHEFS_H__
