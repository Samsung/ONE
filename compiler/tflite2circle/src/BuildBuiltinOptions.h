/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __BUILD_BUITIN_OPTIONS_H__
#define __BUILD_BUITIN_OPTIONS_H__

// NOTE please add new option headers in alphabetical order

#include "BuildBuiltinOptions/AbsOptions.h"
#include "BuildBuiltinOptions/AddOptions.h"
#include "BuildBuiltinOptions/AddNOptions.h"
#include "BuildBuiltinOptions/ArgMaxOptions.h"
#include "BuildBuiltinOptions/ArgMinOptions.h"
#include "BuildBuiltinOptions/BatchMatMulOptions.h"
#include "BuildBuiltinOptions/BatchToSpaceNDOptions.h"
#include "BuildBuiltinOptions/BidirectionalSequenceLSTMOptions.h"
#include "BuildBuiltinOptions/BroadcastToOptions.h"
#include "BuildBuiltinOptions/CastOptions.h"
#include "BuildBuiltinOptions/ConcatenationOptions.h"
#include "BuildBuiltinOptions/Conv2DOptions.h"
#include "BuildBuiltinOptions/CosOptions.h"
#include "BuildBuiltinOptions/CumSumOptions.h"
#include "BuildBuiltinOptions/DensifyOptions.h"
#include "BuildBuiltinOptions/DepthToSpaceOptions.h"
#include "BuildBuiltinOptions/DepthwiseConv2DOptions.h"
#include "BuildBuiltinOptions/DequantizeOptions.h"
#include "BuildBuiltinOptions/DivOptions.h"
#include "BuildBuiltinOptions/EqualOptions.h"
#include "BuildBuiltinOptions/ExpandDimsOptions.h"
#include "BuildBuiltinOptions/ExpOptions.h"
#include "BuildBuiltinOptions/FakeQuantOptions.h"
#include "BuildBuiltinOptions/FillOptions.h"
#include "BuildBuiltinOptions/FloorDivOptions.h"
#include "BuildBuiltinOptions/FloorModOptions.h"
#include "BuildBuiltinOptions/FullyConnectedOptions.h"
#include "BuildBuiltinOptions/GatherOptions.h"
#include "BuildBuiltinOptions/GatherNdOptions.h"
#include "BuildBuiltinOptions/GeluOptions.h"
#include "BuildBuiltinOptions/GreaterOptions.h"
#include "BuildBuiltinOptions/GreaterEqualOptions.h"
#include "BuildBuiltinOptions/IfOptions.h"
#include "BuildBuiltinOptions/L2NormalizeOptions.h"
// L2Pool2D uses Pool2DOptions
#include "BuildBuiltinOptions/LeakyReluOptions.h"
#include "BuildBuiltinOptions/LessOptions.h"
#include "BuildBuiltinOptions/LessEqualOptions.h"
#include "BuildBuiltinOptions/LocalResponseNormalizationOptions.h"
#include "BuildBuiltinOptions/LogicalAndOptions.h"
#include "BuildBuiltinOptions/LogicalNotOptions.h"
#include "BuildBuiltinOptions/LogicalOrOptions.h"
// There is no LogisticOptions
#include "BuildBuiltinOptions/LogSoftmaxOptions.h"
#include "BuildBuiltinOptions/MatrixDiagOptions.h"
#include "BuildBuiltinOptions/MatrixSetDiagOptions.h"
#include "BuildBuiltinOptions/MaximumMinimumOptions.h"
#include "BuildBuiltinOptions/MirrorPadOptions.h"
#include "BuildBuiltinOptions/MulOptions.h"
#include "BuildBuiltinOptions/NegOptions.h"
#include "BuildBuiltinOptions/NonMaxSuppressionV4Options.h"
#include "BuildBuiltinOptions/NonMaxSuppressionV5Options.h"
#include "BuildBuiltinOptions/NotEqualOptions.h"
#include "BuildBuiltinOptions/OneHotOptions.h"
#include "BuildBuiltinOptions/PackOptions.h"
#include "BuildBuiltinOptions/PadOptions.h"
#include "BuildBuiltinOptions/PadV2Options.h"
#include "BuildBuiltinOptions/RangeOptions.h"
#include "BuildBuiltinOptions/Pool2DOptions.h"
#include "BuildBuiltinOptions/PowOptions.h"
#include "BuildBuiltinOptions/RankOptions.h"
// There is no PReluOptions
#include "BuildBuiltinOptions/ReducerOptions.h"
#include "BuildBuiltinOptions/ReshapeOptions.h"
#include "BuildBuiltinOptions/ResizeBilinearOptions.h"
#include "BuildBuiltinOptions/ResizeNearestNeighborOptions.h"
#include "BuildBuiltinOptions/ReverseSequenceOptions.h"
#include "BuildBuiltinOptions/ReverseV2Options.h"
// There is no RoundOptions
// There is no RsqrtOptions
#include "BuildBuiltinOptions/ScatterNdOptions.h"
#include "BuildBuiltinOptions/SegmentSumOptions.h"
#include "BuildBuiltinOptions/SelectOptions.h"
#include "BuildBuiltinOptions/SelectV2Options.h"
#include "BuildBuiltinOptions/ShapeOptions.h"
// There is no SinOptions
#include "BuildBuiltinOptions/SliceOptions.h"
#include "BuildBuiltinOptions/SoftmaxOptions.h"
#include "BuildBuiltinOptions/SpaceToBatchNDOptions.h"
#include "BuildBuiltinOptions/SpaceToDepthOptions.h"
#include "BuildBuiltinOptions/SparseToDenseOptions.h"
#include "BuildBuiltinOptions/SplitOptions.h"
#include "BuildBuiltinOptions/SplitVOptions.h"
#include "BuildBuiltinOptions/SquaredDifferenceOptions.h"
#include "BuildBuiltinOptions/SquareOptions.h"
#include "BuildBuiltinOptions/SqueezeOptions.h"
#include "BuildBuiltinOptions/StridedSliceOptions.h"
#include "BuildBuiltinOptions/SubOptions.h"
#include "BuildBuiltinOptions/SVDFOptions.h"
#include "BuildBuiltinOptions/TileOptions.h"
#include "BuildBuiltinOptions/TopKV2Options.h"
#include "BuildBuiltinOptions/TransposeOptions.h"
#include "BuildBuiltinOptions/TransposeConvOptions.h"
#include "BuildBuiltinOptions/UnidirectionalSequenceLSTMOptions.h"
#include "BuildBuiltinOptions/UniqueOptions.h"
#include "BuildBuiltinOptions/UnpackOptions.h"
#include "BuildBuiltinOptions/WhereOptions.h"
#include "BuildBuiltinOptions/WhileOptions.h"
#include "BuildBuiltinOptions/ZerosLikeOptions.h"

#endif // __BUILD_BUITIN_OPTIONS_H__
