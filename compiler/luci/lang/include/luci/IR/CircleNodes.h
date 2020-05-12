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

#ifndef __LUCI_IR_CIRCLENODES_H__
#define __LUCI_IR_CIRCLENODES_H__

#include "Nodes/CircleAbs.h"
#include "Nodes/CircleAdd.h"
#include "Nodes/CircleArgMax.h"
#include "Nodes/CircleAveragePool2D.h"
#include "Nodes/CircleBatchMatMul.h"
#include "Nodes/CircleBatchToSpaceND.h"
#include "Nodes/CircleCast.h"
#include "Nodes/CircleConcatenation.h"
#include "Nodes/CircleConst.h"
#include "Nodes/CircleConv2D.h"
#include "Nodes/CircleCos.h"
#include "Nodes/CircleCustom.h"
#include "Nodes/CircleDepthwiseConv2D.h"
#include "Nodes/CircleDiv.h"
#include "Nodes/CircleEqual.h"
#include "Nodes/CircleExp.h"
#include "Nodes/CircleFullyConnected.h"
#include "Nodes/CircleGather.h"
#include "Nodes/CircleIf.h"
#include "Nodes/CircleLogicalNot.h"
#include "Nodes/CircleLogicalOr.h"
#include "Nodes/CircleLogistic.h"
#include "Nodes/CircleMaximum.h"
#include "Nodes/CircleMaxPool2D.h"
#include "Nodes/CircleMean.h"
#include "Nodes/CircleMul.h"
#include "Nodes/CirclePack.h"
#include "Nodes/CirclePad.h"
#include "Nodes/CircleReduceProd.h"
#include "Nodes/CircleRelu6.h"
#include "Nodes/CircleRelu.h"
#include "Nodes/CircleReshape.h"
#include "Nodes/CircleRsqrt.h"
#include "Nodes/CircleSin.h"
#include "Nodes/CircleSoftmax.h"
#include "Nodes/CircleSpaceToBatchND.h"
#include "Nodes/CircleSplit.h"
#include "Nodes/CircleSplitV.h"
#include "Nodes/CircleSqrt.h"
#include "Nodes/CircleSquare.h"
#include "Nodes/CircleSquaredDifference.h"
#include "Nodes/CircleStridedSlice.h"
#include "Nodes/CircleSub.h"
#include "Nodes/CircleTanh.h"
#include "Nodes/CircleTile.h"
#include "Nodes/CircleTransposeConv.h"
#include "Nodes/CircleTranspose.h"
#include "Nodes/CircleUnpack.h"
#include "Nodes/CircleWhile.h"
// Circle only
#include "Nodes/CircleInstanceNorm.h"
// Virtual nodes
#include "Nodes/CircleInput.h"
#include "Nodes/CircleOutput.h"
#include "Nodes/CircleIfOut.h"
#include "Nodes/CircleUnpackOut.h"
#include "Nodes/CircleSplitOut.h"
#include "Nodes/CircleSplitVOut.h"
#include "Nodes/CircleWhileOut.h"

namespace luci
{

/**
 * @brief  Set both CircleReshape's 2nd input as CircleConst, and newShape attribute
 *         with same value
 * @note   Shape inference for TFLReshape forces them to be same
 *
 * TODO find better place for this helper
 */
void set_new_shape(CircleReshape *node, int32_t *base, uint32_t size);

} // namespace luci

#endif // __LUCI_IR_CIRCLENODES_H__
