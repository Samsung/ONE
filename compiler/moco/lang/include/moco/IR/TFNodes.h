/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MOCO_IR_TFNODES_H__
#define __MOCO_IR_TFNODES_H__

#include "moco/IR/Nodes/TFAdd.h"
#include "moco/IR/Nodes/TFAvgPool.h"
#include "moco/IR/Nodes/TFBiasAdd.h"
#include "moco/IR/Nodes/TFConcatV2.h"
#include "moco/IR/Nodes/TFConst.h"
#include "moco/IR/Nodes/TFConv2D.h"
#include "moco/IR/Nodes/TFConv2DBackpropInput.h"
#include "moco/IR/Nodes/TFDepthwiseConv2dNative.h"
#include "moco/IR/Nodes/TFFakeQuantWithMinMaxVars.h"
#include "moco/IR/Nodes/TFFusedBatchNorm.h"
#include "moco/IR/Nodes/TFIdentity.h"
#include "moco/IR/Nodes/TFMaximum.h"
#include "moco/IR/Nodes/TFMaxPool.h"
#include "moco/IR/Nodes/TFMean.h"
#include "moco/IR/Nodes/TFMul.h"
#include "moco/IR/Nodes/TFPack.h"
#include "moco/IR/Nodes/TFPad.h"
#include "moco/IR/Nodes/TFPlaceholder.h"
#include "moco/IR/Nodes/TFRealDiv.h"
#include "moco/IR/Nodes/TFRelu.h"
#include "moco/IR/Nodes/TFRelu6.h"
#include "moco/IR/Nodes/TFReshape.h"
#include "moco/IR/Nodes/TFRsqrt.h"
#include "moco/IR/Nodes/TFShape.h"
#include "moco/IR/Nodes/TFSoftmax.h"
#include "moco/IR/Nodes/TFSqrt.h"
#include "moco/IR/Nodes/TFSquaredDifference.h"
#include "moco/IR/Nodes/TFSqueeze.h"
#include "moco/IR/Nodes/TFStopGradient.h"
#include "moco/IR/Nodes/TFStridedSlice.h"
#include "moco/IR/Nodes/TFSub.h"
#include "moco/IR/Nodes/TFTanh.h"
// For virtual node(s)
#include "moco/IR/Nodes/TFPush.h"

#endif // __MOCO_IR_TFNODES_H__
