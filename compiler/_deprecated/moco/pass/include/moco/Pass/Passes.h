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

#ifndef __MOCO_PASS_PASSES_H__
#define __MOCO_PASS_PASSES_H__

#include "Passes/ConstantFoldAdd.h"
#include "Passes/ConstantFoldMul.h"
#include "Passes/ConstantFoldPack.h"
#include "Passes/ConstantFoldStridedSlice.h"
#include "Passes/FuseBinaryIntoPreceding.h"
#include "Passes/RemoveTFIdentityNode.h"
#include "Passes/ResolveConstantShape.h"
#include "Passes/ResolveFusedBatchNorm.h"
#include "Passes/ResolveReshapeWildcardDim.h"
#include "Passes/ResolveSquaredDifference.h"
#include "Passes/SqueezeReduceNode.h"

#endif // __MOCO_PASS_PASSES_H__
