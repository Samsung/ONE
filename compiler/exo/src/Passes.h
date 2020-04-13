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

#ifndef __PASSES_H__
#define __PASSES_H__

// Please add in alphabetical order
// Please append 'Pass' suffix to Pass class and file names

#include "Pass/FoldReshapeOfConstPass.h"
#include "Pass/FoldTransposeOfConstPass.h"
#include "Pass/FuseBiasAddPass.h"
#include "Pass/FuseInstanceNormPass.h"
#include "Pass/FuseReluPass.h"
#include "Pass/FuseRsqrtPass.h"
#include "Pass/FuseSquaredDifferencePass.h"
#include "Pass/MergeConcatNodesPass.h"
#include "Pass/ShapeInferencePass.h"
#include "Pass/TypeInferencePass.h"

#include <logo/RemoveDeadNodePass.h>
#include <logo/RemoveForwardNodePass.h>
#include <logo/SimplifyDomainConversionPass.h>

#endif // __PASSES_H__
