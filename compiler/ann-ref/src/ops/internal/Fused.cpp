/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#include "Fused.h"
#include "NeuralNetworks.h"

static_assert(static_cast<int32_t>(FusedActivationFunc::NONE) == ANEURALNETWORKS_FUSED_NONE,
              "FusedActivationFunc::NONE != ANEURALNETWORKS_FUSED_NONE");
static_assert(static_cast<int32_t>(FusedActivationFunc::RELU) == ANEURALNETWORKS_FUSED_RELU,
              "FusedActivationFunc::RELU != ANEURALNETWORKS_FUSED_RELU");
static_assert(static_cast<int32_t>(FusedActivationFunc::RELU1) == ANEURALNETWORKS_FUSED_RELU1,
              "FusedActivationFunc::RELU1 != ANEURALNETWORKS_FUSED_RELU1");
static_assert(static_cast<int32_t>(FusedActivationFunc::RELU6) == ANEURALNETWORKS_FUSED_RELU6,
              "FusedActivationFunc::RELU6 != ANEURALNETWORKS_FUSED_RELU6");
