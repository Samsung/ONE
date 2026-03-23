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

#include "FullyConnected.h"
#include "Assert.h"

#if 0
#include "internal/Matrix.h"
#include "internal/Fused.h"
#include "internal/GEMM.h"
#include "internal/ActivationUtils.h"
#endif

bool fullyConnectedPrepare(const Shape &input, const Shape &weights, const Shape &bias,
                           Shape *output)
{
  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  ASSERT(input.type == weights.type);
  if (input.type == OperandType::TENSOR_QUANT8_ASYMM)
  {
    ASSERT(bias.type == OperandType::TENSOR_INT32);
  }
  else
  {
    ASSERT(input.type == bias.type);
  }
  ASSERT(getNumberOfDimensions(input) >= 2);
  uint32_t input_size = getNumberOfElements(input);
  uint32_t num_units = getSizeOfDimension(weights, 0);

  // modified to resolve Coverity 118949 (Apr 25, 2018) by hyunsik.yoon
  // Original Code:
  // uint32_t batch_size = input_size / getSizeOfDimension(weights, 1);
  //
  // Coverity Detection: Division by zero
  //
  // Code below is modified code

  uint32_t shape_size = getSizeOfDimension(weights, 1);
  if (shape_size == 0)
  {
    return false;
  }

  uint32_t batch_size = input_size / shape_size;

  ASSERT(getSizeOfDimension(bias, 0) == num_units);
  ASSERT(getSizeOfDimension(weights, 1) * batch_size == input_size);
  ASSERT(getNumberOfDimensions(weights) == 2);

  output->type = input.type;
  output->dimensions = {batch_size, num_units};

  return true;
}
