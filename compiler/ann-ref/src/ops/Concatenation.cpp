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

#include "Concatenation.h"
#include "Assert.h"

bool concatenationPrepare(const std::vector<Shape> &inputShapes, int32_t axis, Shape *output)
{

  int num_inputs = inputShapes.size();
  OperandType input_type = inputShapes[0].type;
  uint32_t num_dimensions = getNumberOfDimensions(inputShapes[0]);

  ASSERT(axis >= 0);
  ASSERT(axis < (int32_t)num_dimensions);

  int sum_axis = getSizeOfDimension(inputShapes[0], axis);
  for (int i = 1; i < num_inputs; ++i)
  {
    ASSERT(getNumberOfDimensions(inputShapes[i]) == num_dimensions);
    ASSERT(inputShapes[i].type == inputShapes[0].type);
    if (input_type == OperandType::TENSOR_QUANT8_ASYMM)
    {
      ASSERT(inputShapes[0].offset == inputShapes[i].offset);
      ASSERT(inputShapes[0].scale == inputShapes[i].scale);
    }
    for (int d = 0; d < (int32_t)num_dimensions; ++d)
    {
      if (d == axis)
      {
        sum_axis += getSizeOfDimension(inputShapes[i], axis);
      }
      else
      {
        ASSERT(getSizeOfDimension(inputShapes[0], d) ==
                     getSizeOfDimension(inputShapes[i], d));
      }
    }
  }

  output->type = input_type;
  output->dimensions = inputShapes[0].dimensions;
  output->dimensions[axis] = sum_axis;

  if (input_type == OperandType::TENSOR_QUANT8_ASYMM)
  {
    ASSERT(inputShapes[0].offset == output->offset);
    ASSERT(inputShapes[0].scale == output->scale);
  }

  return true;
}
