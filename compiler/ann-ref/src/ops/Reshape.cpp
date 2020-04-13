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

#include "Reshape.h"
#include "Operand.h"
#include "Assert.h"

#include <cstring>

bool reshapePrepare(const Shape &input, const int32_t *targetDims, const int32_t targetDimsSize,
                    Shape *output)
{
  // Reshape allows one of the targetDims components to have the
  // special -1 value, meaning it will be calculated automatically based on the
  // input. Here we calculate what that dimension should be so that the number
  // of output elements in the same as the number of input elements.
  int32_t numInputElements = (int32_t)getNumberOfElements(input);

  std::vector<uint32_t> outDims(targetDimsSize);
  int32_t numOutputElements = 1;
  int32_t strechDim = -1;
  for (int32_t i = 0; i < targetDimsSize; ++i)
  {
    int32_t value = targetDims[i];
    if (value == -1)
    {
      ASSERT(strechDim == -1);
      strechDim = i;
    }
    else
    {
      numOutputElements *= value;
      outDims[i] = (uint32_t)value;
    }
  }
  if (strechDim != -1)
  {
    int32_t strechValue = numInputElements / numOutputElements;
    outDims[strechDim] = (uint32_t)strechValue;
    numOutputElements *= strechValue;
  }

  ASSERT(numInputElements == numOutputElements);

  output->type = input.type;
  output->dimensions = outDims;
  output->offset = input.offset;
  output->scale = input.scale;

  return true;
}

bool reshapeGeneric(const void *inputData, const Shape &inputShape, void *outputData,
                    const Shape &outputShape)
{
  size_t count = sizeOfData(inputShape.type, inputShape.dimensions);
  memcpy(outputData, inputData, count);
  return true;
}
