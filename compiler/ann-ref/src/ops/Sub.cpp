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

#include "Sub.h"
#include "Assert.h"

bool subPrepare(const Shape &in1, const Shape &in2, Shape *out)
{
  ASSERT(getNumberOfDimensions(in1) <= 4 && getNumberOfDimensions(in2) <= 4);
  ASSERT(in1.type == in2.type);
  if (SameShape(in1, in2))
  {
    return SetShape(in1, out);
  }
  else
  {
    // BroadcastSub needed
    uint32_t numberOfDims1 = getNumberOfDimensions(in1);
    uint32_t numberOfDims2 = getNumberOfDimensions(in2);
    uint32_t maxDims = std::max(numberOfDims1, numberOfDims2);
    out->dimensions = std::vector<uint32_t>(maxDims);
    for (uint32_t i = 1; i <= maxDims; i++)
    {
      uint32_t dim1 = 1;
      if (i <= numberOfDims1)
      {
        dim1 = getSizeOfDimension(in1, numberOfDims1 - i);
      }
      uint32_t dim2 = 1;
      if (i <= numberOfDims2)
      {
        dim2 = getSizeOfDimension(in2, numberOfDims2 - i);
      }
      if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
      {
        LOG(ERROR) << "Dimensions mismatch for BroadcastSub";
        return false;
      }
      out->dimensions[maxDims - i] = std::max(dim1, dim2);
    }
  }
  return true;
}
