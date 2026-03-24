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

#include "Shape.h"

#include <cstddef> // For 'size_t'

bool SameShape(const Shape &in1, const Shape &in2)
{
  if (in1.type != in2.type || in1.dimensions.size() != in2.dimensions.size())
  {
    return false;
  }
  for (size_t i = 0; i < in1.dimensions.size(); i++)
  {
    if (in1.dimensions[i] != in2.dimensions[i])
    {
      return false;
    }
  }
  return true;
}

bool SetShape(const Shape &in, Shape *out)
{
  if (in.type != out->type || in.dimensions.size() != out->dimensions.size())
  {
    return false;
  }
  out->dimensions = in.dimensions;
  return true;
}

uint32_t getNumberOfElements(const Shape &shape)
{
  uint32_t count = 1;
  for (size_t i = 0; i < shape.dimensions.size(); i++)
  {
    count *= shape.dimensions[i];
  }
  return count;
}

uint32_t getNumberOfDimensions(const Shape &shape) { return shape.dimensions.size(); }

uint32_t getSizeOfDimension(const Shape &shape, uint32_t dimensionIdx)
{
  if (dimensionIdx >= shape.dimensions.size())
  {
    // TODO, log the error
    return 0;
  }
  return shape.dimensions[dimensionIdx];
}
