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

#ifndef __SHAPE_H__
#define __SHAPE_H__

#include "OperandType.h"

#include <vector>
#include <cstdint>

// The type and dimensions of an operand.
struct Shape
{
  OperandType type;
  std::vector<uint32_t> dimensions;
  float scale;
  int32_t offset;
};

// Verifies that the two shapes are the same.
bool SameShape(const Shape &in1, const Shape &in2);

// Sets out to the same shape as in.
bool SetShape(const Shape &in, Shape *out);

// Return the total number of elements, i.e. all the dimensions multiplied
// together. For a scalar, returns one.
uint32_t getNumberOfElements(const Shape &shape);
uint32_t getNumberOfDimensions(const Shape &shape);
uint32_t getSizeOfDimension(const Shape &shape, uint32_t dimensionIdx);

#endif // __SHAPE_H__
