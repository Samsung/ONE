/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "internal/Model.h"

namespace internal
{
namespace tflite
{
namespace operand
{

Shape::Shape(uint32_t rank) : nnfw::misc::tensor::Shape(rank)
{
  // DO NOTHING
}

int32_t Shape::asVector(void) const
{
  assert(rank() == 1);

  return dim(0);
}

nnfw::misc::matrix::Shape Shape::asMatrix(void) const
{
  assert(rank() == 2);

  const auto height = dim(0);
  const auto width = dim(1);

  return nnfw::misc::matrix::Shape(height, width);
}

nnfw::misc::feature::Shape Shape::asFeature(void) const
{
  assert(rank() == 4);

  // Feature Map in NNAPI
  //  - Dimension(0) -> Batch
  //  - Dimension(1) -> Height
  //  - Dimension(2) -> Width
  //  - Dimension(3) -> Depth

  const auto batch = dim(0);
  const auto depth = dim(3);
  const auto height = dim(1);
  const auto width = dim(2);

  return nnfw::misc::feature::Shape(batch, depth, height, width);
}

nnfw::misc::tensor::Shape Shape::asTensor(void) const
{
  return nnfw::misc::tensor::Shape(*this); // this shape represents shape of NNAPI
}

nnfw::misc::kernel::Shape Shape::asKernel(void) const
{
  assert(rank() == 4);

  // Convolution Kernel in NNAPI
  //  - Dimension(0) -> Count
  //  - Dimension(1) -> Height
  //  - Dimension(2) -> Width
  //  - Dimension(3) -> Depth
  const auto count = dim(0);
  const auto depth = dim(3);
  const auto height = dim(1);
  const auto width = dim(2);

  return nnfw::misc::kernel::Shape(count, depth, height, width);
}

// Extended dimension is filled with 1.
void Shape::extendRank(size_t to_rank)
{
  for (int i = rank() + 1; i <= to_rank; ++i)
  {
    prepend(1);
  }
}

} // namespace operand
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace operand
{

Index Set::append(const Shape &shape, int32_t type, float scale, int32_t zeroPoint)
{
  int32_t index = _objects.size();

  _objects.emplace_back(new Object{shape, type, scale, zeroPoint});

  return Index{index};
}

const Object &Set::at(const Index &index) const { return *(_objects.at(index.asInt())); }

Object &Set::at(const Index &index) { return *(_objects.at(index.asInt())); }

bool Set::exist(const Index &index) const
{
  return index.asInt() >= 0 && index.asInt() < _objects.size();
}

} // namespace operand
} // namespace tflite
} // namespace internal
