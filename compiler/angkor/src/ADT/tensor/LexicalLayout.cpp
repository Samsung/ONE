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

#include "nncc/core/ADT/tensor/LexicalLayout.h"

#include <cassert>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::Shape;

// NOTE This forward declaration is introduced to minimize code diff
static uint32_t lexical_offset(const Shape &shape, const Index &index)
{
  assert(shape.rank() > 0);
  assert(shape.rank() == index.rank());

  const uint32_t rank = shape.rank();

  uint32_t res = index.at(0);

  for (uint32_t axis = 1; axis < rank; ++axis)
  {
    res *= shape.dim(axis);
    res += index.at(axis);
  }

  return res;
}

namespace nncc
{
namespace core
{
namespace ADT
{
namespace tensor
{

LexicalLayout::LexicalLayout() : Layout(lexical_offset)
{
  // DO NOTHING
}

} // namespace tensor
} // namespace ADT
} // namespace core
} // namespace nncc
