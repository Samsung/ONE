/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Shape.h"

namespace luci
{

bool is_same_shape(const luci::CircleNode *node, const loco::TensorShape &shape)
{
  if (node == nullptr)
    return false;

  if (node->shape_status() != luci::ShapeStatus::VALID)
    return false;

  if (node->rank() != shape.rank())
    return false;

  for (uint32_t i = 0; i < node->rank(); ++i)
  {
    if (node->dim(i).known() != shape.dim(i).known())
      return false;

    if (node->dim(i).value() != shape.dim(i).value())
      return false;
  }

  return true;
}

bool is_same_shape(const luci::CircleNode *node, const std::initializer_list<uint32_t> shape)
{
  if (node == nullptr)
    return false;

  if (node->rank() != shape.size())
    return false;

  uint32_t i = 0;
  for (auto it = shape.begin(); it != shape.end(); ++it, ++i)
  {
    if (node->dim(i).value() != *it)
      return false;
  }
  return true;
}

} // namespace luci
