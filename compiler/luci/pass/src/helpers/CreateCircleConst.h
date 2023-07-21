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

#ifndef __LUCI_PASS_HELPERS_CREATE_CIRCLE_CONST_H__
#define __LUCI_PASS_HELPERS_CREATE_CIRCLE_CONST_H__

#include <luci/IR/CircleNodes.h>

#include "TypeMapper.h"

#include <vector>

namespace luci
{

// Create CircleConst filled with a single value
// Never return nullptr
// TODO Remove dtype from the argument
template <typename T>
CircleConst *create_const_node(loco::Graph *g, const loco::DataType dtype,
                               const std::vector<uint32_t> &shape, const T value)
{
  auto node = g->nodes()->create<CircleConst>();
  node->dtype(dtype);
  node->rank(shape.size());

  uint32_t size = 1;
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->dim(i) = shape.at(i);
    size *= shape.at(i);
  }
  node->shape_status(ShapeStatus::VALID);

  node->size<TypeMapper<T>::get()>(size);
  for (uint32_t i = 0; i < size; i++)
  {
    node->at<TypeMapper<T>::get()>(i) = value;
  }

  return node;
}

// Create CircleConst filled with values
// Never return nullptr
// TODO Remove dtype from the argument
template <typename T>
luci::CircleConst *create_const_node(loco::Graph *g, const loco::DataType dtype,
                                     const std::vector<uint32_t> &shape,
                                     const std::vector<T> &values)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  node->dtype(dtype);
  node->rank(shape.size());

  uint32_t size = 1;
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->dim(i) = shape.at(i);
    size *= shape.at(i);
  }
  node->shape_status(luci::ShapeStatus::VALID);

  node->size<TypeMapper<T>::get()>(size);
  for (uint32_t i = 0; i < size; i++)
  {
    node->at<TypeMapper<T>::get()>(i) = values[i];
  }

  return node;
}

} // namespace luci

#endif // __LUCI_PASS_HELPERS_CREATE_CIRCLE_CONST_H__
