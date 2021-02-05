/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Service/ShapeDescription.h"

#include <oops/InternalExn.h>

#include <cassert>

namespace luci
{

ShapeDescription to_shape_description(const luci::CircleNode *circle_node)
{
  ShapeDescription res;

  res._rank_known = true;

  res._dims.resize(circle_node->rank());
  for (uint32_t i = 0; i < circle_node->rank(); ++i)
    res._dims.at(i) = circle_node->dim(i).known() ? circle_node->dim(i).value() : -1;

  return res;
}

ShapeDescription to_shape_description(const loco::TensorShape &shape)
{
  ShapeDescription res;

  res._rank_known = true;

  res._dims.resize(shape.rank());
  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    // All the dimensions SHOULD be known
    assert(shape.dim(axis).known());
    res._dims.at(axis) = shape.dim(axis).value();
  }

  return res;
}

ShapeDescription to_shape_description(const loco::NodeShape &shape)
{
  switch (shape.domain())
  {
    case loco::Domain::Tensor:
      return to_shape_description(shape.as<loco::TensorShape>());
    default:
      break;
  }

  INTERNAL_EXN_V("Unsupported loco domain", oops::to_uint32(shape.domain()));
}

} // namespace luci
