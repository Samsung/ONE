/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Service/CircleShapeInference.h"

#include "CircleCloneNode.h"
#include "CircleShapeInferenceHelper.h"

#include <cmath>

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::OPQR>::visit(const luci::CircleRange *)
{
  return _graph->nodes()->create<luci::CircleRange>();
}

namespace sinf
{

loco::TensorShape Algorithm::visit(const luci::CircleRange *node)
{
  loco::TensorShape output_shape;
  output_shape.rank(1);

  auto start_node = dynamic_cast<luci::CircleConst *>(node->start());
  auto limit_node = dynamic_cast<luci::CircleConst *>(node->limit());
  auto delta_node = dynamic_cast<luci::CircleConst *>(node->delta());

  if (start_node == nullptr || limit_node == nullptr || delta_node == nullptr)
  {
    // We use shape from the node itself
    loco::TensorShape shape;
    shape.rank(node->rank());
    for (uint32_t r = 0; r < node->rank(); ++r)
    {
      // Shape inference rules in this file did not consider unknown dimension.
      // If some node has unknown dimension, 0 is inserted and wrong shape
      // inference was done as a result.
      // To fix this, new shape inference algorithm is being implemented.
      // Until new inference algorithm is fully implemented, unknown dimension
      // would be represented as 1 along with TFLite expression.
      shape.dim(r) = node->dim(r).known() ? node->dim(r).value() : 1;
    }
    return shape;
  }

  double start = 0, limit = 0, delta = 0;

#define GET_RANGE_PARAM(DT)         \
  start = start_node->scalar<DT>(); \
  limit = limit_node->scalar<DT>(); \
  delta = delta_node->scalar<DT>();

  switch (start_node->dtype())
  {
    case loco::DataType::FLOAT32:
      GET_RANGE_PARAM(loco::DataType::FLOAT32)
      break;
    case loco::DataType::S32:
      GET_RANGE_PARAM(loco::DataType::S32)
      break;
    default:
      INTERNAL_EXN("Range data type not supported");
  }

#undef GET_RANGE_PARAM

  if (delta == 0)
    INTERNAL_EXN("Delta can not be zero");

  output_shape.dim(0) = ceil((limit - start) / delta);

  return output_shape;
}

} // namespace sinf

} // namespace luci
