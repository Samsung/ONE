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

#include "luci/Service/CircleShapeInference.h"

#include "CircleShapeInferenceHelper.h"

#include <loco.h>

#include <luci/Log.h>

#include <cassert>
#include <iostream>

namespace
{

std::ostream &operator<<(std::ostream &os, const loco::TensorShape &tensor_shape)
{
  os << "[";
  for (uint32_t r = 0; r < tensor_shape.rank(); ++r)
  {
    if (r)
      os << ",";

    if (tensor_shape.dim(r).known())
      os << tensor_shape.dim(r).value();
    else
      os << "?";
  }
  os << "]";
  return os;
}

bool inputs_shape_ready(const luci::CircleNode *node)
{
  for (uint32_t arity = 0; arity < node->arity(); ++arity)
  {
    auto node_input = loco::must_cast<luci::CircleNode *>(node->arg(arity));
    if (node_input->shape_status() == luci::ShapeStatus::UNDEFINED)
      return false;
  }

  return true;
}

} // namespace

namespace luci
{
namespace sinf
{

bool Rule::infer(const luci::CircleNode *circle_node, loco::TensorShape &shape) const
{
  LOGGER(l);
  VERBOSE(l, 1) << "[CircleShapeInference] " << circle_node->name();
  VERBOSE(l, 1) << "  before: " << circle_shape(circle_node);

  if (!inputs_shape_ready(circle_node))
  {
    VERBOSE(l, 1) << " after: Some inputs are not ready for inference";
    return false;
  }

  Algorithm alg;
  shape = circle_node->accept(&alg);
  VERBOSE(l, 1) << " after: " << shape;

  return true;
}

} // namespace sinf
} // namespace luci
