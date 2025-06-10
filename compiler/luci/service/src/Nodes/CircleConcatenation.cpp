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

#include <luci/Service/CircleShapeInference.h>

#include "CircleCloneNode.h"

#include "CircleShapeInferenceHelper.h"

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::ABC>::visit(const luci::CircleConcatenation *node)
{
  if (node->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return nullptr;

  auto *cloned = _graph->nodes()->create<luci::CircleConcatenation>(node->numValues());
  {
    cloned->fusedActivationFunction(node->fusedActivationFunction());
    cloned->axis(node->axis());
  }
  return cloned;
}

namespace sinf
{

loco::TensorShape Algorithm::visit(const luci::CircleConcatenation *node)
{
  // TODO Support when CircleConcatenation has 0 input
  assert(node->numValues() > 0);

  auto first_shape = luci::shape_get(node->values(0)).as<loco::TensorShape>();
  auto axis = node->axis();
  if (axis < 0)
    axis += first_shape.rank();

  assert(0 <= axis);
  assert(first_shape.rank() > static_cast<uint32_t>(axis));

  loco::TensorShape output_shape;

  output_shape.rank(first_shape.rank());
  for (uint32_t i = 0; i < output_shape.rank(); ++i)
    output_shape.dim(i) = first_shape.dim(i);

  for (uint32_t i = 1; i < node->numValues(); ++i)
  {
    auto input_shape = luci::shape_get(node->values(i)).as<loco::TensorShape>();
    if (input_shape.rank() != output_shape.rank())
      INTERNAL_EXN_V("Input has incompatible shape", node->name());

    for (uint32_t j = 0; j < output_shape.rank(); ++j)
    {
      if (j == static_cast<uint32_t>(axis))
      {
        if (output_shape.dim(j).known() and input_shape.dim(j).known())
        {
          output_shape.dim(j) = output_shape.dim(j).value() + input_shape.dim(j).value();
        }
        else
        {
          // If any of inputs is unknown, just mark it as unknown.
          output_shape.dim(j).unset();
        }
      }
      else
      {
        if (output_shape.dim(j).known() and input_shape.dim(j).known())
        {
          if (output_shape.dim(j).value() != input_shape.dim(j).value())
          {
            INTERNAL_EXN_V("Input has incompatible shape.", node->name());
          }
        }
        else
        {
          if (input_shape.dim(j).known())
          {
            assert(not output_shape.dim(j).known()); // FIX_ME_UNLESS
            output_shape.dim(j) = input_shape.dim(j);
          }
          // For unknown input_shape, leave output_shape as-is
        }
      }
    }
  }

  return output_shape;
}

} // namespace sinf
} // namespace luci
