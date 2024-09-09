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
#include "Check.h"

#include "CircleShapeInferenceHelper.h"
#include "CircleCloneNode.h"

#include <oops/InternalExn.h>

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::OPQR>::visit(const luci::CircleReshape *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleReshape>();
  if (cloned != nullptr)
  {
    uint32_t rank = node->newShape()->rank();
    cloned->newShape()->rank(rank);
    for (uint32_t r = 0; r < rank; ++r)
    {
      cloned->newShape()->dim(r) = node->newShape()->dim(r);
    }
  }
  return cloned;
}

namespace sinf
{

/**
 * @note  CircleReshape always has two inputs: tensor and shape.
 *        The shape input can be CircleConst, CircleOutputDummy, or CircleNode.
 *        - If the shape input is CircleConst, the shape is inferred from the constant.
 *        - If the shape input is CircleOutputDummy, the shape is inferred from
 *          the attribute if it exists. If the attribute does not exist,
 *          the shape is inferred from the node iteself.
 *        - If the shape input is CircleNode, the shape is not inferred.
 */
loco::TensorShape Algorithm::visit(const luci::CircleReshape *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  // CircleReshape node must have reshape/shape
  if (node->shape() == nullptr)
  {
    INTERNAL_EXN("2nd input shape() should not be nullptr");
  }

  bool should_infer = true;
  loco::TensorShape output_shape;
  {
    // Check if reshape/shape is CircleConst
    auto const_input = dynamic_cast<luci::CircleConst *>(node->shape());
    if (const_input != nullptr)
    {
      output_shape.rank(const_input->size<S32>());

      for (uint32_t axis = 0; axis < output_shape.rank(); ++axis)
      {
        output_shape.dim(axis) = const_input->at<S32>(axis);
        if (const_input->at<S32>(axis) < 0)
        {
          output_shape.dim(axis).unset();
        }
      }
    }
    else
    {
      // Check if reshape/shape is CircleOutputDummy
      auto dummy_input = dynamic_cast<luci::CircleOutputDummy *>(node->shape());
      if (dummy_input != nullptr)
      {
        if (node->newShape()->rank() > 0)
        {
          output_shape.rank(node->newShape()->rank());

          for (uint32_t axis = 0; axis < output_shape.rank(); ++axis)
          {
            output_shape.dim(axis) = node->newShape()->dim(axis);
            if (node->newShape()->dim(axis) < 0)
            {
              output_shape.dim(axis).unset();
            }
          }
        }
        else
        {
          output_shape = circle_shape(node);
        }
      }
      else
      {
        // Check if reshape/shape is CircleNode
        auto node_input = dynamic_cast<luci::CircleNode *>(node->shape());
        if (node_input != nullptr)
        {
          output_shape.rank(node_input->dim(0).value());

          for (uint32_t axis = 0; axis < output_shape.rank(); ++axis)
          {
            output_shape.dim(axis).unset();
          }

          should_infer = false;
        }
      }
    }
  }

  const auto input = loco::must_cast<luci::CircleNode *>(node->tensor());
  const auto input_shape = circle_shape(input);
  uint32_t input_element_count = 1;
  for (uint32_t axis = 0; axis < input_shape.rank(); ++axis)
  {
    if (input_shape.dim(axis).known())
    {
      input_element_count *= input_shape.dim(axis).value();
    }
    else
    {
      should_infer = false;
      break;
    }
  }

  if (should_infer)
  {
    uint32_t output_element_count = 1;
    uint32_t unknown_dim_index = UINT32_MAX;
    for (uint32_t dim_index = 0; dim_index < output_shape.rank(); ++dim_index)
    {
      if (output_shape.dim(dim_index).known() == false)
      {
        if (unknown_dim_index != UINT32_MAX)
        {
          INTERNAL_EXN("More than one unknown dimension");
        }
        unknown_dim_index = dim_index;
      }
      else
      {
        const uint32_t dim_value = output_shape.dim(dim_index).value();
        output_element_count *= dim_value;
      }
    }
    if (unknown_dim_index != UINT32_MAX)
    {
      output_shape.dim(unknown_dim_index) = input_element_count / output_element_count;
    }
  }

  return output_shape;
}

} // namespace sinf

} // namespace luci
