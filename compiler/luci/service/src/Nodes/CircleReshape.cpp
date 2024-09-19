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

#include <luci/Log.h>

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

} // namespace

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

loco::TensorShape Algorithm::visit(const luci::CircleReshape *node)
{
  LOGGER(l);

  const loco::DataType S32 = loco::DataType::S32;

  loco::TensorShape shape_by_input;
  {
    LUCI_ASSERT(node->shape(), "2nd input shape() should not be nullptr");

    // Only support node's shape() is CircleConst with S32
    // TODO support other node with other types
    auto const_shape_node = dynamic_cast<luci::CircleConst *>(node->shape());
    if (const_shape_node != nullptr)
    {
      LUCI_ASSERT(const_shape_node->dtype() == S32, "Only support int32 CircleConst");

      shape_by_input.rank(const_shape_node->size<S32>());

      for (uint32_t axis = 0; axis < shape_by_input.rank(); ++axis)
      {
        shape_by_input.dim(axis) = const_shape_node->at<S32>(axis);
        if (const_shape_node->at<S32>(axis) < 0)
        {
          shape_by_input.dim(axis).unset();
        }
      }
    }
    else
    {
      // We use shape from the node itself
      loco::TensorShape shape;
      shape.rank(node->rank());
      for (uint32_t r = 0; r < node->rank(); ++r)
      {
        // TODO remove this copy from `use_own(node);`
        // Shape inference rules in this file did not consider unknown dimension.
        // If some node has unknown dimension, 0 is inserted and wrong shape
        // inference was done as a result.
        // To fix this, new shape inference algorithm is being implemented.
        // Until new inference algorithm is fully implemented, unknown dimension
        // would be represented as 1 along with TFLite expression.
        shape.dim(r) = node->dim(r).known() ? node->dim(r).value() : 1;
      }
      shape_by_input = shape;
    }
  }

  loco::TensorShape shape_by_attr;
  {
    shape_by_attr.rank(node->newShape()->rank());

    for (uint32_t axis = 0; axis < shape_by_attr.rank(); ++axis)
    {
      shape_by_attr.dim(axis) = node->newShape()->dim(axis);
    }
  }

  if (!(shape_by_input == shape_by_attr))
  {
    INFO(l) << "CircleReshape: Two new shape information mismatched : " << std::endl;
    INFO(l) << "   shape_by_input : " << shape_by_input << std::endl;
    INFO(l) << "   shape_by_attr : " << shape_by_attr << std::endl;
  }

  loco::TensorShape output_shape = shape_by_input;

  // One of the dimensions can have special value -1, meaning its actual value should be inferred.
  const auto input = loco::must_cast<luci::CircleNode *>(node->tensor());
  const auto input_shape = circle_shape(input);
  uint32_t input_element_count = 1;
  uint32_t output_element_count = 1;
  uint32_t unknown_dim_index = UINT32_MAX;
  for (uint32_t i = 0; i < input_shape.rank(); ++i)
    input_element_count *= (input_shape.dim(i).known() ? input_shape.dim(i).value() : 1);
  for (uint32_t dim_index = 0; dim_index < output_shape.rank(); ++dim_index)
  {
    const uint32_t dim_value = output_shape.dim(dim_index).value();
    if (output_shape.dim(dim_index).known() == false)
    {
      LUCI_ASSERT(unknown_dim_index == UINT32_MAX, "More than one unknown dimension");
      unknown_dim_index = dim_index;
    }
    else
    {
      output_element_count *= dim_value;
    }
  }
  if (unknown_dim_index != UINT32_MAX)
  {
    output_shape.dim(unknown_dim_index) = input_element_count / output_element_count;
  }

  return output_shape;
}

} // namespace sinf

} // namespace luci
