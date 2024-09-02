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

#include "CircleCloneNode.h"
#include "CircleShapeInferenceHelper.h"

#include <luci/Log.h>
#include "CircleShapeInferenceHelper.h"

#include <luci/Log.h>

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
 * @note  CircleReshape has new shape info in two places: 2nd input and attribute.
 *        This shape inference uses shape from input 'shape' node when it's constant.
 *        If not, shape will be from node itself. shape from attribute is not used.
 *
 * TODO Change this policy when not appropriate
 */
loco::TensorShape Algorithm::visit(const luci::CircleReshape *node)
{
  LOGGER(l);

  const loco::DataType S32 = loco::DataType::S32;

  bool is_input_known = true;
  auto input_tensor = dynamic_cast<luci::CircleNode *>(node->tensor());
  for (auto axis = 0; axis < input_tensor->rank(); ++axis)
  {
    if (!input_tensor->dim(axis).known())
    {
      is_input_known = false;
      break;
    }
  }

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
          shape_by_input.dim(axis).unset();
      }
    }
    else
    {
      // We use shape from the node itself
      shape_by_input = own_shape(node);
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
  if (is_input_known)
  {
    const auto input_shape = luci::shape_get(node->tensor()).as<loco::TensorShape>();
    uint32_t input_element_count = 1;
    uint32_t output_element_count = 1;
    uint32_t unknown_dim_index = UINT32_MAX;
    for (uint32_t i = 0; i < input_shape.rank(); ++i)
      input_element_count *= (input_shape.dim(i).known() ? input_shape.dim(i).value() : 1);
    for (uint32_t dim_index = 0; dim_index < output_shape.rank(); ++dim_index)
    {
      if (!output_shape.dim(dim_index).known())
      {
        if (unknown_dim_index != UINT32_MAX)
          INTERNAL_EXN("More than one unknown dimension");
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
