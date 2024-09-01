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

  // loco::TensorShape shape_by_input;
  // {
  //   LUCI_ASSERT(node->shape(), "2nd input shape() should not be nullptr");

  //   // Only support node's shape() is CircleConst with S32
  //   // TODO support other node with other types
  //   auto const_shape_node = dynamic_cast<luci::CircleConst *>(node->shape());
  //   if (const_shape_node != nullptr)
  //   {
  //     LUCI_ASSERT(const_shape_node->dtype() == S32, "Only support int32 CircleConst");

  //     shape_by_input.rank(const_shape_node->size<S32>());

  //     for (uint32_t axis = 0; axis < shape_by_input.rank(); ++axis)
  //     {
  //       shape_by_input.dim(axis) = const_shape_node->at<S32>(axis);
  //       if (const_shape_node->at<S32>(axis) < 0)
  //         shape_by_input.dim(axis).unset();
  //     }
  //   }
  //   else
  //   {
  //     // We use shape from the node itself
  //     shape_by_input = own_shape(node);
  //   }
  // }

  // loco::TensorShape shape_by_attr;
  // {
  //   shape_by_attr.rank(node->newShape()->rank());

  //   for (uint32_t axis = 0; axis < shape_by_attr.rank(); ++axis)
  //   {
  //     shape_by_attr.dim(axis) = node->newShape()->dim(axis);
  //   }
  // }

  // if (!(shape_by_input == shape_by_attr))
  // {
  //   INFO(l) << "CircleReshape: Two new shape information mismatched : " << std::endl;
  //   INFO(l) << "   shape_by_input : " << shape_by_input << std::endl;
  //   INFO(l) << "   shape_by_attr : " << shape_by_attr << std::endl;
  // }

  // loco::TensorShape output_shape = shape_by_input;

  loco::TensorShape output_shape;
  {
    const auto *shape_by_attr = node->newShape();
    
    if (shape_by_attr != nullptr)
    {
      output_shape.rank(shape_by_attr->rank());
      for (uint32_t axis = 0; axis < output_shape.rank(); ++axis)
      {
        output_shape.dim(axis) = shape_by_attr->dim(axis);
        if (shape_by_attr->dim(axis) < 0)
          output_shape.dim(axis).unset();
      }
    }
    else
    {
      node->shape();
    }
  }

  loco::TensorShape output_shape;
  {
    LUCI_ASSERT(node->shape(), "2nd input shape() should not be nullptr");
    // Only support node's shape() is CircleConst with S32
    // https://github.com/tensorflow/tensorflow/blob/2fe29379f6449d263b787eb636020fd81017b8bf/tensorflow/lite/kernels/reshape.cc#L130

    auto const_shape_node = dynamic_cast<luci::CircleConst *>(node->shape());
    std::cout << "node->shape(): " << node->shape() << std::endl;
    std::cout << "node->newShape(): " << node->newShape() << std::endl;
    std::cout << "const_shape_node: " << const_shape_node << std::endl;


    if (const_shape_node != nullptr)
    {
      LUCI_ASSERT(const_shape_node->dtype() == S32, "Only support int32 CircleConst");
      output_shape.rank(const_shape_node->size<S32>());
      for (uint32_t axis = 0; axis < output_shape.rank(); ++axis)
      {
        output_shape.dim(axis) = const_shape_node->at<S32>(axis);
        if (const_shape_node->at<S32>(axis) < 0)
          output_shape.dim(axis).unset();
      }
    }
    else
    {
      std::cout << "const_shape_node IS nullptr " << std::endl;
      std::cout << "node->newShape(): " << node->newShape() << std::endl;
      std::cout << "node->newShape()->rank(): " << node->newShape()->rank() << std::endl;
      output_shape.rank(node->newShape()->rank());
      for (uint32_t axis = 0; axis < output_shape.rank(); ++axis)
      {
        output_shape.dim(axis) = node->newShape()->dim(axis);
      }
    }
  }

  // One of the dimensions of input shape can have special value -1,
  // which should not be inferred.
  const auto input_shape = circle_shape(loco::must_cast<CircleNode *>(node->tensor()));

  uint32_t input_unknown_dim_index = UINT32_MAX;
  uint32_t input_element_count = 1;
  for (uint32_t i = 0; i < input_shape.rank(); i++)
  {
    input_element_count *= (input_shape.dim(i).known() ? input_shape.dim(i).value() : 1);
    if (!input_shape.dim(i).known())
    {
      if (input_unknown_dim_index != UINT32_MAX)
        INTERNAL_EXN("More than one unknown dimension");
      input_unknown_dim_index = i;
    }
  }

  // One of the dimensions of output shape can have special value -1,
  // meaning its actual value might be inferred.
  uint32_t output_unknown_dim_index = UINT32_MAX;
  uint32_t output_element_count = 1;
  for (uint32_t i = 0; i < output_shape.rank(); i++)
  {
    output_element_count *= (output_shape.dim(i).known() ? output_shape.dim(i).value() : 1);
    if (!output_shape.dim(i).known())
    {
      if (output_unknown_dim_index != UINT32_MAX)
        INTERNAL_EXN("More than one unknown dimension");
      output_unknown_dim_index = i;
    }
  }

  if (output_unknown_dim_index != UINT32_MAX && input_unknown_dim_index == UINT32_MAX)
  {
    output_shape.dim(output_unknown_dim_index) = input_element_count / output_element_count;
  }

  return output_shape;
}
} // namespace sinf

} // namespace luci
