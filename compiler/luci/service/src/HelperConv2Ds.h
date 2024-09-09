/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Check.h"
#include "CircleShapeInferenceHelper.h"

namespace luci
{

namespace sinf
{

template <class Conv2DType> loco::TensorShape conv2d_output_shape(const Conv2DType *node)
{
  auto ifm_shape = circle_shape(loco::must_cast<const luci::CircleNode *>(node->input()));
  auto ker_shape = circle_shape(loco::must_cast<const luci::CircleNode *>(node->filter()));
  assert(ifm_shape.rank() == 4);
  assert(ker_shape.rank() == 4);
  assert(ker_shape.dim(1).known());
  assert(ker_shape.dim(2).known());

  // Output shape is 4D in NHWC order.
  // This will only specify output shape for height and width.
  // The other two dimensions (batch and channel) will be set by caller.
  loco::TensorShape output_shape;
  output_shape.rank(4);

  if (ifm_shape.dim(1).known())
  {
    uint32_t input_height = ifm_shape.dim(1).value();
    uint32_t ker_height = ker_shape.dim(1).value();
    uint32_t stride_height = node->stride()->h();
    uint32_t dilation_height = node->dilation()->h();
    uint32_t effective_ker_height = dilation_height * (ker_height - 1) + 1;

    if (node->padding() == luci::Padding::VALID)
    {
      LUCI_ASSERT(input_height + stride_height > effective_ker_height, "Invalid shape");
      output_shape.dim(1).set((input_height + stride_height - effective_ker_height) /
                              stride_height);
    }
    else if (node->padding() == luci::Padding::SAME)
    {
      output_shape.dim(1).set((input_height + stride_height - 1) / stride_height);
    }
    else
    {
      LUCI_ASSERT(false, "Wrong padding type");
    }
  }

  if (ifm_shape.dim(2).known())
  {
    uint32_t input_width = ifm_shape.dim(2).value();
    uint32_t ker_width = ker_shape.dim(2).value();
    uint32_t stride_width = node->stride()->w();
    uint32_t dilation_width = node->dilation()->w();
    uint32_t effective_ker_width = dilation_width * (ker_width - 1) + 1;

    if (node->padding() == luci::Padding::VALID)
    {
      LUCI_ASSERT(input_width + stride_width > effective_ker_width, "Invalid shape");
      output_shape.dim(2).set((input_width + stride_width - effective_ker_width) / stride_width);
    }
    else if (node->padding() == luci::Padding::SAME)
    {
      output_shape.dim(2).set((input_width + stride_width - 1) / stride_width);
    }
    else
    {
      LUCI_ASSERT(false, "Wrong padding type");
    }
  }

  return output_shape;
}

} // namespace sinf

} // namespace luci
