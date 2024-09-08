/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_CIRCLE_SHAPE_INFERENCE_HELPER_H__
#define __LUCI_CIRCLE_SHAPE_INFERENCE_HELPER_H__

#include <loco/IR/NodeShape.h>
#include <loco/IR/TensorShape.h>

#include <luci/IR/CircleNodes.h>

namespace luci
{

// NOTE Functions in this namespace will be removed after new inference
//      algorithms are fully implemented.

// This function is temporary function for deprecating loco::shape_get
loco::NodeShape shape_get(const loco::Node *node);

// This function is temporary function for deprecating loco::shape_known
bool shape_known(const loco::Node *node);

} // namespace luci

namespace luci
{
namespace sinf // Namespace for Shape Inference
{

// Return shape of circle node as loco::TensorShape
loco::TensorShape circle_shape(const luci::CircleNode *node);

// Return broadcasted shape of output from two input shapes
// Throw an exception if x and y are not broadcastable.
loco::TensorShape broadcast_shape(const loco::TensorShape &x, const loco::TensorShape &y);

// Return shape of pad ops using paddings.
// If paddings is not static, return the shape filled with unknown dimensions.
loco::TensorShape pad_shape(const loco::TensorShape &input_shape, const luci::CircleNode *paddings);

/**
 * @brief Create a higher-rank TensorShape following NumPy broadcasting semantics
 *
 * HOW TO USE:
 *
 *   auto expanded_tensor_shape = expand(tensor_shape).to(N);
 */
class TensorShapeExpander final
{
public:
  TensorShapeExpander(const loco::TensorShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  loco::TensorShape to(uint32_t output_rank)
  {
    auto const &input_shape = _shape;
    uint32_t const input_rank = input_shape.rank();

    assert(input_rank <= output_rank && "Cannot shrink rank");
    uint32_t const axis_shift = output_rank - input_rank;

    loco::TensorShape output_shape;

    output_shape.rank(output_rank);
    for (uint32_t axis = 0; axis < output_rank; ++axis)
    {
      output_shape.dim(axis) = (axis < axis_shift) ? 1 : input_shape.dim(axis - axis_shift);
    }

    return output_shape;
  }

private:
  const loco::TensorShape _shape;
};

} // namespace sinf
} // namespace luci

#endif // __LUCI_CIRCLE_SHAPE_INFERENCE_HELPER_H__
