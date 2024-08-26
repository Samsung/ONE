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

namespace
{

loco::TensorShape remove_last_two(const loco::TensorShape &original_shape)
{
  assert(original_shape.rank() >= 2); // FIX CALLER UNLESS

  loco::TensorShape ret;
  ret.rank(original_shape.rank() - 2);

  for (uint i = 0; i < ret.rank(); ++i)
  {
    ret.dim(i) = original_shape.dim(i);
  }
  return ret;
}

} // namespace

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::ABC>::visit(const luci::CircleBatchMatMul *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleBatchMatMul>();
  if (cloned != nullptr)
  {
    cloned->adj_x(node->adj_x());
    cloned->adj_y(node->adj_y());
  }
  return cloned;
}

// BatchMatMulV2 supports broadcasting in the batch dimensions(BatchMatMul doesn't)
// TODO Distinguish BatchMatMul and BatchMatMulV2
loco::TensorShape sinf::Algorithm::visit(const luci::CircleBatchMatMul *node)
{
  const auto x = loco::must_cast<CircleNode *>(node->x());
  const auto y = loco::must_cast<CircleNode *>(node->y());

  const auto x_shape = sinf::circle_shape(x);
  const auto y_shape = sinf::circle_shape(y);

  uint32_t x_rank = x_shape.rank();
  uint32_t y_rank = y_shape.rank();
  assert(x_rank >= 2 && y_rank >= 2);

  uint32_t max_rank = x_rank > y_rank ? x_rank : y_rank;
  loco::TensorShape output_shape;
  output_shape.rank(max_rank);

  // broadcast in the batch dimensions
  if (x_rank > 2 || y_rank > 2)
  {
    const auto x_batch_dims = remove_last_two(x_shape);
    const auto y_batch_dims = remove_last_two(y_shape);

    const auto o_batch_dims = sinf::broadcast_shape(x_batch_dims, y_batch_dims);

    const auto o_batch_rank = o_batch_dims.rank();
    for (uint i = 0u; i < o_batch_rank; ++i)
    {
      output_shape.dim(i) = o_batch_dims.dim(i);
    }
  }

  // shape inference in contracting dimensions
  auto adj_x = node->adj_x();
  auto adj_y = node->adj_y();

  loco::Dimension x_lhs = adj_x ? x_shape.dim(x_rank - 1) : x_shape.dim(x_rank - 2);
  loco::Dimension x_rhs = adj_x ? x_shape.dim(x_rank - 2) : x_shape.dim(x_rank - 1);
  loco::Dimension y_lhs = adj_y ? y_shape.dim(y_rank - 1) : y_shape.dim(y_rank - 2);
  loco::Dimension y_rhs = adj_y ? y_shape.dim(y_rank - 2) : y_shape.dim(y_rank - 1);

  if (x_rhs.known() && y_lhs.known() && not(x_rhs == y_lhs))
    INTERNAL_EXN("x_rhs and y_lhs should be same");

  uint32_t out_rank = output_shape.rank();
  output_shape.dim(out_rank - 2) = x_lhs;
  output_shape.dim(out_rank - 1) = y_rhs;

  return output_shape;
}

} // namespace luci
