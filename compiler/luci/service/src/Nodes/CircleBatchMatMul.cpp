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

bool contain_zero(const loco::TensorShape &shape)
{
  bool zero_found = false;

  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    const auto &dim = shape.dim(axis);
    if (dim.known() && dim.value() == 0)
    {
      zero_found = true;
      break;
    }
  }
  return zero_found;
}

void throw_unless(bool condition_result, const char *exception_msg)
{
  if (not condition_result)
  {
    INTERNAL_EXN(exception_msg);
  }
}

} // namespace

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::ABC>::visit(const luci::CircleBatchMatMul *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleBatchMatMul>();
  {
    cloned->adj_x(node->adj_x());
    cloned->adj_y(node->adj_y());
  }
  return cloned;
}

namespace sinf
{

// BatchMatMulV2 supports broadcasting in the batch dimensions(BatchMatMul doesn't)
// TODO Distinguish BatchMatMul and BatchMatMulV2
loco::TensorShape Algorithm::visit(const luci::CircleBatchMatMul *node)
{
  const auto x = loco::must_cast<CircleNode *>(node->x());
  const auto y = loco::must_cast<CircleNode *>(node->y());

  const auto x_shape = circle_shape(x);
  const auto y_shape = circle_shape(y);

  uint32_t x_rank = x_shape.rank();
  uint32_t y_rank = y_shape.rank();

  // throw internal exception if condition not met
  throw_unless(x_rank >= 2, "x_rank shoud be >= 2");
  throw_unless(y_rank >= 2, "y_rank shoud be >= 2");
  throw_unless((not contain_zero(x_shape)), "x_shape should NOT have 0");
  throw_unless((not contain_zero(y_shape)), "y_shape should NOT have 0");

  // BatchMatMul shape inference rule works with two-part
  //
  // 1) Batch dimensions part
  //    - Batch dimensions correspond to the input_shape[:-2]
  //    - General broadcast rules are used to infer the shape of the output batch
  //
  // 2) Contracting dimensions part
  //    - Contracting dimensions correspond to the input_shape[-2:]
  //    - General matrix multiplication shape inference applied for this part,
  //      which means '(x_lhs, x_rhs) x (y_lhs, y_rhs) => (x_lhs, y_rhs)'

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
  const auto adj_x = node->adj_x();
  const auto adj_y = node->adj_y();

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

} // namespace sinf
} // namespace luci
