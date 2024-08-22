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

#include "CircleShapeInferenceHelper.h"

#include <oops/InternalExn.h>

using namespace luci::sinf;

namespace
{

/**
 * @brief  Expand shape x and y to same rank by align right and filling with 1
 */
void expand_rank(loco::TensorShape &x, loco::TensorShape &y)
{
  auto x_rank = x.rank();
  auto y_rank = y.rank();

  if (x_rank == y_rank)
    return;

  TensorShapeExpander x_exp(x);
  TensorShapeExpander y_exp(y);

  auto xy_rank = std::max(x_rank, y_rank);

  x = x_rank > y_rank ? x : x_exp.to(xy_rank);
  y = y_rank > x_rank ? y : y_exp.to(xy_rank);
}

/**
 * @brief  Return shape of expanded dimension of input x and y having same rank
 */
loco::TensorShape expand_dimension(const loco::TensorShape &x, const loco::TensorShape &y)
{
  assert(x.rank() == y.rank()); // FIX_CALLER_UNLESS

  auto rank = x.rank();

  loco::TensorShape output_shape;

  output_shape.rank(rank);

  // Shape inference rule (commutative)
  // Values of x, y, and Result are categorized as 1, X, ? where
  // X is a positive integer greater than 1 and ? is unknown.
  //     x | y | Result
  // c1. 1 | X | X
  // c2. 1 | ? | ?
  // c3. X | X | X
  // c4. X | ? | X
  // c5. ? | ? | ?
  // Throw exception if none of the below conditions are met
  // e1. x == y
  // e2. x != y and x == 1
  // e3. x != y and y == 1
  // e4. x is unknown
  // e5. y is unknown
  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    const bool x_is_known = x.dim(axis).known();
    const bool y_is_known = y.dim(axis).known();

    if (x_is_known and y_is_known)
    {
      const auto x_dim = x.dim(axis).value();
      const auto y_dim = y.dim(axis).value();

      // each dimension of x and y should be same or one must be 1 if different
      if (!((x_dim == y_dim) || (x_dim == 1 || y_dim == 1)))
        INTERNAL_EXN("Cannot produce expand_dimension of two shapes");

      // c1, c3
      output_shape.dim(axis).set(std::max(x_dim, y_dim));
    }
    else if (not x_is_known and not y_is_known)
    {
      // c5
      output_shape.dim(axis).unset();
    }
    else
    {
      const uint32_t known_dim = x_is_known ? x.dim(axis).value() : y.dim(axis).value();
      if (known_dim == 1)
      {
        // c2
        output_shape.dim(axis).unset();
      }
      else
      {
        // c4
        output_shape.dim(axis).set(known_dim);
      }
    }
  }

  return output_shape;
}

} // namespace

namespace luci
{

loco::NodeShape shape_get(const loco::Node *node)
{
  assert(luci::shape_known(node));
  return loco::NodeShape{sinf::circle_shape(loco::must_cast<const luci::CircleNode *>(node))};
}

bool shape_known(const loco::Node *node)
{
  return loco::must_cast<const luci::CircleNode *>(node)->shape_status() !=
         luci::ShapeStatus::UNDEFINED;
}

} // namespace luci

namespace luci
{
namespace sinf
{

loco::TensorShape circle_shape(const luci::CircleNode *node)
{
  loco::TensorShape shape;
  shape.rank(node->rank());
  for (uint32_t r = 0; r < node->rank(); ++r)
    shape.dim(r) = node->dim(r);
  return shape;
}

loco::TensorShape broadcast_shape(const loco::TensorShape &x, const loco::TensorShape &y)
{
  auto x_match = x;
  auto y_match = y;

  expand_rank(x_match, y_match);

  auto output_shape = expand_dimension(x_match, y_match);

  return output_shape;
}

} // namespace sinf
} // namespace luci
