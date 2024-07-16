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

#include "luci/Pass/FoldMulPass.h"

#include <luci/IR/CircleNodes.h>

#include <algorithm>

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

namespace
{

/**
 * @return higher rank of x, y or nullptr if not compatible
 */
const luci::CircleConst *compatible_shape(const luci::CircleConst *x, const luci::CircleConst *y)
{
  if (x->rank() >= y->rank())
  {
    uint32_t d = x->rank() - y->rank();
    for (uint32_t i = 0; i < y->rank(); i++)
    {
      // NOTE dim() has only '==' operator
      if (!(x->dim(i + d) == y->dim(i)))
        return nullptr;
    }
    return x;
  }
  else
  {
    uint32_t d = y->rank() - x->rank();
    for (uint32_t i = 0; i < x->rank(); i++)
    {
      if (!(x->dim(i) == y->dim(i + d)))
        return nullptr;
    }
    return y;
  }
}

/**
 * Fold Mul to const if both inputs are const
 **/
bool fold_mul(luci::CircleMul *mul)
{
  CHECK_OR_FALSE(mul);
  CHECK_OR_FALSE(mul->dtype() == loco::DataType::FLOAT32);

  // Check inputs are const and compatible
  auto x = dynamic_cast<luci::CircleConst *>(mul->x());
  auto y = dynamic_cast<luci::CircleConst *>(mul->y());
  CHECK_OR_FALSE(x);
  CHECK_OR_FALSE(y);
  CHECK_OR_FALSE(x->dtype() == y->dtype());
  const auto xy = compatible_shape(x, y);
  CHECK_OR_FALSE(xy);

  auto name_x = x->name();
  auto name_y = y->name();
  assert(name_x.length() > 0);
  assert(name_y.length() > 0);
  auto folded_const = mul->graph()->nodes()->create<luci::CircleConst>();
  folded_const->dtype(xy->dtype());
  folded_const->rank(xy->rank());
  for (uint32_t i = 0; i < xy->rank(); i++)
    folded_const->dim(i).set(xy->dim(i).value());

  const auto size_x = x->size<loco::DataType::FLOAT32>();
  const auto size_y = y->size<loco::DataType::FLOAT32>();
  const auto size_xy = xy->size<loco::DataType::FLOAT32>();
  folded_const->size<loco::DataType::FLOAT32>(size_xy);
  for (uint32_t i = 0; i < size_xy; i++)
  {
    auto xv = x->at<loco::DataType::FLOAT32>(i % size_x);
    auto yv = y->at<loco::DataType::FLOAT32>(i % size_y);
    folded_const->at<loco::DataType::FLOAT32>(i) = xv * yv;
  }

  folded_const->shape_status(luci::ShapeStatus::VALID);
  folded_const->name(name_x + "_" + name_y);

  loco::replace(mul).with(folded_const);

  return true;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for Mul Op
 **/
bool FoldMulPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto mul = dynamic_cast<luci::CircleMul *>(node))
    {
      if (fold_mul(mul))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
