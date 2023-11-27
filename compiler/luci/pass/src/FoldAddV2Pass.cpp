/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FoldAddV2Pass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool same_shape(const luci::CircleConst *x, const luci::CircleConst *y)
{
  if (x->rank() != y->rank())
    return false;

  for (uint32_t i = 0; i < x->rank(); i++)
  {
    if (!(x->dim(i) == y->dim(i)))
      return false;
  }

  return true;
}

/**
 * Fold AddV2 to const if both inputs are const
 **/
template <loco::DataType T> bool fold_add_v2(luci::CircleCustom *add_v2)
{
  // This should hold for AddV2
  if (add_v2->numInputs() != 2)
    return false;

  // Check first input is const
  auto x = dynamic_cast<luci::CircleConst *>(add_v2->inputs(0));
  if (not x)
    return false;

  // Check second input is const
  auto y = dynamic_cast<luci::CircleConst *>(add_v2->inputs(1));
  if (not y)
    return false;

  if (x->dtype() != y->dtype())
    return false;

  if (!same_shape(x, y))
    return false;

  auto name_x = x->name();
  auto name_y = y->name();
  assert(name_x.length() > 0);
  assert(name_y.length() > 0);
  auto constant = add_v2->graph()->nodes()->create<luci::CircleConst>();
  constant->dtype(x->dtype());
  constant->rank(x->rank());
  for (uint32_t i = 0; i < x->rank(); i++)
    constant->dim(i).set(x->dim(i).value());

  const auto size = x->size<T>();
  constant->size<T>(size);
  for (uint32_t i = 0; i < size; i++)
    constant->at<T>(i) = x->at<T>(i) + y->at<T>(i);

  constant->shape_status(luci::ShapeStatus::VALID);
  constant->name(name_x + ";" + name_y);

  for (auto succ : loco::succs(add_v2))
  {
    auto custom_out = loco::must_cast<luci::CircleCustomOut *>(succ);
    loco::replace(custom_out).with(constant);
  }

  return true;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for AddV2 Op
 **/
bool FoldAddV2Pass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto custom = dynamic_cast<luci::CircleCustom *>(node))
    {
      if (custom->custom_code() == "AddV2")
      {
        // TODO: Support more data types
        if (custom->dtype() == loco::DataType::S64)
        {
          if (fold_add_v2<loco::DataType::S64>(custom))
            changed = true;
        }
      }
    }
  }

  return changed;
}

} // namespace luci
