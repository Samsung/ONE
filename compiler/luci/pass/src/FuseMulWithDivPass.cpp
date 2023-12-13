/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseMulWithDivPass.h"

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/IR/CircleNodes.h>

namespace luci
{

namespace
{

// Return a new CircleConst with a new value
luci::CircleConst *create_div_const_with_new_value(luci::CircleConst *div_const,
                                                   luci::CircleConst *mul_const, float new_value)
{
  assert(div_const);                                       // FIX_CALLER_UNLESS
  assert(div_const->dtype() == loco::DataType::FLOAT32);   // FIX_CALLER_UNLESS
  assert(div_const->size<loco::DataType::FLOAT32>() == 1); // FIX_CALLER_UNLESS

  auto new_div_const = div_const->graph()->nodes()->create<luci::CircleConst>();
  new_div_const->dtype(loco::DataType::FLOAT32);
  new_div_const->size<loco::DataType::FLOAT32>(1);
  new_div_const->rank(1);
  new_div_const->dim(0) = 1;
  new_div_const->at<loco::DataType::FLOAT32>(0) = new_value;
  new_div_const->shape_status(luci::ShapeStatus::VALID);
  new_div_const->name(div_const->name() + ";" + mul_const->name());

  luci::add_origin(new_div_const, luci::composite_origin(
                                    {luci::get_origin(div_const), luci::get_origin(mul_const)}));

  return new_div_const;
}

/**
 * Pass to fuse mul(one of the input is const scalar) and
 * div(numerator is const scalar) as div
 *
 * BEFORE
 *             [CircleNode]                                [Scalar_Mul_Const]
 *                  |                                               |
 *          [CirlceMul, (x=CircleNode, y=Scalar_Mul_Const)] --------
 *                  |
 *                  |                                     [Scalar_Div_Const]
 *                  |                                             |
 *           [CircleDiv, (x=Scalar_Div_Const, y=CirlceMul)] ------
 *                  |
 *             [CircleNode]
 *
 *  AFTER
 *            [CircleNode]
 *                 |                                          [Scalar_new_Div_Const]
 *                 |                                                   |
 *          [CircleDiv, (x=Scalar_new_Div_Const, y=CircleNode)] -------
 *                 |
 *            [CircleNode]
 *
 *          where Scalar_new_Div_Const = Scalar_Div_Const / Scalar_Mul_Const
 *
 **/
bool fuse_mul_with_div(luci::CircleDiv *div)
{
  auto div_const = dynamic_cast<luci::CircleConst *>(div->x());
  if (not div_const)
    return false;

  if (div_const->dtype() != loco::DataType::FLOAT32)
    return false;

  if (div_const->size<loco::DataType::FLOAT32>() != 1)
    return false;

  auto mul = dynamic_cast<luci::CircleMul *>(div->y());
  if (not mul)
    return false;

  auto mul_const = dynamic_cast<luci::CircleConst *>(mul->y());
  if (not mul_const)
    return false;

  if (mul_const->dtype() != loco::DataType::FLOAT32)
    return false;

  if (mul_const->size<loco::DataType::FLOAT32>() != 1)
    return false;

  const auto div_value = div_const->at<loco::DataType::FLOAT32>(0);
  const auto mul_value = mul_const->at<loco::DataType::FLOAT32>(0);

  if (mul_value == 0)
    return false;

  const auto new_value = div_value / mul_value;

  auto new_div_const = create_div_const_with_new_value(div_const, mul_const, new_value);

  div->x(new_div_const);

  div->y(mul->x());

  return true;
}

} // namespace

bool FuseMulWithDivPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto div = dynamic_cast<luci::CircleDiv *>(node);
    if (not div)
      continue;

    if (fuse_mul_with_div(div))
      changed = true;
  }

  return changed;
}

} // namespace luci
