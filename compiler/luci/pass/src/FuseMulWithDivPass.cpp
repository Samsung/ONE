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

#include "helpers/NodeFiller.h"

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

// Return a new CircleConst with a new value
luci::CircleConst *create_mul_const_with_new_value(luci::CircleConst *mul_const,
                                                   luci::CircleConst *div_const, float new_value)
{
  assert(mul_const);                                       // FIX_CALLER_UNLESS
  assert(mul_const->dtype() == loco::DataType::FLOAT32);   // FIX_CALLER_UNLESS
  assert(mul_const->size<loco::DataType::FLOAT32>() == 1); // FIX_CALLER_UNLESS

  auto new_mul_const = mul_const->graph()->nodes()->create<luci::CircleConst>();
  new_mul_const->dtype(loco::DataType::FLOAT32);
  new_mul_const->rank(0);
  new_mul_const->size<loco::DataType::FLOAT32>(1);
  new_mul_const->scalar<loco::DataType::FLOAT32>() = new_value;
  new_mul_const->shape_status(luci::ShapeStatus::VALID);
  new_mul_const->name(mul_const->name() + ";" + div_const->name());

  luci::add_origin(new_mul_const, luci::composite_origin(
                                    {luci::get_origin(mul_const), luci::get_origin(div_const)}));

  return new_mul_const;
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
bool fuse_mul_with_div_to_div(luci::CircleDiv *div)
{
  if (div->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  luci::CircleConst *div_const = nullptr;
  luci::CircleMul *mul = nullptr;
  if (not luci::fill(&div_const, &mul).with_args_of(div))
    return false;

  if (div_const->dtype() != loco::DataType::FLOAT32)
    return false;

  if (div_const->size<loco::DataType::FLOAT32>() != 1)
    return false;

  if (mul->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  luci::CircleNode *mul_input = nullptr;
  luci::CircleConst *mul_const = nullptr;
  if (not luci::fill(&mul_input, &mul_const).with_commutative_args_of(mul))
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
  auto new_div = div->graph()->nodes()->create<luci::CircleDiv>();
  new_div->fusedActivationFunction(luci::FusedActFunc::NONE);
  new_div->x(new_div_const);
  new_div->y(mul_input);
  new_div->name(div->name());
  luci::add_origin(new_div, luci::composite_origin({luci::get_origin(div), luci::get_origin(mul)}));

  replace(div).with(new_div);

  return true;
}

/**
 * Pass to fuse mul(one of the input is const scalar) and
 * div(numerator is const scalar) as mul
 *
 * BEFORE
 *             [CircleNode]                                [Scalar_Mul_Const]
 *                  |                                               |
 *          [CirlceMul, (x=CircleNode, y=Scalar_Mul_Const)] --------
 *                  |
 *                  |                                     [Scalar_Div_Const]
 *                  |                                             |
 *           [CircleDiv, (x=CirlceMul, y=Scalar_Div_Const)] ------
 *                  |
 *             [CircleNode]
 *
 *  AFTER
 *            [CircleNode]
 *                 |                                          [Scalar_new_Mul_Const]
 *                 |                                                   |
 *          [CircleMul, (x=CircleNode, y=Scalar_new_Mul_Const)] -------
 *                 |
 *            [CircleNode]
 *
 *          where Scalar_new_Mul_Const = Scalar_Mul_Const / Scalar_Div_Const
 *
 **/
bool fuse_mul_with_div_to_mul(luci::CircleDiv *div)
{
  if (div->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  luci::CircleMul *mul = nullptr;
  luci::CircleConst *div_const = nullptr;
  if (not luci::fill(&mul, &div_const).with_args_of(div))
    return false;

  if (mul->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  if (div_const->dtype() != loco::DataType::FLOAT32)
    return false;
  // TODO support other shape
  if (div_const->size<loco::DataType::FLOAT32>() != 1)
    return false;

  luci::CircleNode *mul_input = nullptr;
  luci::CircleConst *mul_const = nullptr;
  if (not luci::fill(&mul_input, &mul_const).with_commutative_args_of(mul))
    return false;

  if (mul_const->dtype() != loco::DataType::FLOAT32)
    return false;
  // TODO support other shape
  if (mul_const->size<loco::DataType::FLOAT32>() != 1)
    return false;

  const auto mul_value = mul_const->at<loco::DataType::FLOAT32>(0);
  const auto div_value = div_const->at<loco::DataType::FLOAT32>(0);
  const auto new_value = mul_value / div_value;
  auto new_mul_const = create_mul_const_with_new_value(mul_const, div_const, new_value);

  auto new_mul = div->graph()->nodes()->create<luci::CircleMul>();
  new_mul->fusedActivationFunction(luci::FusedActFunc::NONE);
  new_mul->x(mul_input);
  new_mul->y(new_mul_const);
  new_mul->name(mul->name());
  luci::add_origin(new_mul, luci::composite_origin({luci::get_origin(div), luci::get_origin(mul)}));

  replace(div).with(new_mul);

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

    if (fuse_mul_with_div_to_div(div))
      changed = true;

    if (fuse_mul_with_div_to_mul(div))
      changed = true;
  }

  return changed;
}

} // namespace luci
