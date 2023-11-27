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

#include "luci/Pass/FuseMulDivPass.h"
#include "CircleOptimizerUtils.h"

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/IR/CircleNodes.h>

namespace luci
{

/**
 * Pass to fuse mul(one of the input is const scalar) and
 * div(one of the input is const scalar) as div
 *
 * BEFORE
 *             [CircleNode]                         [Scalar_Mul_Const]
 *                  |                                      |
 * [CirlceMul, (x=CircleNode, y=Scalar_Mul_Const)] --------
 *                  |
 *                  |                             [Scalar_Div_Const]
 *                  |                                             |
 *           [CircleDiv, (x=CirlceMul, y=Scalar_Div_Const] --------
 *                 |
 *             [CircleNode]
 *
 *  AFTER
 *          [CircleNode]
 *                 |                                        [Scalar_new_Div_Const]
 *                 |                                         |
 *          [Div, (x=CircleNode, y=Scalar_new_Div_Const] -------
 *
 *          where Scalar_new_Div_Const = Scalar_Div_Const / Scalar_Mul_Const
 *
 **/
bool FuseMulDivPass::run(loco::Graph *g)
{
  bool changed = false;

  auto nodes = loco::active_nodes(loco::output_nodes(g));
  for (auto node : nodes)
  {
    auto div = dynamic_cast<luci::CircleDiv *>(node);
    if (not div)
      continue;

    auto div_const = dynamic_cast<luci::CircleConst *>(div->x());
    if (not div_const)
      continue;

    if (div_const->dtype() != loco::DataType::FLOAT32)
      continue;

    if (div_const->size<loco::DataType::FLOAT32>() != 1)
      continue;

    auto mul = dynamic_cast<luci::CircleMul *>(div->y());
    if (not mul)
      continue;

    auto mul_const = dynamic_cast<luci::CircleConst *>(mul->y());
    if (not mul_const)
      continue;

    if (mul_const->dtype() != loco::DataType::FLOAT32)
      continue;

    if (mul_const->size<loco::DataType::FLOAT32>() != 1)
      continue;

    const auto div_value = div_const->at<loco::DataType::FLOAT32>(0);
    const auto mul_value = mul_const->at<loco::DataType::FLOAT32>(0);

    const auto new_value = div_value / mul_value;

    div_const->at<loco::DataType::FLOAT32>(0) = new_value;

    div->y(mul->x());

    changed = true;
  }

  return changed;
}

} // namespace luci
