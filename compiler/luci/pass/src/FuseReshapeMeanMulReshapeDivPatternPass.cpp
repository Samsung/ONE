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

#include "luci/Pass/FuseReshapeMeanMulReshapeDivPatternPass.h"
#include "CircleOptimizerUtils.h"

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/IR/CircleNodes.h>

namespace luci
{

/**
 * Pass to fuse reshape mean mul reshape div pattern to mean reshape div
 *  BEFORE
 *              [Reshape]
 *                  |
 *          [CircleMean, axis<-1>]                        [Scalar_Mul_Const]
 *                  |                                             |
 *               [Mul, (x=CircleMean, y=Scalar_Mul_Const] --------
 *                  |
 *              [Reshape]                              [Scalar_Div_Const]
 *                 |                                      |
 *           [Div, (x=Reshape, y=Scalar_Div_Const] --------
 *
 *  AFTER
 *          [CircleMean, axis<-1>]
 *                  |
 *              [Reshape]                              [Scalar_new_Div_Const]
 *                 |                                         |
 *          [Div, (x=Reshape, y=Scalar_new_Div_Const] -------
 *
 *          where Scalar_new_Div_Const = Scalar_Div_Const / Scalar_Mul_Const
 *
 **/
bool FuseReshapeMeanMulReshapeDivPatternPass::run(loco::Graph *g)
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

    auto reshape_after_div = dynamic_cast<luci::CircleReshape *>(div->y());
    if (not reshape_after_div)
      continue;

    auto mul = dynamic_cast<luci::CircleMul *>(reshape_after_div->tensor());
    if (not mul)
      continue;

    luci::CircleConst *mul_const = nullptr;
    luci::CircleMean *mean = nullptr;

    mean = dynamic_cast<luci::CircleMean *>(mul->x());
    if (not mean)
    {
      mul_const = dynamic_cast<luci::CircleConst *>(mul->x());
      mean = dynamic_cast<luci::CircleMean *>(mul->y());
    }
    else
    {
      mul_const = dynamic_cast<luci::CircleConst *>(mul->y());
    }

    if (mul_const == nullptr or mean == nullptr)
      continue;

    if (mul_const->dtype() != loco::DataType::FLOAT32)
      continue;

    if (mul_const->size<loco::DataType::FLOAT32>() != 1)
      continue;

    auto reshape_after_mean = dynamic_cast<luci::CircleReshape *>(mean->input());
    if (not reshape_after_mean)
      continue;

    auto input_node = dynamic_cast<luci::CircleNode *>(reshape_after_mean->tensor());

    if (not input_node)
      continue;

    if (mean->rank() != input_node->rank())
      continue;

    auto div_value = div_const->at<loco::DataType::FLOAT32>(0);
    auto mul_value = mul_const->at<loco::DataType::FLOAT32>(0);

    auto new_value = div_value / mul_value;

    mean->input(input_node);
    reshape_after_div->tensor(mean);
    div_const->at<loco::DataType::FLOAT32>(0) = new_value;

    changed = true;
  }

  return changed;
}

} // namespace luci
