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

#include "luci/Pass/TransformMinMaxToRelu6Pass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

/**
 *  BEFORE
 *        [CircleNode]
 *              |
 *       [CircleMinimum]
 *              |
 *       [CircleMaximum]
 *              |
 *        [CircleNode]
 *
 *  AFTER
 *
 *        [CircleNode]
 *              |
 *        [CircleRelu6]
 *              |
 *        [CircleNode]
 *              |
 *
 *  NOTE Only max(min(input, 6), 0) pattern will be transformed.
 */
bool transform_min_max_pattern(luci::CircleMaximum *maxi)
{
  if (not maxi)
    return false;

  luci::CircleConst *maxi_const = nullptr;
  luci::CircleMinimum *mini = nullptr;

  // There are two ways Maximum takes inputs.
  // 1. Maximum(x = CircleConst, y = CircleMinimum)
  if (auto lhs = dynamic_cast<luci::CircleConst *>(maxi->x()))
  {
    maxi_const = lhs;
    mini = dynamic_cast<luci::CircleMinimum *>(maxi->y());
  }
  // 2. Maximum(x = CircleMinimum, y = CircleConst)
  else if (auto lhs = dynamic_cast<luci::CircleMinimum *>(maxi->x()))
  {
    maxi_const = dynamic_cast<luci::CircleConst *>(maxi->y());
    mini = lhs;
  }
  else
    return false;

  if (maxi_const == nullptr || mini == nullptr)
    return false;

  // Maximum constant should be scalar whose value is 0.
  if (maxi_const->rank() != 0)
    return false;
  if (maxi_const->size<loco::DataType::FLOAT32>() != 1)
    return false;
  if (maxi_const->at<loco::DataType::FLOAT32>(0) != 0.)
    return false;

  luci::CircleConst *mini_const = nullptr;
  loco::Node *mini_input = nullptr;

  // There are two ways Miminum takes inputs.
  // 1. Miminum(x = CircleNode, y = CircleMinimum)
  if (auto lhs = dynamic_cast<luci::CircleConst *>(mini->x()))
  {
    mini_const = lhs;
    mini_input = mini->y();
  }
  // 2. Miminum(x = CircleMinimum, y = CircleNode)
  else if (auto rhs = dynamic_cast<luci::CircleConst *>(mini->y()))
  {
    mini_const = rhs;
    mini_input = mini->x();
  }
  else
    return false;

  if (mini_const == nullptr || mini_input == nullptr)
    return false;

  // Miminum constant should be scalar whose value is 6.
  if (mini_const->rank() != 0)
    return false;
  if (mini_const->size<loco::DataType::FLOAT32>() != 1)
    return false;
  if (mini_const->at<loco::DataType::FLOAT32>(0) != 6.)
    return false;

  // Create Relu6 op
  auto relu6 = mini->graph()->nodes()->create<luci::CircleRelu6>();
  relu6->features(mini_input);

  replace(maxi).with(relu6);

  return true;
}

} // namespace

namespace luci
{

bool TransformMinMaxToRelu6Pass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto maxi = dynamic_cast<luci::CircleMaximum *>(node))
    {
      if (transform_min_max_pattern(maxi))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
