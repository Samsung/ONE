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

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

template <loco::DataType DT>
bool is_scalar_with_value(luci::CircleConst *node, typename loco::DataTypeImpl<DT>::Type val)
{
  if (node->dtype() != DT)
    return false;
  if (node->rank() != 0)
    return false;
  if (node->size<DT>() != 1)
    return false;
  if (node->at<DT>(0) != static_cast<typename loco::DataTypeImpl<DT>::Type>(val))
    return false;

  return true;
}

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
 *
 *  NOTE Only max(min(input, 6), 0) pattern will be transformed.
 */
template <loco::DataType DT> bool transform_min_max_pattern(luci::CircleMaximum *maxi)
{
  if (not maxi)
    return false;

  if (maxi->dtype() != DT)
    return false;

  luci::CircleConst *maxi_const = nullptr;
  luci::CircleMinimum *mini = nullptr;

  // There are two ways Maximum takes inputs.
  // 1. Maximum(x = CircleConst, y = CircleMinimum)
  // 2. Maximum(x = CircleMinimum, y = CircleConst)
  if (not luci::fill(&maxi_const, &mini).with_commutative_args_of(maxi))
    return false;

  // Maximum constant should be scalar whose value is 0.
  if (not is_scalar_with_value<DT>(maxi_const,
                                   static_cast<typename loco::DataTypeImpl<DT>::Type>(0)))
    return false;

  luci::CircleConst *mini_const = nullptr;
  loco::Node *mini_input = nullptr;

  // There are two ways Miminum takes inputs.
  // 1. Miminum(x = CircleNode, y = CircleConst)
  // 2. Miminum(x = CircleConst, y = CircleNode)
  if (not luci::fill(&mini_const, &mini_input).with_commutative_args_of(mini))
    return false;

  // Miminum constant should be scalar whose value is 6.
  if (not is_scalar_with_value<DT>(mini_const,
                                   static_cast<typename loco::DataTypeImpl<DT>::Type>(6)))
    return false;

  auto name = maxi->name();
  assert(name.length() > 0);

  // Create Relu6 op
  auto relu6 = mini->graph()->nodes()->create<luci::CircleRelu6>();
  relu6->features(mini_input);
  relu6->name(name + "/Relu6");
  luci::add_origin(relu6, luci::composite_origin({luci::get_origin(maxi), luci::get_origin(mini)}));

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
      if (transform_min_max_pattern<loco::DataType::FLOAT32>(maxi))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
