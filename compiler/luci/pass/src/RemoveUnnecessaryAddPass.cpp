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

#include "luci/Pass/RemoveUnnecessaryAddPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <limits> // std::numeric_limits

namespace
{

bool remove_no_effect_add(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleAdd *>(node);
  if (target_node == nullptr || target_node->dtype() != loco::DataType::FLOAT32)
    return false;

  //  NOTE for general activation function A: Act(A + 0) != A
  if (target_node->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  luci::CircleConst *const_operand = nullptr;
  luci::CircleNode *nonconst_operand = nullptr;
  if (not luci::fill(&const_operand, &nonconst_operand).with_commutative_args_of(target_node))
    return false;

  if (dynamic_cast<luci::CircleConst *>(nonconst_operand) != nullptr)
  {
    // NOTE this is degenerated '(const1 + const2)' case
    return false;
  }

  // check const_operand is zero

  // NOTE we assume graph is valid, so no need to check shape.
  // just check that const operand is zero
  auto const size = const_operand->size<loco::DataType::FLOAT32>();
  for (uint32_t index = 0; index < size; index++)
  {
    auto const value = const_operand->at<loco::DataType::FLOAT32>(index);
    if (std::abs(value) > std::numeric_limits<float>::min())
    {
      // at least one value is not zero
      return false;
    }
  }

  replace(target_node).with(nonconst_operand);
  return true;
}

} // namespace

namespace luci
{

/**
 * BEFORE
 *
 *      [CircleNode]
 *            |
 *            |      [CircleConst(=0)]
 *            |     /
 *            |    /
 *       [CircleAdd] (no activation)
 *            |
 *      [CircleNode]
 *
 * AFTER
 *
 *      [CircleNode]
 *            |
 *            |
 *      [CircleNode]
 *
 **/
bool RemoveUnnecessaryAddPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_no_effect_add(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
