/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveUnnecessaryMulDivPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <limits> // std::numeric_limits

namespace
{

template <class MULDIV> bool remove_no_effect_muldiv(MULDIV *target_node)
{
  if (target_node == nullptr)
    return false;
  if (target_node->dtype() != loco::DataType::FLOAT32)
    return false;

  //  NOTE for general activation function A: Act(A / 1) != A
  if (target_node->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  luci::CircleConst *const_operand = nullptr;
  luci::CircleNode *nonconst_operand = nullptr;

  if (target_node->opcode() == luci::CircleOpcode::MUL)
  {
    if (not luci::fill(&const_operand, &nonconst_operand).with_commutative_args_of(target_node))
      return false;
  }
  else if (target_node->opcode() == luci::CircleOpcode::DIV)
  {
    nonconst_operand = loco::must_cast<luci::CircleNode *>(target_node->x());
    const_operand = dynamic_cast<luci::CircleConst *>(target_node->y());
    if (const_operand == nullptr)
      return false;
  }

  // check const_operand is one
  // NOTE we assume graph is valid, so no need to check shape.
  // just check that const operand is one
  auto const size = const_operand->size<loco::DataType::FLOAT32>();
  for (uint32_t index = 0; index < size; index++)
  {
    auto const value = const_operand->at<loco::DataType::FLOAT32>(index) - 1.0f;
    if (std::abs(value) > std::numeric_limits<float>::min())
    {
      // at least one value is not 1.0
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
 *      [CircleNode]
 *            |
 *            |      [CircleConst(=1.0)]
 *            |     /
 *            |    /
 *  [CircleDiv/CircleMul] (no activation)
 *            |
 *      [CircleNode]
 *
 * AFTER
 *      [CircleNode]
 *            |
 *            |
 *      [CircleNode]
 *
 **/
bool RemoveUnnecessaryDivPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto div_node = dynamic_cast<luci::CircleDiv *>(node);
    if (remove_no_effect_muldiv<luci::CircleDiv>(div_node))
    {
      changed = true;
    }
  }
  return changed;
}

bool RemoveUnnecessaryMulPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mul_node = dynamic_cast<luci::CircleMul *>(node);
    if (remove_no_effect_muldiv<luci::CircleMul>(mul_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
