/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveUnnecessaryReshapeNetPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool acceptable_intermediate_op(const loco::Node *node)
{
  if (not node)
    return false;

  const auto opcode = loco::must_cast<const luci::CircleNode *>(node)->opcode();

  switch (opcode)
  {
    case luci::CircleOpcode::ADD:
    case luci::CircleOpcode::MUL:
    case luci::CircleOpcode::TANH:
    case luci::CircleOpcode::LOGISTIC:
    case luci::CircleOpcode::RELU:
      break;

    default:
      return false;
  }

  return true;
}

bool same_shape(const loco::Node *a, const loco::Node *b)
{
  auto a_cnode = loco::must_cast<const luci::CircleNode *>(a);
  auto b_cnode = loco::must_cast<const luci::CircleNode *>(b);

  if (a_cnode->rank() != b_cnode->rank())
    return false;

  for (uint32_t i = 0; i < a_cnode->rank(); i++)
  {
    if (not(a_cnode->dim(i) == b_cnode->dim(i)))
      return false;
  }
  return true;
}

class PreReshapeFinder
{
public:
  PreReshapeFinder(const luci::CircleReshape *post_reshape) : _post_reshape(post_reshape)
  {
    assert(post_reshape != nullptr); // FIX_CALLER_UNLESS
  }

public:
  // Return true if pre_reshapes are found
  bool collect_pre_reshapes(loco::Node *node)
  {
    // TODO Support diamond case
    if (loco::succs(node).size() != 1)
      return false;

    if (auto pre_reshape = dynamic_cast<luci::CircleReshape *>(node))
    {
      // Check ifm of pre-reshape and ofm of post_reshape
      if (not same_shape(pre_reshape->tensor(), _post_reshape))
        return false;

      // Check ofm of pre-reshape and ifm of post_reshape
      if (not same_shape(pre_reshape, _post_reshape->tensor()))
        return false;

      _pre_reshapes.emplace_back(pre_reshape);
      return true;
    }

    if (not acceptable_intermediate_op(node))
      return false;

    for (uint32_t i = 0; i < node->arity(); i++)
    {
      if (not collect_pre_reshapes(node->arg(i)))
        return false;
    }

    return true;
  }

public:
  std::vector<luci::CircleReshape *> pre_reshapes(void) const { return _pre_reshapes; }

private:
  const luci::CircleReshape *_post_reshape = nullptr;
  std::vector<luci::CircleReshape *> _pre_reshapes;
};

bool remove_unnecessary_reshape_net(luci::CircleReshape *reshape)
{
  PreReshapeFinder finder(reshape);
  if (not finder.collect_pre_reshapes(reshape->tensor()))
    return false;

  // Remove pre_reshapes
  for (auto pre_reshape : finder.pre_reshapes())
  {
    loco::replace(pre_reshape).with(pre_reshape->tensor());
  }

  // Remove post_reshape
  loco::replace(reshape).with(reshape->tensor());

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
 *    [CircleReshape_1] (shape: A -> B)
 *            |
 *      [CircleNode] (ex: Add/Mul/Tanh/Logistic ..)
 *            |
 *    [CircleReshape_2] (shape: B -> A)
 *            |
 *      [CircleNode]
 *
 * AFTER
 *
 *      [CircleNode]
 *            |   \
 *            |   [CircleReshape_1]
 *      [CircleNode]
 *            |   \
 *            |   [CircleReshape_2]
 *      [CircleNode]
 **/
bool RemoveUnnecessaryReshapeNetPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto reshape_node = dynamic_cast<luci::CircleReshape *>(node))
    {
      if (remove_unnecessary_reshape_net(reshape_node))
        changed = true;
    }
  }
  return changed;
}

} // namespace luci
