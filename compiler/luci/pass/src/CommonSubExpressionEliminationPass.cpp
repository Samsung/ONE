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

#include "luci/Pass/CommonSubExpressionEliminationPass.h"
#include "helpers/ExpressionCache.h"

#include <luci/IR/CircleNodes.h>

using namespace luci::pass;

namespace
{

// Return true if node is a virtual node
// TODO Extract this helper to somewhere else
bool virtual_op(const luci::CircleOpcode opcode)
{
  switch (opcode)
  {
#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) \
  case luci::CircleOpcode::OPCODE:        \
    return false;
#define CIRCLE_VNODE(OPCODE, CIRCLE_CLASS) \
  case luci::CircleOpcode::OPCODE:         \
    return true;
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_NODE
#undef CIRCLE_VNODE
    default:
      throw std::runtime_error("Unknown opcode detected");
  }
}

} // namespace

namespace luci
{

bool CommonSubExpressionEliminationPass::run(loco::Graph *g)
{
  // Build cache
  ExpressionCache cache;

  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cnode = loco::must_cast<luci::CircleNode *>(node);

    // Skip virtual Ops
    // Why? virtual Ops do not perform actual computations
    // NOTE Fix this if the assumption is not true
    if (virtual_op(cnode->opcode()))
      continue;

    // Build expression
    auto expr = Expression::build(cnode);

    // Cache hit
    if (auto saved_node = cache.get(expr))
    {
      loco::replace(cnode).with(saved_node);
      changed = true;
    }
    // Cache miss
    else
    {
      cache.put(expr, cnode);
    }
  }

  return changed;
}

} // namespace luci
