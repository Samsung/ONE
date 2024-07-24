/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ForwardPreShiftPass.h"
#include "Support.Cast.h"
#include "Support.Misc.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>

using namespace fme_apply;

namespace
{

class ForwardPreShift final : public luci::CircleNodeMutableVisitor<bool>
{
protected:
  bool visit(luci::CircleNode *node) { return false; }

  bool visit(luci::CircleSlice *node)
  {
    auto pre_shift = to_pre_shift(node->input());
    if (not pre_shift)
      return false;

    if (loco::succs(pre_shift).size() != 1)
      return false;

    node->input(pre_shift->inputs(0));
    loco::replace(node).with(pre_shift);
    pre_shift->inputs(0, node);

    // Shape should be copied, because
    // shape inference does not work well for Custom Op (PreShift)
    copy_shape(node, pre_shift);

    return true;
  }
};

} // namespace

namespace fme_apply
{

bool ForwardPreShiftPass::run(loco::Graph *g)
{
  bool changed = false;

  ForwardPreShift fps;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cnode = loco::must_cast<luci::CircleNode *>(node);
    if (cnode->accept(&fps))
      changed = true;
  }

  return changed;
}

} // namespace fme_apply
