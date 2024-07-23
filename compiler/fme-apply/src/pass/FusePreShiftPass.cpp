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

#include "FusePreShiftPass.h"
#include "Support.Cast.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>
#include <luci/Service/Nodes/CircleConst.h>

using namespace fme_apply;

namespace
{

// Fuse CircleCustom(PreShift) + Op
struct FusePreShift final : public luci::CircleNodeMutableVisitor<bool>
{
  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleInstanceNorm *node)
  {
    auto pre_shift = to_pre_shift(node->input());
    if (not pre_shift)
      return false;

    auto param = loco::must_cast<luci::CircleConst *>(pre_shift->inputs(1)); // FIX_PreScale_UNLESS
    auto channel = node->dim(node->rank() - 1).value();
    if (channel != param->size<loco::DataType::FLOAT32>())
    {
      assert(false); // FIX_PreShift_Unless
      return false;
    }

    // Output of InstanceNorm is not affected by PreShift
    node->input(pre_shift->inputs(0));

    return true;
  }
};

} // namespace

namespace fme_apply
{

bool FusePreShiftPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    FusePreShift fps;
    auto cnode = loco::must_cast<luci::CircleNode *>(node);
    if (cnode->accept(&fps))
      changed = true;
  }

  return changed;
}

} // namespace fme_apply
