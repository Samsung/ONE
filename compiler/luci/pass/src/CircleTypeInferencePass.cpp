/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/CircleTypeInferencePass.h"

#include <luci/IR/DeadNodeQueryService.h>
#include <luci/Service/CircleTypeInference.h>

#include <loco.h>

namespace luci
{

bool CircleTypeInferencePass::run(luci::Module *m)
{
  bool changed = false;

  for (size_t g = 0; g < m->size(); ++g)
  {
    if (run(m->graph(g)))
      changed = true;
  }

  return changed;
}

bool CircleTypeInferencePass::run(loco::Graph *g)
{
  luci::tinf::Rule type_infer_rule;
  bool changed = false;

  // Use all_nodes to prevent an error such as WHILE_002
  for (auto node : loco::all_nodes(g))
  {
    if (!node->dialect()->service<DeadNodeQueryServiceImpl>()->isDeadNode(node))
    {
      loco::DataType dtype;
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);

      if (type_infer_rule.infer(circle_node, dtype) && circle_node->dtype() != dtype)
      {
        circle_node->dtype(dtype);
        changed = true;
      }
    }
  }

  return changed;
}

} // namespace luci
