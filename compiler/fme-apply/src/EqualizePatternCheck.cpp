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

#include "EqualizePattern.h"

#include <loco.h>
#include <luci/IR/CircleNode.h>

#include <vector>
#include <map>
#include <string>

using namespace fme_apply;

namespace fme_apply
{

void check_patterns_valid(loco::Graph *g, const std::vector<EqualizePattern> &patterns)
{
  // Create a map to find node by its name
  std::map<std::string, const luci::CircleNode *> node_by_name;
  {
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      node_by_name[cnode->name()] = cnode;
    }
  }

  auto check_negative_scale_across_relu = [&node_by_name](const EqualizePattern *p) {
    auto front = node_by_name.at(p->front); // FIX_ME_UNLESS
    auto node =
      dynamic_cast<const luci::CircleNodeMixin<luci::CircleNodeTrait::FusedActFunc> *>(front);
    if (not node)
      return;

    if (node->fusedActivationFunction() != luci::FusedActFunc::RELU)
      return;

    if (p->type != EqualizePattern::Type::ScaleOnly && p->type != EqualizePattern::Type::ScaleShift)
      return;

    for (auto s : p->scale)
      if (s < 0.0)
        throw std::runtime_error("Negative scale cannot be fused across ReLU");
  };

  for (const auto &pattern : patterns)
  {
    check_negative_scale_across_relu(&pattern);
  }
}

} // namespace fme_apply
