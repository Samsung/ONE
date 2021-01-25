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

#include "InferenceCandidates.h"

#include <luci/IR/DeadNodeQueryService.h>

namespace luci
{

std::vector<loco::Node *> inference_candidates(loco::Graph *g)
{
  auto candidates = loco::postorder_traversal(loco::output_nodes(g));

  for (auto node : loco::all_nodes(g))
  {
    // already included as candidate
    if (std::find(candidates.begin(), candidates.end(), node) != candidates.end())
      continue;

    // As the node is not used for both graph output and multiple output operation,
    // it cannot be candidate.
    if (node->dialect()->service<DeadNodeQueryServiceImpl>()->isDeadNode(node))
      continue;

    candidates.emplace_back(node);
  }

  return candidates;
}

} // namespace luci
