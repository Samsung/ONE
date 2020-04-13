/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/GraphPatternMatcher.h"

#include <algorithm>
#include <vector>

namespace mir
{

std::vector<std::pair<Operation *, Operation *>>
GraphPatternMatcher::matchEdge(GraphPatternMatcher::Predicate p1, GraphPatternMatcher::Predicate p2)
{

  std::vector<std::pair<Operation *, Operation *>> matches;
  for (auto *start : _g->getNodes())
  {
    if (p1(start))
    {
      for (auto &out : start->getOutputs())
      {
        for (auto use : out.getUses())
        {
          Operation *end = use.getNode();
          if (p2(end))
          {
            matches.emplace_back(std::make_pair(start, end));
            break;
          }
        }
      }
    }
  }
  return matches;
}

std::vector<std::pair<std::vector<Operation *>, Operation *>>
GraphPatternMatcher::matchUpBush(mir::GraphPatternMatcher::Predicate p1,
                                 mir::GraphPatternMatcher::Predicate p2)
{
  std::vector<std::pair<std::vector<Operation *>, Operation *>> matches;
  for (auto *root : _g->getNodes())
  {
    if (p2(root))
    {
      const auto &inputs = root->getInputs();
      if (std::all_of(inputs.begin(), inputs.end(),
                      [p1](const Operation::Output *input) { return p1(input->getNode()); }))
      {
        std::vector<Operation *> tops;
        tops.reserve(inputs.size());
        for (Operation::Output *pr : inputs)
        {
          tops.emplace_back(pr->getNode());
        }
        matches.emplace_back(std::make_pair(tops, root));
      }
    }
  }
  return matches;
}
} // namespace mir
