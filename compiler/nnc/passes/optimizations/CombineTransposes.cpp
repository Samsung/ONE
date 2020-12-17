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

#include "passes/optimizations/CombineTransposes.h"
#include "mir/ops/TransposeOp.h"
#include "mir/Graph.h"
#include "mir/GraphPatternMatcher.h"
#include <algorithm>

namespace nnc
{

using namespace mir;

std::vector<size_t> combineAxisOrders(const std::vector<std::size_t> &order1,
                                      const std::vector<std::size_t> &order2)
{
  assert(order1.size() == order2.size());
  std::vector<size_t> res(order1.size());
  for (size_t i = 0; i < order1.size(); i++)
  {
    res[order2[order1[i]]] = i;
  }
  return res;
}

static bool isIdentityTranspose(const std::vector<size_t> &axis_order)
{
  for (size_t i = 0; i < (axis_order.size()); i++)
  {
    if (axis_order[i] != i)
    {
      return false;
    }
  }
  return true;
}

nnc::PassData nnc::CombineTransposes::run(nnc::PassData data)
{
  auto g = static_cast<Graph *>(data);
  assert(g);
  GraphPatternMatcher matcher(g);
  auto is_tr = [](const Operation *op1) { return op1->getType() == Operation::Type::transpose; };
  std::vector<std::pair<Operation *, Operation *>> matches = matcher.matchEdge(is_tr, is_tr);
  std::unordered_set<Operation *> deleted_nodes;
  while (!matches.empty())
  {
    for (std::pair<Operation *, Operation *> match : matches)
    {
      if (deleted_nodes.find(match.first) != deleted_nodes.end())
      {
        break;
      };
      auto *top_transpose = dynamic_cast<mir::ops::TransposeOp *>(match.first);
      if (deleted_nodes.find(match.second) != deleted_nodes.end())
      {
        break;
      };
      auto *bottom_transpose = dynamic_cast<mir::ops::TransposeOp *>(match.second);
      auto combined_axis_order =
        combineAxisOrders(top_transpose->getAxisOrder(), bottom_transpose->getAxisOrder());

      if (!isIdentityTranspose(combined_axis_order))
      {
        auto new_tr_op =
          g->create<mir::ops::TransposeOp>(top_transpose->getInput(0), combined_axis_order);

        g->replaceNode(bottom_transpose, new_tr_op);
      }
      else
      {
        // Connect top input to all outputs of bottom
        Operation *top = top_transpose->getInput(0)->getNode();
        g->replaceNode(bottom_transpose, top);
      }
      deleted_nodes.emplace(bottom_transpose);
      if (top_transpose->getOutput(0)->getUses().empty())
      {
        g->removeNode(top_transpose);
        deleted_nodes.emplace(top_transpose);
      }
    }
    matches = matcher.matchEdge(is_tr, is_tr);
  };
  return g;
}

} // namespace nnc
