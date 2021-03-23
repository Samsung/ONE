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

#include "PartitionIR.h"
#include "CircleOpCode.h"

#include "luci/Log.h"

#include <cassert>
#include <ostream>
#include <iostream>

namespace luci
{

std::unique_ptr<PGraphs> PGraphs::make_copy(void) const
{
  auto d_pgraphs = std::make_unique<luci::PGraphs>();

  for (auto &s_pgraph : pgraphs)
  {
    // make a copy of s_pgraph to d_pgraph
    std::unique_ptr<luci::PGraph> d_pgraph = std::make_unique<luci::PGraph>();

    d_pgraph->group = s_pgraph->group;
    d_pgraph->id = s_pgraph->id;

    for (auto &pnode : s_pgraph->pnodes)
    {
      auto pnodec = std::make_unique<luci::PNode>();
      pnodec->node = pnode->node;
      pnodec->group = pnode->group;
      pnodec->pgraph = d_pgraph.get();
      d_pgraph->pnodes.push_back(std::move(pnodec));
    }

    for (auto &input : s_pgraph->inputs)
      d_pgraph->inputs.push_back(input);

    for (auto &output : s_pgraph->outputs)
      d_pgraph->outputs.push_back(output);

    // copy node2group
    for (auto it = node2group.begin(); it != node2group.end(); ++it)
      d_pgraphs->node2group[it->first] = it->second;

    // build id2pgraph
    d_pgraphs->id2pgraph[d_pgraph->id] = d_pgraph.get();

    d_pgraphs->pgraphs.push_back(std::move(d_pgraph));
    // note: d_pgraph is now nullptr as it's moved
  }

  return std::move(d_pgraphs);
}

std::string PGraphs::group_of(luci::CircleNode *node) const
{
  assert(node != nullptr);

  LOGGER(l);

  auto it = node2group.find(node);
  if (it == node2group.end())
    INFO(l) << "PGraphs::group_of " << node << "(" << node->name() << ") not found" << std::endl;
  assert(it != node2group.end());
  auto group = it->second;
  assert(!group.empty());
  return group;
}

const PGraph *PGraphs::pgraph_of(luci::CircleNode *node) const
{
  assert(node != nullptr);

  for (auto &pgraph : pgraphs)
  {
    for (auto &pnode : pgraph->pnodes)
    {
      if (node == pnode->node)
        return pgraph.get();
    }
  }
  // node maybe graph input (CircleInput)
  return nullptr;
}

} // namespace luci
