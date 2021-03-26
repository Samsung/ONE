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

std::unique_ptr<PGroups> PGroups::make_copy(void) const
{
  auto d_pgroups = std::make_unique<luci::PGroups>();

  for (auto &s_pgroup : pgroups)
  {
    // make a copy of s_pgroup to d_pgroup
    std::unique_ptr<luci::PGroup> d_pgroup = std::make_unique<luci::PGroup>();

    d_pgroup->group = s_pgroup->group;
    d_pgroup->id = s_pgroup->id;

    for (auto &pnode : s_pgroup->pnodes)
    {
      auto pnodec = std::make_unique<luci::PNode>();
      pnodec->node = pnode->node;
      pnodec->group = pnode->group;
      pnodec->pgroup = d_pgroup.get();
      d_pgroup->pnodes.push_back(std::move(pnodec));
    }

    for (auto &input : s_pgroup->inputs)
      d_pgroup->inputs.push_back(input);

    for (auto &output : s_pgroup->outputs)
      d_pgroup->outputs.push_back(output);

    // copy node2group
    for (auto it = node2group.begin(); it != node2group.end(); ++it)
      d_pgroups->node2group[it->first] = it->second;

    // build id2pgroup
    d_pgroups->id2pgroup[d_pgroup->id] = d_pgroup.get();

    d_pgroups->pgroups.push_back(std::move(d_pgroup));
    // note: d_pgroup is now nullptr as it's moved
  }

  return std::move(d_pgroups);
}

std::string PGroups::group_of(luci::CircleNode *node) const
{
  assert(node != nullptr);

  LOGGER(l);

  auto it = node2group.find(node);
  if (it == node2group.end())
  {
    INFO(l) << "PGroups::group_of " << node << "(" << node->name() << ") not found" << std::endl;
    return "";
  }
  return it->second;
}

const PGroup *PGroups::pgroup_of(luci::CircleNode *node) const
{
  assert(node != nullptr);

  for (auto &pgroup : pgroups)
  {
    for (auto &pnode : pgroup->pnodes)
    {
      if (node == pnode->node)
        return pgroup.get();
    }
  }
  // node maybe graph input (CircleInput)
  return nullptr;
}

} // namespace luci
