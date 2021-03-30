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

#include "PartitionMerge.h"

#include <algorithm>

namespace
{

/**
 * @brief return true if pgroup_i output is one of the inputs of pgroup
 */
bool is_input_of(const luci::PGroup *pgroup_i, const luci::PGroup *pgroup)
{
  for (auto *output : pgroup_i->outputs)
  {
    for (auto *input : pgroup->inputs)
    {
      if (input == output)
        return true;
    }
  }
  return false;
}

/**
 * @brief return true if there is only one input or all the inputs have same group
 * @note  pgroups is used to find group of pgroup
 */
bool is_input_same(const luci::PGroup *pgroup, const luci::PGroups *pgroups)
{
  assert(pgroups != nullptr);
  assert(pgroup != nullptr);

  const luci::PGroup *input_pgroup = nullptr;
  std::string group;
  for (auto &input : pgroup->inputs)
  {
    auto input_group = pgroups->group_of(input);
    // NOTE: all the nodes should be registered and return should be valid group.
    // convert_to_proups() should ensure this.
    // assert here to find if there is any problem with this.
    assert(not input_group.empty());
    if (input_group.empty())
      input_group = pgroups->default_group;

    if (group.empty())
      group = input_group;
    else
    {
      if (group != input_group)
        return false;
    }
    // if there are multiple inputs, all the inputs should be in same pgroup
    // https://github.com/Samsung/ONE/issues/6230#issuecomment-801618150
    // https://github.com/Samsung/ONE/issues/6230#issuecomment-801680531
    auto pgroup_input = pgroups->pgroup_of(input);
    if (pgroup_input != nullptr)
    {
      if (input_pgroup == nullptr)
        input_pgroup = pgroup_input;
      else
      {
        if (input_pgroup != pgroup_input)
          return false;
      }
    }
  }
  return true;
}

/**
 * @brief merge pgroup into pgroup_i
 * @note  output of pgroup_i should be input of pgroup
 */
void merge_into(luci::PGroup *pgroup, luci::PGroup *pgroup_i)
{
  for (auto &pnode : pgroup->pnodes)
  {
    // update pgroup for this pnode
    pnode->pgroup = pgroup_i;
    assert(pnode->group == pgroup_i->group);

    // we don't need to add this in topological order:
    // all the nodes will be created first then connection will be held
    pgroup_i->pnodes.push_back(std::move(pnode));
    // note: pnode is now nullptr as it's moved into pgroup_i->pnodes
  }

  for (auto &input : pgroup->inputs)
  {
    // add inputs of pgroup to pgroup_i if not member of pgroup_i
    bool found_in_pgroup_i = false;
    for (auto &pnode : pgroup_i->pnodes)
    {
      if (input == pnode->node)
      {
        found_in_pgroup_i = true;
        break;
      }
    }
    // skip if this input is already in the inputs
    auto fit = std::find(pgroup_i->inputs.begin(), pgroup_i->inputs.end(), input);
    if (fit != pgroup_i->inputs.end())
    {
      found_in_pgroup_i = true;
    }
    // note: if we force found_in_pgroup_i to false, for testing there will be
    // unnecessary inputs
    if (not found_in_pgroup_i)
    {
      // node input maybe in another pgroup
      pgroup_i->inputs.push_back(input);
    }
  }
  // add outputs of pgroup to pgroup_i outputs if not exist
  for (auto &output : pgroup->outputs)
  {
    auto it = std::find(pgroup_i->outputs.begin(), pgroup_i->outputs.end(), output);
    if (it == pgroup_i->outputs.end())
    {
      pgroup_i->outputs.push_back(output);
    }
  }
}

} // namespace

namespace luci
{

/**
 * @brief This will merge pgroups with same group values in topological order
 */
std::unique_ptr<luci::PGroups> merge_pgroups(const luci::PGroups *s_pgroups)
{
  // Make a copy of pgroups to apply merge action
  // Q) do we really need a copy?
  auto d_pgroups = s_pgroups->make_copy();

  // Merge partition graphs
  // - This is initial implementation that works for limited networks
  // - if A and B is same group -> if A is input of B -> ... -> merge B into A
  auto &pgroups = d_pgroups->pgroups;
  bool changed;
  do
  {
    changed = false;
    for (auto &pgroup_i : pgroups)
    {
      bool merged = false;
      for (auto it = pgroups.begin(); it != pgroups.end(); ++it)
      {
        auto &pgroup = *it;

        // skip if same object
        if (pgroup->id == pgroup_i->id)
          continue;
        // skip if different group
        if (pgroup->group != pgroup_i->group)
          continue;
        // skip if not connected
        if (!is_input_of(pgroup_i.get(), pgroup.get()))
          continue;
        // skip if there are multiple inputs but inputs differ in group
        if (!is_input_same(pgroup.get(), d_pgroups.get()))
          continue;
        // TODO add more condition may be needed

        merge_into(pgroup.get(), pgroup_i.get());

        auto eit = d_pgroups->id2pgroup.find(pgroup->id);
        assert(eit != d_pgroups->id2pgroup.end());
        d_pgroups->id2pgroup.erase(eit);

        // remove merged pgroup from pgroups
        pgroups.erase(it);

        merged = true;
        break;
      }
      if (merged)
      {
        changed = true;
        break;
      }
    }
  } while (changed);

  return std::move(d_pgroups);
}

} // namespace luci
