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

#include "PartitionCleanup.h"

#include "luci/Log.h"

namespace
{

using CircleNodes = std::vector<luci::CircleNode *>;

/**
 * @note Original source outputs should be outputs
 */
void gather_graph_outputs(CircleNodes &nodes, const luci::Module *source)
{
  // graph outputs are treated as used
  auto graph = source->graph();
  for (uint32_t n = 0; n < graph->outputs()->size(); ++n)
  {
    auto output = luci::output_node(graph, n); // output is CircleOutput
    assert(output != nullptr);

    auto node = loco::must_cast<luci::CircleNode *>(output->from());

    nodes.push_back(node);
  }

  // TODO add unused virtual outputs
}

/**
 * @note If one PGroup requires an input, that input should be an output
 *        from another PGroup
 */
void gather_pgroups_outputs(CircleNodes &nodes, const luci::PGroups *pgroups)
{
  // input of a pgroup is used output
  for (auto &pgroup : pgroups->pgroups)
  {
    for (auto input : pgroup->inputs)
    {
      nodes.push_back(input);
    }
  }
}

} // namespace

namespace luci
{

void remove_unused_inputoutputs(luci::PGroups *pgroups, const luci::Module *source)
{
  assert(source != nullptr);
  assert(pgroups != nullptr);

  LOGGER(l);

  INFO(l) << "--- Cleanup unused inputs/outputs";

  // remove input within same pgroup
  for (auto &pgroup : pgroups->pgroups)
  {
    bool changed;
    do
    {
      changed = false;
      for (auto it = pgroup->inputs.begin(); it != pgroup->inputs.end(); ++it)
      {
        auto input = *it;
        if (pgroups->pgroup_of(input) == pgroup.get())
        {
          INFO(l) << "  Cleanup input " << input->name() << " from group " << pgroup->group;
          pgroup->inputs.erase(it);
          changed = true;
          break;
        }
        // NOTE CircleConst is one of input type, as they are registered as
        //      input to some node and then (should be) merged.
        //      Remove if this input is CircleConst
        if (dynamic_cast<CircleConst *>(input) != nullptr)
        {
          INFO(l) << "  Cleanup CircleConst " << input->name() << " from group " << pgroup->group;
          pgroup->inputs.erase(it);
          changed = true;
          break;
        }
      }
    } while (changed);
  }

  // remove unused output(s)
  // 'used_outputs' will hold actual used outputs for all PGroups
  CircleNodes used_outputs;

  gather_graph_outputs(used_outputs, source);
  gather_pgroups_outputs(used_outputs, pgroups);

  for (auto &pgroup : pgroups->pgroups)
  {
    bool changed;
    do
    {
      changed = false;
      for (auto it = pgroup->outputs.begin(); it != pgroup->outputs.end(); ++it)
      {
        auto output = *it;
        auto oit = std::find(used_outputs.begin(), used_outputs.end(), output);
        if (oit == used_outputs.end())
        {
          INFO(l) << "  Cleanup output " << output->name() << " from group " << pgroup->group;
          pgroup->outputs.erase(it);
          changed = true;
          break;
        }
      }
    } while (changed);
  }
}

} // namespace luci
