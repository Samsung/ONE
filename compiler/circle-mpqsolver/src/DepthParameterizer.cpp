/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "DepthParameterizer.h"

using namespace mpqsolver;

int DepthParameterizer::compute_depth(const luci::Module *module, NodeDepthType &nodes_depth,
                                      float &min_depth, float &max_depth)
{
  if (module == nullptr)
    return EXIT_FAILURE;

  if (module->size() != 1)
    return EXIT_FAILURE;

  auto graph = module->graph(0);
  if (!graph)
    return EXIT_FAILURE;

  auto nodes = graph->nodes();
  uint32_t nodes_size = nodes->size();
  std::set<std::string> input_names;
  {
    auto inp_nodes = graph->inputs();
    for (uint32_t i = 0; i < inp_nodes->size(); ++i)
    {
      auto inp_node = inp_nodes->at(i);
      auto inp_name = inp_node->name();
      input_names.insert(inp_name);
    }
  }

  // initializing
  std::vector<luci::CircleNode *> to_process;
  std::map<std::string, float> named_depth;
  {
    auto inputs = loco::input_nodes(graph);
    for (auto &node : inputs)
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      to_process.emplace_back(cnode);
      nodes_depth[cnode] = 0.f;
      named_depth[cnode->name()] = 0.f;
    }
  }

  // enumerating
  while (!to_process.empty())
  {
    auto cur_node = to_process.back();
    to_process.pop_back();
    auto iter = nodes_depth.find(cur_node);
    if (iter == nodes_depth.end())
    {
      return EXIT_FAILURE; // unexpected
    }
    float cur_depth = iter->second + 1;
    // processing children
    auto children = loco::succs(cur_node);
    for (auto &child : children)
    {
      auto cichild = loco::must_cast<luci::CircleNode *>(child);
      auto node_depth = nodes_depth.find(cichild);
      if (node_depth == nodes_depth.end() || node_depth->second < cur_depth)
      {
        // initialize depth
        nodes_depth[cichild] = cur_depth;
        to_process.push_back(cichild);
        named_depth[cichild->name()] = cur_depth;
      }
    }
  }

  auto minmax = std::minmax_element(
    nodes_depth.begin(), nodes_depth.end(),
    [=](const std::pair<luci::CircleNode *, float> &el1,
        const std::pair<luci::CircleNode *, float> &el2) { return el1.second < el2.second; });

  min_depth = minmax.first->second;
  max_depth = minmax.second->second;

  return EXIT_SUCCESS;
}
