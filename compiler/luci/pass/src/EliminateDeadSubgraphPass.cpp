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

#include "luci/Pass/EliminateDeadSubgraphPass.h"

#include <luci/IR/CircleNodes.h>

#include <unordered_set>
#include <deque>

namespace luci
{

namespace
{

// Go through the current graph and check all other graphs reachable from it and save it.
// Note: The main idea for finding achievable graphs is that we can reach other graphs only
// from some operations (see the list below) and we check the graph indexes from these operations.
void checkGraph(loco::Graph *current_graph, std::deque<size_t> &reachable_graphs_indexes_q)
{
  assert(current_graph != nullptr);

  // 1 - Obtain all active nodes in current graph
  // 2 - Go through all active nodes and check its types
  // 3 - If it is possible to reach another graph from the current operation (see the list below),
  //     then add the graph numbers to our queue

  // 1 - Obtain all active nodes in current graph
  // Let's enumerate nodes required to compute output nodes
  auto active_nodes = loco::active_nodes(loco::output_nodes(current_graph));

  // 2 - Go through all active nodes and check its types
  // Nodes from we can obtain different subgraph:
  // While, If, ...
  // TODO: check all nodes which can be used to reach different subgraph
  for (auto &node : active_nodes)
  {
    auto *circle_node = loco::must_cast<luci::CircleNode *>(node);

    switch (circle_node->opcode())
    {
      case CircleOpcode::WHILE:
      {
        auto *while_node = loco::must_cast<luci::CircleWhile *>(circle_node);
        // Get body and cond graph indexes
        int32_t body_graph_index = while_node->body_branch();
        int32_t cond_graph_index = while_node->cond_branch();
        assert(body_graph_index >= 0);
        assert(cond_graph_index >= 0);
        // Add indexes into queue
        reachable_graphs_indexes_q.push_back(static_cast<size_t>(body_graph_index));
        reachable_graphs_indexes_q.push_back(static_cast<size_t>(cond_graph_index));
      }
      break;
      case CircleOpcode::IF:
      {
        auto *if_node = loco::must_cast<luci::CircleIf *>(circle_node);
        // Get then and else graph indexes
        int32_t else_index = if_node->else_branch();
        int32_t then_index = if_node->then_branch();
        assert(else_index >= 0);
        assert(then_index >= 0);
        // Add indexes into queue
        reachable_graphs_indexes_q.push_back(static_cast<size_t>(else_index));
        reachable_graphs_indexes_q.push_back(static_cast<size_t>(then_index));
      }
      break;
      default:
        continue;
    }
  }
}

} // namespace

/**
 * Eliminate dead subgraph.
 * Note: dead means inaccessible from the main (with index zero) graph
 **/
bool EliminateDeadSubgraphPass::run(luci::Module *m)
{
  bool changed = false;

  // Nothing check
  if (m->size() == 1 or m->size() == 0)
    return false;

  std::unordered_set<size_t> reachable_indexes;

  // Queue with reachable graphs indexes
  std::deque<size_t> reachable_graphs_indexes_q;
  // Insert main graph - with index zero
  reachable_graphs_indexes_q.push_back(0);

  while (reachable_graphs_indexes_q.empty() == false)
  {
    // Get first index from queue and remove it from queue
    auto current_graph_index = reachable_graphs_indexes_q.front();
    reachable_graphs_indexes_q.pop_front();

    // If already check this graph - continue
    if (reachable_indexes.find(current_graph_index) != reachable_indexes.end())
      continue;

    // Add current index to reachable set
    reachable_indexes.insert(current_graph_index);

    // Check current graph and add all graph indexes which can be reached from current graph
    loco::Graph *graph = m->graph(current_graph_index);
    assert(graph != nullptr);
    checkGraph(graph, reachable_graphs_indexes_q);
  }

  assert(!reachable_indexes.empty());
  // Let's remove all indexes which can not be reached from main graph
  for (size_t i = 0; i < m->size(); ++i)
  {
    if (reachable_indexes.find(i) != reachable_indexes.end())
      continue;

    m->removeGraphByIndex(i);
    changed = true;
  }

  return changed;
}

} // namespace luci
