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
#include "luci/IR/CircleNodes.h"
#include "luci/IR/Module.h"

#include <gtest/gtest.h>

namespace
{

class EliminateDeadSubgraphPassTest : public ::testing::Test
{
public:
  EliminateDeadSubgraphPassTest()
  {
    auto main_g = loco::make_graph();
    _main_graph = main_g.get();
    _module.add(std::move(main_g));

    auto graph_1 = loco::make_graph();
    _graph_1 = graph_1.get();
    _module.add(std::move(graph_1));

    auto graph_2 = loco::make_graph();
    _graph_2 = graph_2.get();
    _module.add(std::move(graph_2));

    // This graph is unreachable
    auto graph_3 = loco::make_graph();
    _graph_3 = graph_3.get();
    _module.add(std::move(graph_3));

    // For main graph
    {
      auto input_main_node = _main_graph->nodes()->create<luci::CircleInput>();
      auto if_node = _main_graph->nodes()->create<luci::CircleIf>(1, 1);
      if_node->input(0, input_main_node);
      if_node->then_branch(1);
      if_node->else_branch(2);
      auto output_main_node = _main_graph->nodes()->create<luci::CircleOutput>();
      output_main_node->from(if_node);

      auto graph_input = _main_graph->inputs()->create();
      input_main_node->index(graph_input->index());

      auto graph_output = _main_graph->outputs()->create();
      output_main_node->index(graph_output->index());
    }

    // For first graph
    {
      auto input_main_node = _graph_1->nodes()->create<luci::CircleInput>();
      auto output_main_node = _graph_1->nodes()->create<luci::CircleOutput>();
      output_main_node->from(input_main_node);

      auto graph_input = _graph_1->inputs()->create();
      input_main_node->index(graph_input->index());

      auto graph_output = _graph_1->outputs()->create();
      output_main_node->index(graph_output->index());
    }

    // For second graph
    {
      auto input_main_node = _graph_2->nodes()->create<luci::CircleInput>();
      auto output_main_node = _graph_2->nodes()->create<luci::CircleOutput>();
      output_main_node->from(input_main_node);

      auto graph_input = _graph_2->inputs()->create();
      input_main_node->index(graph_input->index());

      auto graph_output = _graph_2->outputs()->create();
      output_main_node->index(graph_output->index());
    }

    // For third (dead) graph
    {
      auto input_main_node = _graph_3->nodes()->create<luci::CircleInput>();
      auto output_main_node = _graph_3->nodes()->create<luci::CircleOutput>();
      output_main_node->from(input_main_node);

      auto graph_input = _graph_3->inputs()->create();
      input_main_node->index(graph_input->index());

      auto graph_output = _graph_3->outputs()->create();
      output_main_node->index(graph_output->index());
    }
  }

protected:
  luci::Module _module;
  loco::Graph *_main_graph = nullptr;
  loco::Graph *_graph_1 = nullptr;
  loco::Graph *_graph_2 = nullptr;
  loco::Graph *_graph_3 = nullptr;
};

} // namespace

TEST_F(EliminateDeadSubgraphPassTest, remove_dead_subgraph)
{
  luci::EliminateDeadSubgraphPass pass;

  // Before removing dead nodes it is has 4 graphs
  ASSERT_EQ(_module.size(), 4);

  ASSERT_TRUE(pass.run(&_module));

  // After remove one dead graph  - result is 3
  ASSERT_EQ(_module.size(), 3);
}

TEST_F(EliminateDeadSubgraphPassTest, no_graphs_NEG)
{
  luci::EliminateDeadSubgraphPass pass;
  auto m = luci::make_module();
  ASSERT_FALSE(pass.run(m.get()));
}
