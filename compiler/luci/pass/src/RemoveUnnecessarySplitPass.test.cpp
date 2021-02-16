/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveUnnecessarySplitPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

void create_unnecessary_split_graph(loco::Graph *g, bool remove)
{
  assert(g);

  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());
  input->name("input");

  auto dim = g->nodes()->create<luci::CircleConst>();
  dim->dtype(loco::DataType::S32);
  dim->size<loco::DataType::S32>(1);
  dim->rank(1);
  dim->dim(0).set(1);
  dim->at<loco::DataType::S32>(0) = 0;
  dim->name("dim");
  auto split_node = g->nodes()->create<luci::CircleSplit>();
  split_node->split_dim(dim);
  split_node->input(input);
  if (remove)
    split_node->num_split(1);
  else
    split_node->num_split(2);
  split_node->name("split_node");

  auto split_out_node0 = g->nodes()->create<luci::CircleSplitOut>();
  split_out_node0->input(split_node);
  split_out_node0->index(0);
  split_out_node0->name("split_out_node0");

  auto output0 = g->nodes()->create<luci::CircleOutput>();
  output0->from(split_out_node0);
  auto graph_output0 = g->outputs()->create();
  output0->index(graph_output0->index());
  output0->name("output0");

  if (!remove)
  {
    auto split_out_node1 = g->nodes()->create<luci::CircleSplitOut>();
    split_out_node1->input(split_node);
    split_out_node1->index(1);
    split_out_node1->name("split_out_node1");

    auto output1 = g->nodes()->create<luci::CircleOutput>();
    output1->from(split_out_node1);
    auto graph_output1 = g->outputs()->create();
    output1->index(graph_output1->index());
    output1->name("output1");
  }
}

} // namespace

TEST(RemoveUnnecessarySplitPass, name)
{
  luci::RemoveUnnecessarySplitPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveUnnecessarySplitPass, create_unnecessary_split)
{
  auto graph = loco::make_graph();
  create_unnecessary_split_graph(graph.get(), true);

  luci::RemoveUnnecessarySplitPass pass;
  while (pass.run(graph.get()))
    ;
  luci::CircleSplit *split_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto split = dynamic_cast<luci::CircleSplit *>(node);
    if (not split)
      continue;
    split_node = split;
  }
  // No Split node is in graph.
  ASSERT_EQ(nullptr, split_node);
}

TEST(RemoveUnnecessarySplitPass, create_unnecessary_split_NEG)
{
  auto graph = loco::make_graph();
  create_unnecessary_split_graph(graph.get(), false);

  luci::RemoveUnnecessarySplitPass pass;
  while (pass.run(graph.get()))
    ;
  luci::CircleSplit *split_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto split = dynamic_cast<luci::CircleSplit *>(node);
    if (not split)
      continue;
    split_node = split;
  }
  // Split node is in graph.
  ASSERT_NE(nullptr, split_node);
}
