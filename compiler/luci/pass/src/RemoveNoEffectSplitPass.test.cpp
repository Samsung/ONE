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

#include "luci/Pass/RemoveNoEffectSplitPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

void create_no_effect_split_graph(loco::Graph *g)
{
  assert(g);

  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());

  auto dim = g->nodes()->create<luci::CircleConst>();
  dim->dtype(loco::DataType::S32);
  dim->size<loco::DataType::S32>(1);
  dim->rank(1);
  dim->dim(0).set(1);
  dim->at<loco::DataType::S32>(0) = 0;

  auto split_node = g->nodes()->create<luci::CircleSplit>();
  split_node->split_dim(dim);
  split_node->input(input);
  split_node->num_split(1);

  auto split_out_node = g->nodes()->create<luci::CircleSplitOut>();
  split_out_node->input(split_node);
  split_out_node->index(0);

  // Output
  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(split_out_node);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
}

} // namespace

TEST(RemoveNoEffectSplitPass, remove_no_effect_split)
{
  auto graph = loco::make_graph();
  create_no_effect_split_graph(graph.get());

  luci::RemoveNoEffectSplitPass pass;
  while (pass.run(graph.get()))
    ;
  luci::CircleSplit *split_node = nullptr;
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto split = dynamic_cast<luci::CircleSplit *>(node);
    if (not split)
      continue;
    split_node = split;
    count++;
  }
  // No transpose node is in graph.
  ASSERT_EQ(0, count);
  ASSERT_EQ(nullptr, split_node);
}
