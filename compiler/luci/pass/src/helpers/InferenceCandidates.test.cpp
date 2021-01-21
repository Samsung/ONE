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

#include "InferenceCandidates.h"
#include "luci/IR/CircleNode.h"

#include <algorithm>

#include <gtest/gtest.h>

namespace
{

bool contains(const std::vector<loco::Node *> &vec, loco::Node *val)
{
  return std::any_of(vec.begin(), vec.end(), [val](loco::Node *node) { return node == val; });
}

} // namespace

TEST(LuciPassHelpersInferenceCandidates, inference_candidates)
{
  auto g = loco::make_graph();

  // Create nodes
  auto input = g->nodes()->create<luci::CircleInput>();
  auto split = g->nodes()->create<luci::CircleSplit>();
  auto split_out1 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_out2 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_dim = g->nodes()->create<luci::CircleConst>();
  auto output = g->nodes()->create<luci::CircleOutput>();

  // Build up initial graph
  auto graph_input1 = g->inputs()->create();
  input->index(graph_input1->index());

  split->split_dim(split_dim);
  split->input(input);
  split->num_split(2);

  split_out1->input(split);
  split_out1->index(0);

  split_out2->input(split);
  split_out2->index(1);

  auto graph_output = g->outputs()->create();
  output->from(split_out1);
  output->index(graph_output->index());

  auto s = luci::inference_candidates(g.get());

  ASSERT_EQ(6, s.size());
  ASSERT_TRUE(contains(s, input));
  ASSERT_TRUE(contains(s, split));
  ASSERT_TRUE(contains(s, split_out1));
  ASSERT_TRUE(contains(s, split_out2));
  ASSERT_TRUE(contains(s, split_dim));
  ASSERT_TRUE(contains(s, output));
}

TEST(LuciPassHelpersInferenceCandidates, inference_candidates_NEG)
{
  auto g = loco::make_graph();

  // Create nodes
  auto input = g->nodes()->create<luci::CircleInput>();
  auto split = g->nodes()->create<luci::CircleSplit>();
  auto split_out1 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_out2 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_dim = g->nodes()->create<luci::CircleConst>();
  auto relu1 = g->nodes()->create<luci::CircleRelu>();
  auto relu2 = g->nodes()->create<luci::CircleRelu>();
  auto output = g->nodes()->create<luci::CircleOutput>();

  // Build up initial graph
  auto graph_input1 = g->inputs()->create();
  input->index(graph_input1->index());

  split->split_dim(split_dim);
  split->input(input);
  split->num_split(2);

  split_out1->input(split);
  split_out1->index(0);

  split_out2->input(split);
  split_out2->index(1);

  relu1->features(split_out2);

  relu2->features(input);

  auto graph_output = g->outputs()->create();
  output->from(split_out1);
  output->index(graph_output->index());

  auto s = luci::inference_candidates(g.get());

  ASSERT_EQ(6, s.size());
  ASSERT_TRUE(contains(s, input));
  ASSERT_TRUE(contains(s, split));
  ASSERT_TRUE(contains(s, split_out1));
  ASSERT_TRUE(contains(s, split_out2));
  ASSERT_TRUE(contains(s, split_dim));
  ASSERT_TRUE(contains(s, output));
  ASSERT_FALSE(contains(s, relu1));
  ASSERT_FALSE(contains(s, relu2));
}
