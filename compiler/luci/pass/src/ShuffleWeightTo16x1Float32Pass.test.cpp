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

#include "luci/Pass/ShuffleWeightTo16x1Float32Pass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

void create_fc_net(loco::Graph *g)
{
  assert(g);

  const uint32_t ROW = 16;
  const uint32_t COL = 2;
  const uint32_t elements_num = ROW * COL;

  // input
  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());
  input->name("input");

  // fc weights
  auto weights = g->nodes()->create<luci::CircleConst>();
  weights->dtype(loco::DataType::FLOAT32);
  weights->size<loco::DataType::FLOAT32>(elements_num);
  weights->rank(2);
  weights->dim(0).set(ROW);
  weights->dim(1).set(COL);
  for (uint32_t idx = 0; idx < elements_num; idx++)
  {
    weights->at<loco::DataType::FLOAT32>(idx) = idx;
  }
  weights->name("weights");

  // fc
  auto fc = g->nodes()->create<luci::CircleFullyConnected>();
  fc->dtype(loco::DataType::FLOAT32);
  fc->input(input);
  fc->weights(weights);
  fc->name("fc");

  // output
  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(fc);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
  output->name("output");
}

TEST(ShuffleWeightTo16x1Float32PassTest, name)
{
  luci::ShuffleWeightTo16x1Float32Pass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(ShuffleWeightTo16x1Float32PassTest, SimpleTest1)
{
  auto graph = loco::make_graph();
  create_fc_net(graph.get());

  luci::CircleFullyConnected *fc_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;

    fc_node = fc;
    break;
  }
  ASSERT_NE(fc_node, nullptr);
  auto weights = loco::must_cast<luci::CircleConst *>(fc_node->weights());
  // before
  ASSERT_EQ(0, weights->at<loco::DataType::FLOAT32>(0));
  ASSERT_EQ(1, weights->at<loco::DataType::FLOAT32>(1));
  ASSERT_EQ(2, weights->at<loco::DataType::FLOAT32>(2));
  ASSERT_EQ(3, weights->at<loco::DataType::FLOAT32>(3));
  ASSERT_EQ(4, weights->at<loco::DataType::FLOAT32>(4));
  ASSERT_EQ(5, weights->at<loco::DataType::FLOAT32>(5));
  ASSERT_EQ(6, weights->at<loco::DataType::FLOAT32>(6));
  ASSERT_EQ(7, weights->at<loco::DataType::FLOAT32>(7));
  ASSERT_EQ(8, weights->at<loco::DataType::FLOAT32>(8));
  ASSERT_EQ(9, weights->at<loco::DataType::FLOAT32>(9));
  ASSERT_EQ(10, weights->at<loco::DataType::FLOAT32>(10));
  ASSERT_EQ(11, weights->at<loco::DataType::FLOAT32>(11));
  ASSERT_EQ(12, weights->at<loco::DataType::FLOAT32>(12));
  ASSERT_EQ(13, weights->at<loco::DataType::FLOAT32>(13));
  ASSERT_EQ(14, weights->at<loco::DataType::FLOAT32>(14));
  ASSERT_EQ(15, weights->at<loco::DataType::FLOAT32>(15));

  luci::ShuffleWeightTo16x1Float32Pass pass;
  while (pass.run(graph.get()))
    ;

  weights = loco::must_cast<luci::CircleConst *>(fc_node->weights());
  // after
  ASSERT_EQ(0, weights->at<loco::DataType::FLOAT32>(0));
  ASSERT_EQ(2, weights->at<loco::DataType::FLOAT32>(1));
  ASSERT_EQ(4, weights->at<loco::DataType::FLOAT32>(2));
  ASSERT_EQ(6, weights->at<loco::DataType::FLOAT32>(3));
  ASSERT_EQ(8, weights->at<loco::DataType::FLOAT32>(4));
  ASSERT_EQ(10, weights->at<loco::DataType::FLOAT32>(5));
  ASSERT_EQ(12, weights->at<loco::DataType::FLOAT32>(6));
  ASSERT_EQ(14, weights->at<loco::DataType::FLOAT32>(7));
  ASSERT_EQ(16, weights->at<loco::DataType::FLOAT32>(8));
  ASSERT_EQ(18, weights->at<loco::DataType::FLOAT32>(9));
  ASSERT_EQ(20, weights->at<loco::DataType::FLOAT32>(10));
  ASSERT_EQ(22, weights->at<loco::DataType::FLOAT32>(11));
  ASSERT_EQ(24, weights->at<loco::DataType::FLOAT32>(12));
  ASSERT_EQ(26, weights->at<loco::DataType::FLOAT32>(13));
  ASSERT_EQ(28, weights->at<loco::DataType::FLOAT32>(14));
  ASSERT_EQ(30, weights->at<loco::DataType::FLOAT32>(15));
}
