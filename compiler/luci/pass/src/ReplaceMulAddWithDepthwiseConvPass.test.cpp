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

#include "luci/Pass/ReplaceMulAddWithDepthwiseConvPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *             [Node] [gamma]
 *                |  /
 *              [Mul]  [beta]
 *                |   /
 *               [Add]
 *
 *  AFTER
 *
 *              [Node]  [weights]  [bias]
 *                  \      /       /
 *                [DepthwiseConv2D]
 */
class SimpleGraph
{
public:
  SimpleGraph()
  {
    input = g.nodes()->create<luci::CircleInput>();
    mul = g.nodes()->create<luci::CircleMul>();
    gamma = g.nodes()->create<luci::CircleConst>();
    add = g.nodes()->create<luci::CircleAdd>();
    beta = g.nodes()->create<luci::CircleConst>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    input->dtype(loco::DataType::FLOAT32);
    mul->dtype(loco::DataType::FLOAT32);
    gamma->dtype(loco::DataType::FLOAT32);
    add->dtype(loco::DataType::FLOAT32);
    beta->dtype(loco::DataType::FLOAT32);
    output->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    input->shape({1, 4, 4, channel_size});
    mul->shape({1, 4, 4, channel_size});
    gamma->shape({channel_size});
    add->shape({1, 4, 4, channel_size});
    beta->shape({channel_size});
    output->shape({1, 4, 4, channel_size});

    gamma->size<loco::DataType::FLOAT32>(channel_size);
    beta->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      gamma->at<loco::DataType::FLOAT32>(i) = i;
      beta->at<loco::DataType::FLOAT32>(i) = i;
    }

    mul->x(input);
    mul->y(gamma);
    add->x(mul);
    add->y(beta);
    output->from(add);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *gamma = nullptr;
  luci::CircleAdd *add = nullptr;
  luci::CircleConst *beta = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

TEST(ReplaceMulAddWithDepthwiseConv, name)
{
  luci::ReplaceMulAddWithDepthwiseConvPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(ReplaceMulAddWithDepthwiseConv, simple)
{
  SimpleGraph g;

  luci::ReplaceMulAddWithDepthwiseConvPass pass;
  while (pass.run(&g.g))
    ;

  auto dwconv = dynamic_cast<luci::CircleDepthwiseConv2D *>(g.output->from());
  EXPECT_NE(nullptr, dwconv);

  uint32_t channel_size = 16;
  auto weights = dynamic_cast<luci::CircleConst *>(dwconv->filter());
  auto bias = dynamic_cast<luci::CircleConst *>(dwconv->bias());
  EXPECT_NE(nullptr, weights);
  EXPECT_EQ(4, weights->rank());
  EXPECT_EQ(channel_size, weights->dim(3).value());
  EXPECT_NE(nullptr, bias);
  EXPECT_EQ(1, bias->rank());
  EXPECT_EQ(channel_size, bias->dim(0).value());

  for (int i = 0; i < channel_size; i++)
  {
    EXPECT_FLOAT_EQ(i, weights->at<loco::DataType::FLOAT32>(i));
    EXPECT_FLOAT_EQ(i, bias->at<loco::DataType::FLOAT32>(i));
  }
}

TEST(ReplaceMulAddWithDepthwiseConv, wrong_op_NEG)
{
  SimpleGraph g;
  // swap mul/add (changed to add->mul)
  g.add->x(g.input);
  loco::replace(g.add).with(g.mul);
  g.mul->x(g.add);

  luci::ReplaceMulAddWithDepthwiseConvPass pass;
  auto changed = pass.run(&g.g);

  EXPECT_EQ(false, changed);
}
