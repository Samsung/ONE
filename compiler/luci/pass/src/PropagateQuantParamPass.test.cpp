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

#include "luci/Pass/PropagateQuantParamPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

void addQuantParam(luci::CircleNode *node, const std::vector<float> &scale,
                   const std::vector<int64_t> &zp)
{
  assert(node->quantparam() == nullptr);

  auto quantparam = std::make_unique<luci::CircleQuantParam>();
  quantparam->scale = scale;
  quantparam->zerop = zp;
  node->quantparam(std::move(quantparam));
}

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *        [Conv] (qparam 1)
 *           |
 *       [Reshape] (qparam 2)
 *
 *  AFTER
 *
 *        [Conv] (qparam 2)
 *           |
 *       [Reshape] (qparam 2)
 *
 */
class SimpleGraph
{
public:
  SimpleGraph()
  {
    input = g.nodes()->create<luci::CircleInput>();
    conv = g.nodes()->create<luci::CircleConv2D>();
    reshape = g.nodes()->create<luci::CircleReshape>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    addQuantParam(conv, {0.1, 0.2, 0.3}, {0, 10, 20});
    addQuantParam(reshape, {0.2, 0.4, 0.6}, {-10, 0, 10});

    conv->input(input);
    reshape->tensor(conv);
    output->from(reshape);
  }

public:
  loco::Graph g;
  luci::CircleInput *input;
  luci::CircleConv2D *conv;
  luci::CircleReshape *reshape;
  luci::CircleOutput *output;
};

} // namespace

TEST(PropagateQuantParamPassTest, name)
{
  luci::PropagateQuantParamPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(PropagateQuantParam, simple)
{
  SimpleGraph g;

  luci::PropagateQuantParamPass pass;
  while (pass.run(&g.g))
    ;

  EXPECT_FLOAT_EQ(0.2, g.conv->quantparam()->scale[0]);
  EXPECT_FLOAT_EQ(0.4, g.conv->quantparam()->scale[1]);
  EXPECT_FLOAT_EQ(0.6, g.conv->quantparam()->scale[2]);
  EXPECT_EQ(-10, g.conv->quantparam()->zerop[0]);
  EXPECT_EQ(0, g.conv->quantparam()->zerop[1]);
  EXPECT_EQ(10, g.conv->quantparam()->zerop[2]);
}

TEST(PropagateQuantParam, wrong_op_NEG)
{
  SimpleGraph g;
  g.output->from(g.conv);
  g.reshape->drop();

  luci::PropagateQuantParamPass pass;
  while (pass.run(&g.g))
    ;

  EXPECT_FLOAT_EQ(0.1, g.conv->quantparam()->scale[0]);
  EXPECT_FLOAT_EQ(0.2, g.conv->quantparam()->scale[1]);
  EXPECT_FLOAT_EQ(0.3, g.conv->quantparam()->scale[2]);
  EXPECT_EQ(0, g.conv->quantparam()->zerop[0]);
  EXPECT_EQ(10, g.conv->quantparam()->zerop[1]);
  EXPECT_EQ(20, g.conv->quantparam()->zerop[2]);
}
