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

#include "luci/Pass/PropagateQParamForwardPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/test/TestIOGraph.h>

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

/**
 *  Test graph for forward propagation in Quantize Op
 *
 *  BEFORE
 *
 *         [Tanh U8] (qparam 1 - pre-defined for U8)
 *             |
 *       [Quantize S16] (qparam 2 - not pre-defined value)
 *
 *  AFTER
 *
 *         [Tanh U8] (qparam 1 - pre-defined for U8)
 *             |
 *       [Quantize S16] (qparam 3 - pre-defined for S16)
 *
 */
class TanhQuantizeGraph
{
public:
  TanhQuantizeGraph()
  {
    input = g.nodes()->create<luci::CircleInput>();
    tanh = g.nodes()->create<luci::CircleTanh>();
    quantize = g.nodes()->create<luci::CircleQuantize>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    tanh->dtype(loco::DataType::U8);
    quantize->dtype(loco::DataType::S16);

    addQuantParam(tanh, {2.0f / 256.0f}, {128}); // pre-defined qparam for U8
    addQuantParam(quantize, {1.0}, {0});         // not pre-defined values

    tanh->x(input);
    quantize->input(tanh);
    output->from(quantize);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleTanh *tanh = nullptr;
  luci::CircleQuantize *quantize = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

TEST(PropagateQParamForwardPassTest, name)
{
  luci::PropagateQParamForwardPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(PropagateQParamForward, simple)
{
  SimpleGraph g;

  luci::PropagateQParamForwardPass pass;
  while (pass.run(&g.g))
    ;

  EXPECT_FLOAT_EQ(0.1, g.reshape->quantparam()->scale[0]);
  EXPECT_FLOAT_EQ(0.2, g.reshape->quantparam()->scale[1]);
  EXPECT_FLOAT_EQ(0.3, g.reshape->quantparam()->scale[2]);
  EXPECT_EQ(0, g.reshape->quantparam()->zerop[0]);
  EXPECT_EQ(10, g.reshape->quantparam()->zerop[1]);
  EXPECT_EQ(20, g.reshape->quantparam()->zerop[2]);
}

TEST(PropagateQParamForward, wrong_op_NEG)
{
  SimpleGraph g;
  g.output->from(g.conv);
  g.reshape->drop();

  luci::PropagateQParamForwardPass pass;
  while (pass.run(&g.g))
    ;

  EXPECT_FLOAT_EQ(0.1, g.conv->quantparam()->scale[0]);
  EXPECT_FLOAT_EQ(0.2, g.conv->quantparam()->scale[1]);
  EXPECT_FLOAT_EQ(0.3, g.conv->quantparam()->scale[2]);
  EXPECT_EQ(0, g.conv->quantparam()->zerop[0]);
  EXPECT_EQ(10, g.conv->quantparam()->zerop[1]);
  EXPECT_EQ(20, g.conv->quantparam()->zerop[2]);
}

TEST(PropagateQParamForward, tanh_predefined_value)
{
  TanhQuantizeGraph g;

  luci::PropagateQParamForwardPass pass;
  while (pass.run(&g.g))
    ;

  EXPECT_FLOAT_EQ(1.0f / 32768.0f, g.quantize->quantparam()->scale[0]);
}
