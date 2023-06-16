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

#include "luci/Pass/PropagateQParamBackwardPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

using namespace luci;

namespace
{

void set_qparam(luci::CircleNode *node, float scale, int64_t zp)
{
  auto qparam = std::make_unique<luci::CircleQuantParam>();
  qparam->scale.emplace_back(scale);
  qparam->zerop.emplace_back(zp);

  node->quantparam(std::move(qparam));
}

/**
 * @brief Base Test Graph
 */
struct TestGraph
{
public:
  virtual void init(void) = 0;
};

/**
 *  Graph with two concats
 *
 *  [CircleInput]  [CircleConst]
 *         \         /
 *  [CircleConcatenation]  [CircleConst]
 *           |                |
 *          [CircleConcatenation]
 *                  |
 *            [CircleOutput]
 *
 *  BEFORE
 *  - Concat1 and Concat 2 have different qparams
 *
 *  AFTER
 *  - All Ops have the same qparam
 */
struct SubsequentConcatGraph : public TestGraph
{
public:
  void init(void) final
  {
    // graph input and output
    auto graph_input = g.inputs()->create();
    auto graph_output = g.outputs()->create();

    // input
    input = g.nodes()->create<luci::CircleInput>();
    input->index(graph_input->index());
    input->shape({1, 4, 4, 3});
    input->dtype(loco::DataType::U8);
    set_qparam(input, 1.0, 1);

    // const1
    const1 = g.nodes()->create<luci::CircleConst>();
    const1->shape({1, 4, 4, 3});
    const1->dtype(loco::DataType::FLOAT32);
    const1->size<loco::DataType::FLOAT32>(48);
    for (uint32_t i = 0; i < 48; i++)
      const1->at<loco::DataType::FLOAT32>(i) = i;

    // concat1
    concat1 = g.nodes()->create<luci::CircleConcatenation>(2);
    concat1->shape({1, 4, 4, 6});
    concat1->dtype(loco::DataType::U8);
    set_qparam(concat1, 2.0, 2);
    concat1->values(0, input);
    concat1->values(1, const1);
    concat1->fusedActivationFunction(luci::FusedActFunc::NONE);

    // const2
    const2 = g.nodes()->create<luci::CircleConst>();
    const2->shape({1, 4, 4, 3});
    const2->dtype(loco::DataType::FLOAT32);
    const2->size<loco::DataType::FLOAT32>(48);
    for (uint32_t i = 0; i < 48; i++)
      const2->at<loco::DataType::FLOAT32>(i) = i;

    // concat2
    concat2 = g.nodes()->create<luci::CircleConcatenation>(2);
    concat2->shape({1, 4, 4, 9});
    concat2->dtype(loco::DataType::U8);
    set_qparam(concat2, 3.0, 3);
    concat2->values(0, concat1);
    concat2->values(1, const2);
    concat2->fusedActivationFunction(luci::FusedActFunc::NONE);

    // output
    output = g.nodes()->create<luci::CircleOutput>();
    output->index(graph_output->index());
    output->from(concat2);
    output->shape({1, 4, 4, 9});
    output->dtype(loco::DataType::U8);
    set_qparam(output, 3.0, 3);
  }

public:
  loco::Graph g;
  CircleInput *input = nullptr;
  CircleConcatenation *concat1 = nullptr;
  CircleConcatenation *concat2 = nullptr;
  CircleConst *const1 = nullptr;
  CircleConst *const2 = nullptr;
  CircleOutput *output = nullptr;
};

/**
 *  BEFORE
 *
 *        [Input]
 *           |
 *        [Conv] (qparam 1)
 *           |
 *       [Reshape] (qparam 2)
 *           |
 *       [Output]
 *
 *  AFTER
 *
 *        [Input]
 *           |
 *        [Conv] (qparam 2)
 *           |
 *       [Reshape] (qparam 2)
 *           |
 *       [Output]
 */
class ConvReshapeGraph
{
public:
  ConvReshapeGraph()
  {
    input = g.nodes()->create<luci::CircleInput>();
    conv = g.nodes()->create<luci::CircleConv2D>();
    reshape = g.nodes()->create<luci::CircleReshape>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    set_qparam(conv, 2.0, 2);
    set_qparam(reshape, 1.0, 1);

    conv->input(input);
    reshape->tensor(conv);
    output->from(reshape);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleConv2D *conv = nullptr;
  luci::CircleReshape *reshape = nullptr;
  luci::CircleOutput *output = nullptr;
};

/**
 *  BEFORE
 *
 *        [Input]
 *           |
 *        [Conv] (qparam 1)
 *           |
 *           +---------------------+
 *           |                     |
 *       [Reshape] (qparam 2)   [Output]
 *           |
 *       [Output]
 *
 *  AFTER (qparam is not propagated as Conv has multiple users)
 *
 *        [Input]
 *           |
 *        [Conv] (qparam 1)
 *           |
 *           +---------------------+
 *           |                     |
 *       [Reshape] (qparam 2)   [Output]
 *           |
 *       [Output]
 */
class ConvReshapeMultiOutGraph
{
public:
  ConvReshapeMultiOutGraph()
  {
    input = g.nodes()->create<luci::CircleInput>();
    conv = g.nodes()->create<luci::CircleConv2D>();
    reshape = g.nodes()->create<luci::CircleReshape>();
    output1 = g.nodes()->create<luci::CircleOutput>();
    output2 = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output1 = g.outputs()->create();
    output1->index(graph_output1->index());
    auto graph_output2 = g.outputs()->create();
    output2->index(graph_output2->index());

    set_qparam(conv, 2.0, 2);
    set_qparam(reshape, 1.0, 1);

    conv->input(input);
    reshape->tensor(conv);
    output1->from(reshape);
    output2->from(conv);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleConv2D *conv = nullptr;
  luci::CircleReshape *reshape = nullptr;
  luci::CircleOutput *output1 = nullptr;
  luci::CircleOutput *output2 = nullptr;
};

} // namespace

TEST(PropagateQParamBackwardPassTest, name)
{
  luci::PropagateQParamBackwardPass pass(loco::DataType::U8);
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(PropagateQParamBackwardPassTest, subsequent_propagation)
{
  SubsequentConcatGraph graph;

  graph.init();

  luci::PropagateQParamBackwardPass pass(loco::DataType::U8);

  pass.run(&graph.g);

  EXPECT_EQ(3.0, graph.concat2->quantparam()->scale[0]);
  EXPECT_EQ(3, graph.concat2->quantparam()->zerop[0]);

  auto const2 = loco::must_cast<CircleNode *>(graph.concat2->values(1));
  EXPECT_EQ(3.0, const2->quantparam()->scale[0]);
  EXPECT_EQ(3, const2->quantparam()->zerop[0]);

  EXPECT_EQ(3.0, graph.concat1->quantparam()->scale[0]);
  EXPECT_EQ(3, graph.concat1->quantparam()->zerop[0]);

  auto const1 = loco::must_cast<CircleNode *>(graph.concat1->values(1));
  EXPECT_EQ(3.0, const1->quantparam()->scale[0]);
  EXPECT_EQ(3, const1->quantparam()->zerop[0]);

  EXPECT_EQ(3.0, graph.input->quantparam()->scale[0]);
  EXPECT_EQ(3, graph.input->quantparam()->zerop[0]);
}

TEST(PropagateQParamBackwardPassTest, reshape)
{
  ConvReshapeGraph graph;

  EXPECT_NE(graph.conv->quantparam()->scale, graph.reshape->quantparam()->scale);
  EXPECT_NE(graph.conv->quantparam()->zerop, graph.reshape->quantparam()->zerop);

  luci::PropagateQParamBackwardPass pass(loco::DataType::U8);

  pass.run(&graph.g);

  EXPECT_EQ(graph.conv->quantparam()->scale, graph.reshape->quantparam()->scale);
  EXPECT_EQ(graph.conv->quantparam()->zerop, graph.reshape->quantparam()->zerop);
}

TEST(PropagateQParamBackwardPassTest, reshape_multi_use_NEG)
{
  ConvReshapeMultiOutGraph graph;

  EXPECT_NE(graph.conv->quantparam()->scale, graph.reshape->quantparam()->scale);
  EXPECT_NE(graph.conv->quantparam()->zerop, graph.reshape->quantparam()->zerop);

  luci::PropagateQParamBackwardPass pass(loco::DataType::U8);

  pass.run(&graph.g);

  EXPECT_NE(graph.conv->quantparam()->scale, graph.reshape->quantparam()->scale);
  EXPECT_NE(graph.conv->quantparam()->zerop, graph.reshape->quantparam()->zerop);
}
