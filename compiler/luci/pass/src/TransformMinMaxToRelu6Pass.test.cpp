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

#include "luci/Pass/TransformMinMaxToRelu6Pass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Minimum-Maximum pattern graph
 *
 *  [CircleInput]  [CircleConst]
 *         \         /
 *    [CircleMinimum]   [CircleConst]
 *             |       /
 *       [CircleMaximum]
 *             |
 *       [CircleOutput]
 */
class MinMaxGraph
{
public:
  MinMaxGraph()
  {
    const int N = 1;
    const int H = 4;
    const int W = 4;
    const int C = 3;

    // graph input and output
    auto graph_input = g.inputs()->create();
    auto graph_output = g.outputs()->create();

    // CircleInput
    input = g.nodes()->create<luci::CircleInput>();
    input->index(graph_input->index());
    input->shape({N, H, W, C});
    input->dtype(loco::DataType::FLOAT32);

    // CircleConst
    mini_const = g.nodes()->create<luci::CircleConst>();
    mini_const->shape({}); // scalar
    mini_const->dtype(loco::DataType::FLOAT32);
    mini_const->size<loco::DataType::FLOAT32>(1);
    mini_const->at<loco::DataType::FLOAT32>(0) = 6.;

    // CircleMinimum
    mini = g.nodes()->create<luci::CircleMinimum>();
    mini->x(input);
    mini->y(mini_const);
    mini->shape({N, H, W, C});
    mini->dtype(loco::DataType::FLOAT32);

    // CircleConst
    maxi_const = g.nodes()->create<luci::CircleConst>();
    mini_const->shape({}); // scalar
    maxi_const->dtype(loco::DataType::FLOAT32);
    maxi_const->size<loco::DataType::FLOAT32>(1);
    maxi_const->at<loco::DataType::FLOAT32>(0) = 0.;

    // CircleMaximum
    maxi = g.nodes()->create<luci::CircleMaximum>();
    maxi->x(mini);
    maxi->y(maxi_const);
    maxi->shape({N, H, W, C});
    maxi->dtype(loco::DataType::FLOAT32);

    // CircleOutput
    output = g.nodes()->create<luci::CircleOutput>();
    output->index(graph_output->index());
    output->from(maxi);
    output->shape({N, H, W, C});
    output->dtype(loco::DataType::FLOAT32);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleMinimum *mini = nullptr;
  luci::CircleConst *mini_const = nullptr;
  luci::CircleMaximum *maxi = nullptr;
  luci::CircleConst *maxi_const = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

/**
 *  Optimized graph looks like below.
 *
 *  [CircleInput]
 *        |
 *  [CircleRelu6]
 *        |
 *  [CircleOutput]
 */
TEST(ReplaceMulAddWithDepthwiseConv, simple_test)
{
  MinMaxGraph g;

  luci::TransformMinMaxToRelu6Pass pass;
  auto ret = pass.run(&g.g);
  EXPECT_EQ(true, ret);

  auto relu6 = dynamic_cast<luci::CircleRelu6 *>(g.output->from());
  EXPECT_NE(nullptr, relu6);

  auto input = dynamic_cast<luci::CircleInput *>(relu6->features());
  EXPECT_NE(nullptr, input);
}

TEST(ReplaceMulAddWithDepthwiseConv, wrong_condition_NEG)
{
  MinMaxGraph g;

  g.maxi_const->at<loco::DataType::FLOAT32>(0) = 2.;

  luci::TransformMinMaxToRelu6Pass pass;
  auto ret = pass.run(&g.g);

  EXPECT_EQ(false, ret);
}
