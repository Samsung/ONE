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
class MinMaxGraph : public ::testing::Test
{
protected:
  MinMaxGraph()
  {
    const int N = 1;
    const int H = 4;
    const int W = 4;
    const int C = 3;

    // graph input and output
    auto graph_input = _g.inputs()->create();
    auto graph_output = _g.outputs()->create();

    // CircleInput
    _input = _g.nodes()->create<luci::CircleInput>();
    _input->index(graph_input->index());
    _input->shape({N, H, W, C});
    _input->dtype(loco::DataType::FLOAT32);

    // CircleConst
    _mini_const = _g.nodes()->create<luci::CircleConst>();
    _mini_const->shape({}); // scalar
    _mini_const->dtype(loco::DataType::FLOAT32);
    _mini_const->size<loco::DataType::FLOAT32>(1);
    _mini_const->at<loco::DataType::FLOAT32>(0) = 6.;

    // CircleMinimum
    _mini = _g.nodes()->create<luci::CircleMinimum>();
    _mini->x(_input);
    _mini->y(_mini_const);
    _mini->shape({N, H, W, C});
    _mini->dtype(loco::DataType::FLOAT32);

    // CircleConst
    _maxi_const = _g.nodes()->create<luci::CircleConst>();
    _mini_const->shape({}); // scalar
    _maxi_const->dtype(loco::DataType::FLOAT32);
    _maxi_const->size<loco::DataType::FLOAT32>(1);
    _maxi_const->at<loco::DataType::FLOAT32>(0) = 0.;

    // CircleMaximum
    _maxi = _g.nodes()->create<luci::CircleMaximum>();
    _maxi->x(_mini);
    _maxi->y(_maxi_const);
    _maxi->shape({N, H, W, C});
    _maxi->dtype(loco::DataType::FLOAT32);

    // CircleOutput
    _output = _g.nodes()->create<luci::CircleOutput>();
    _output->index(graph_output->index());
    _output->from(_maxi);
    _output->shape({N, H, W, C});
    _output->dtype(loco::DataType::FLOAT32);
  }

protected:
  loco::Graph _g;
  luci::CircleInput *_input = nullptr;
  luci::CircleMinimum *_mini = nullptr;
  luci::CircleConst *_mini_const = nullptr;
  luci::CircleMaximum *_maxi = nullptr;
  luci::CircleConst *_maxi_const = nullptr;
  luci::CircleOutput *_output = nullptr;
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
TEST_F(MinMaxGraph, simple_test)
{
  luci::TransformMinMaxToRelu6Pass pass;
  auto ret = pass.run(&_g);
  EXPECT_TRUE(ret);

  auto relu6 = dynamic_cast<luci::CircleRelu6 *>(_output->from());
  EXPECT_NE(nullptr, relu6);

  auto input = dynamic_cast<luci::CircleInput *>(relu6->features());
  EXPECT_NE(nullptr, input);
}

TEST_F(MinMaxGraph, wrong_condition_NEG)
{
  _maxi_const->at<loco::DataType::FLOAT32>(0) = 2.;

  luci::TransformMinMaxToRelu6Pass pass;
  auto ret = pass.run(&_g);

  EXPECT_FALSE(ret);
}
