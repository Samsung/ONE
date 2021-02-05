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
struct MinMaxGraph
{
  loco::Graph _g;
  luci::CircleInput *_input = nullptr;
  luci::CircleMinimum *_mini = nullptr;
  luci::CircleConst *_mini_const = nullptr;
  luci::CircleMaximum *_maxi = nullptr;
  luci::CircleConst *_maxi_const = nullptr;
  luci::CircleOutput *_output = nullptr;
};

class TransformMinMaxToRelu6PassTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    const int N = 1;
    const int H = 4;
    const int W = 4;
    const int C = 3;

    // graph input and output
    auto graph_input = _min_max_g._g.inputs()->create();
    auto graph_output = _min_max_g._g.outputs()->create();

    // CircleInput
    _min_max_g._input = _min_max_g._g.nodes()->create<luci::CircleInput>();
    _min_max_g._input->index(graph_input->index());
    _min_max_g._input->shape({N, H, W, C});
    _min_max_g._input->dtype(loco::DataType::FLOAT32);

    // CircleConst
    _min_max_g._mini_const = _min_max_g._g.nodes()->create<luci::CircleConst>();
    _min_max_g._mini_const->shape({}); // scalar
    _min_max_g._mini_const->dtype(loco::DataType::FLOAT32);
    _min_max_g._mini_const->size<loco::DataType::FLOAT32>(1);
    _min_max_g._mini_const->at<loco::DataType::FLOAT32>(0) = 6.;

    // CircleMinimum
    _min_max_g._mini = _min_max_g._g.nodes()->create<luci::CircleMinimum>();
    _min_max_g._mini->x(_min_max_g._input);
    _min_max_g._mini->y(_min_max_g._mini_const);
    _min_max_g._mini->shape({N, H, W, C});
    _min_max_g._mini->dtype(loco::DataType::FLOAT32);

    // CircleConst
    _min_max_g._maxi_const = _min_max_g._g.nodes()->create<luci::CircleConst>();
    _min_max_g._mini_const->shape({}); // scalar
    _min_max_g._maxi_const->dtype(loco::DataType::FLOAT32);
    _min_max_g._maxi_const->size<loco::DataType::FLOAT32>(1);
    _min_max_g._maxi_const->at<loco::DataType::FLOAT32>(0) = 0.;

    // CircleMaximum
    _min_max_g._maxi = _min_max_g._g.nodes()->create<luci::CircleMaximum>();
    _min_max_g._maxi->x(_min_max_g._mini);
    _min_max_g._maxi->y(_min_max_g._maxi_const);
    _min_max_g._maxi->shape({N, H, W, C});
    _min_max_g._maxi->dtype(loco::DataType::FLOAT32);

    // CircleOutput
    _min_max_g._output = _min_max_g._g.nodes()->create<luci::CircleOutput>();
    _min_max_g._output->index(graph_output->index());
    _min_max_g._output->from(_min_max_g._maxi);
    _min_max_g._output->shape({N, H, W, C});
    _min_max_g._output->dtype(loco::DataType::FLOAT32);
  }

protected:
  luci::TransformMinMaxToRelu6Pass _pass;
  MinMaxGraph _min_max_g;
};

} // namespace

TEST_F(TransformMinMaxToRelu6PassTest, name)
{
  auto const name = _pass.name();
  ASSERT_NE(nullptr, name);
}

/**
 *  Optimized graph looks like below.
 *
 *  [CircleInput]
 *        |
 *  [CircleRelu6]
 *        |
 *  [CircleOutput]
 */
TEST_F(TransformMinMaxToRelu6PassTest, simple_test)
{
  auto ret = _pass.run(&_min_max_g._g);
  EXPECT_TRUE(ret);

  auto relu6 = dynamic_cast<luci::CircleRelu6 *>(_min_max_g._output->from());
  EXPECT_NE(nullptr, relu6);

  auto input = dynamic_cast<luci::CircleInput *>(relu6->features());
  EXPECT_NE(nullptr, input);
}

TEST_F(TransformMinMaxToRelu6PassTest, wrong_condition_NEG)
{
  _min_max_g._maxi_const->at<loco::DataType::FLOAT32>(0) = 2.;

  auto ret = _pass.run(&_min_max_g._g);

  EXPECT_FALSE(ret);
}
