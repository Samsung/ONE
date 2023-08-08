/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/DecomposeHardSwishPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  HardSwish graph
 *
 *        [CircleInput]
 *              |
 *              |
 *      [CircleHardSwish]
 *              |
 *              |
 *        [CircleOutput]
 */
struct HardSwishGraph
{
  loco::Graph _g;
  luci::CircleInput *_input = nullptr;
  luci::CircleHardSwish *_hardswish = nullptr;
  luci::CircleOutput *_output = nullptr;
};

class DecomposeHardSwishPass : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    const int N = 1;
    const int H = 4;
    const int W = 4;
    const int C = 3;

    // graph input and output
    auto graph_input = _hardswish_g._g.inputs()->create();
    auto graph_output = _hardswish_g._g.outputs()->create();

    // CircleInput
    _hardswish_g._input = _hardswish_g._g.nodes()->create<luci::CircleInput>();
    _hardswish_g._input->index(graph_input->index());
    _hardswish_g._input->shape({N, H, W, C});
    _hardswish_g._input->dtype(loco::DataType::FLOAT32);
    _hardswish_g._input->name("input");

    // CircleHardSwish
    _hardswish_g._hardswish = _hardswish_g._g.nodes()->create<luci::CircleHardSwish>();
    _hardswish_g._hardswish->features(_hardswish_g._input);
    _hardswish_g._hardswish->shape({N, H, W, C});
    _hardswish_g._hardswish->dtype(loco::DataType::FLOAT32);
    _hardswish_g._hardswish->name("hardswish");

    // CircleOutput
    _hardswish_g._output = _hardswish_g._g.nodes()->create<luci::CircleOutput>();
    _hardswish_g._output->index(graph_output->index());
    _hardswish_g._output->from(_hardswish_g._hardswish);
    _hardswish_g._output->shape({N, H, W, C});
    _hardswish_g._output->dtype(loco::DataType::FLOAT32);
    _hardswish_g._output->name("output");
  }

protected:
  luci::DecomposeHardSwishPass _pass;
  HardSwishGraph _hardswish_g;
};

} // namespace

TEST_F(DecomposeHardSwishPass, name)
{
  auto const name = _pass.name();
  ASSERT_NE(nullptr, name);
}

/**
 *  Decomposed graph looks like below.
 *
 *      [CircleInput]  [CircleConst]
 *          |    \       /
 *          |     \     /
 *          |   [CircleAdd]
 *          |        |
 *          |        |
 *          \  [CircleRelu6] [CircleConst]
 *           \        \        /
 *            \        \      /
 *             \      [CircleMul]
 *              \       /
 *               \     /
 *             [CircleMul]
 *                  |
 *                  |
 *             [CircleOutput]
 *
 */
TEST_F(DecomposeHardSwishPass, simple_test)
{
  auto ret = _pass.run(&_hardswish_g._g);
  EXPECT_TRUE(ret);

  auto mul2 = dynamic_cast<luci::CircleMul *>(_hardswish_g._output->from());
  EXPECT_NE(nullptr, mul2);

  auto input2 = dynamic_cast<luci::CircleInput *>(mul2->x());
  EXPECT_NE(nullptr, input2);

  auto mul1 = dynamic_cast<luci::CircleMul *>(mul2->y());
  EXPECT_NE(nullptr, mul1);

  auto relu6 = dynamic_cast<luci::CircleRelu6 *>(mul1->x());
  EXPECT_NE(nullptr, relu6);

  auto mul_const = dynamic_cast<luci::CircleConst *>(mul1->y());
  EXPECT_NE(nullptr, mul_const);
  EXPECT_FLOAT_EQ(1. / 6., mul_const->at<loco::DataType::FLOAT32>(0));

  auto add = dynamic_cast<luci::CircleAdd *>(relu6->features());
  EXPECT_NE(nullptr, add);

  auto input1 = dynamic_cast<luci::CircleInput *>(add->x());
  EXPECT_NE(nullptr, input1);

  auto add_const = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, add_const);
  EXPECT_FLOAT_EQ(3., add_const->at<loco::DataType::FLOAT32>(0));
}
