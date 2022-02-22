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

#include "luci/Pass/ExpandBroadcastConstPass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

class ExpandBroadcastConstTest : public ::testing::Test
{
public:
  ExpandBroadcastConstTest()
  {
    _x = _g.nodes()->create<luci::CircleInput>();
    _y = _g.nodes()->create<luci::CircleConst>();
    _add = _g.nodes()->create<luci::CircleAdd>();
    _output = _g.nodes()->create<luci::CircleOutput>();

    auto graph_input = _g.inputs()->create();
    graph_input->dtype(loco::DataType::FLOAT32);
    graph_input->shape({1, H, W, D});
    _x->index(graph_input->index());
    _x->dtype(graph_input->dtype());
    _x->shape({1, H, W, D});

    auto graph_output = _g.outputs()->create();
    graph_output->dtype(loco::DataType::FLOAT32);
    graph_output->shape({1, H, W, D});
    _output->index(graph_output->index());
    _output->dtype(graph_output->dtype());
    _output->shape({1, H, W, D});

    _y->dtype(loco::DataType::FLOAT32);
    _y->shape({1, H, W, 1});
    _y->size<loco::DataType::FLOAT32>(16);

    _add->dtype(loco::DataType::FLOAT32);
    _add->fusedActivationFunction(luci::FusedActFunc::NONE);
    _add->x(_x);
    _add->y(_y);
    _add->shape({1, H, W, D});

    _output->from(_add);

    _x->name("input");
    _output->name("output");
  }

protected:
  uint32_t const H = 4;
  uint32_t const W = 4;
  uint32_t const D = 3;

protected:
  loco::Graph _g;
  luci::CircleAdd *_add = nullptr;
  luci::CircleInput *_x = nullptr;
  luci::CircleConst *_y = nullptr;
  luci::CircleOutput *_output = nullptr;
};

} // namespace

TEST_F(ExpandBroadcastConstTest, name)
{
  luci::ExpandBroadcastConstPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(ExpandBroadcastConstTest, remove_broadcast)
{
  for (uint32_t i = 0; i < H * W; ++i)
    _y->at<loco::DataType::FLOAT32>(i) = static_cast<float>(i);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto broadcasted_const = dynamic_cast<luci::CircleConst *>(_add->y());
  ASSERT_NE(broadcasted_const, nullptr);

  EXPECT_EQ(broadcasted_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(broadcasted_const->dim(1).value(), H);
  EXPECT_EQ(broadcasted_const->dim(2).value(), W);
  EXPECT_EQ(broadcasted_const->dim(3).value(), D);
  EXPECT_EQ(broadcasted_const->size<loco::DataType::FLOAT32>(), H * W * D);

  for (uint32_t i = 0; i < H * W; ++i)
  {
    for (uint32_t d = 0; d < D; ++d)
    {
      EXPECT_NEAR(broadcasted_const->at<loco::DataType::FLOAT32>(i + H * W * d),
                  static_cast<float>(i), std::numeric_limits<float>::min());
    }
  }
}

TEST_F(ExpandBroadcastConstTest, remove_broadcast_multiple_successors)
{
  auto const circle_sqrt = _g.nodes()->create<luci::CircleSqrt>();
  circle_sqrt->dtype(loco::DataType::FLOAT32);
  circle_sqrt->shape({1, H, W, 1});
  circle_sqrt->x(_y);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto broadcasted_const = dynamic_cast<luci::CircleConst *>(_add->y());
  auto original_const = dynamic_cast<luci::CircleConst *>(circle_sqrt->x());

  ASSERT_NE(broadcasted_const, nullptr);
  EXPECT_EQ(broadcasted_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(broadcasted_const->dim(3).value(), D);
  EXPECT_EQ(broadcasted_const->size<loco::DataType::FLOAT32>(), H * W * D);

  // Check if another successor's node was left intact
  ASSERT_NE(original_const, nullptr);
  EXPECT_EQ(original_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(original_const->dim(3).value(), 1);
  EXPECT_EQ(original_const->size<loco::DataType::FLOAT32>(), H * W * 1);
}

TEST_F(ExpandBroadcastConstTest, broadcast_impossible_NEG)
{
  _y->shape({1, H, W, 2});
  _y->size<loco::DataType::FLOAT32>(H * W * (D - 1));

  luci::ExpandBroadcastConstPass pass;
  ASSERT_FALSE(pass.run(&_g));
}
