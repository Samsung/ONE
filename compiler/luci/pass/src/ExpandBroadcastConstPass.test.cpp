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
#include "helpers/ArrayIndex.h"

#include <luci/IR/CircleNodes.h>

#include <limits> // std::numeric_limits

#include <gtest/gtest.h>

/****************************************************************************
 * TESTS FOR RANK 2
 ****************************************************************************/
namespace
{

class ExpandBroadcastConstRank2Graph
{
public:
  ExpandBroadcastConstRank2Graph()
  {
    _x = _g.nodes()->create<luci::CircleInput>();
    _y = _g.nodes()->create<luci::CircleConst>();
    _add = _g.nodes()->create<luci::CircleAdd>();
    _output = _g.nodes()->create<luci::CircleOutput>();

    auto graph_input = _g.inputs()->create();
    graph_input->dtype(loco::DataType::FLOAT32);
    graph_input->shape({N, D});
    _x->index(graph_input->index());
    _x->dtype(graph_input->dtype());
    _x->shape({N, D});

    _y->dtype(loco::DataType::FLOAT32);
    _y->shape({N, 1});
    _y->size<loco::DataType::FLOAT32>(N);

    auto graph_output = _g.outputs()->create();
    graph_output->dtype(loco::DataType::FLOAT32);
    graph_output->shape({N, D});
    _output->index(graph_output->index());
    _output->dtype(graph_output->dtype());
    _output->shape({N, D});

    _add->dtype(loco::DataType::FLOAT32);
    _add->fusedActivationFunction(luci::FusedActFunc::NONE);
    _add->x(_x);
    _add->y(_y);
    _add->shape({N, D});

    _output->from(_add);

    _x->name("input");
    _output->name("output");
  }

protected:
  uint32_t const N = 4;
  uint32_t const D = 3;

protected:
  loco::Graph _g;
  luci::CircleAdd *_add = nullptr;
  luci::CircleInput *_x = nullptr;
  luci::CircleConst *_y = nullptr;
  luci::CircleOutput *_output = nullptr;
};

class ExpandBroadcastRank2ConstTest : public ExpandBroadcastConstRank2Graph, public ::testing::Test
{
public:
  ExpandBroadcastRank2ConstTest() {}
};
} // namespace

TEST_F(ExpandBroadcastRank2ConstTest, remove_broadcast)
{
  for (uint32_t i = 0; i < N; ++i)
    _y->at<loco::DataType::FLOAT32>(i) = static_cast<float>(i);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto broadcasted_const = dynamic_cast<luci::CircleConst *>(_add->y());
  ASSERT_NE(broadcasted_const, nullptr);

  EXPECT_EQ(broadcasted_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(broadcasted_const->dim(0).value(), N);
  EXPECT_EQ(broadcasted_const->dim(1).value(), D);
  EXPECT_EQ(broadcasted_const->size<loco::DataType::FLOAT32>(), N * D);

  for (uint32_t i = 0; i < N; ++i)
  {
    for (uint32_t d = 0; d < D; ++d)
    {
      EXPECT_NEAR(broadcasted_const->at<loco::DataType::FLOAT32>(i * D + d), static_cast<float>(i),
                  std::numeric_limits<float>::min());
    }
  }
}

TEST_F(ExpandBroadcastRank2ConstTest, broadcast_impossible_NEG)
{
  _y->shape({N, D + 1});
  _y->size<loco::DataType::FLOAT32>(N * (D + 1));

  luci::ExpandBroadcastConstPass pass;
  ASSERT_FALSE(pass.run(&_g));
}

TEST_F(ExpandBroadcastRank2ConstTest, broadcast_diff_rank_NEG)
{
  _y->shape({N});
  _y->size<loco::DataType::FLOAT32>(N);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_FALSE(pass.run(&_g));
}

/****************************************************************************
 * TESTS FOR RANK 4
 ****************************************************************************/

namespace
{
class ExpandBroadcastConstRank4Graph
{
public:
  ExpandBroadcastConstRank4Graph()
  {
    _x = _g.nodes()->create<luci::CircleInput>();
    _y = _g.nodes()->create<luci::CircleConst>();
    _add = _g.nodes()->create<luci::CircleAdd>();
    _output = _g.nodes()->create<luci::CircleOutput>();

    auto graph_input = _g.inputs()->create();
    graph_input->dtype(loco::DataType::FLOAT32);
    graph_input->shape({N, H, W, D});
    _x->index(graph_input->index());
    _x->dtype(graph_input->dtype());
    _x->shape({N, H, W, D});

    auto graph_output = _g.outputs()->create();
    graph_output->dtype(loco::DataType::FLOAT32);
    graph_output->shape({N, H, W, D});
    _output->index(graph_output->index());
    _output->dtype(graph_output->dtype());
    _output->shape({N, H, W, D});

    _add->dtype(loco::DataType::FLOAT32);
    _add->fusedActivationFunction(luci::FusedActFunc::NONE);
    _add->x(_x);
    _add->y(_y);
    _add->shape({N, H, W, D});

    _output->from(_add);

    _x->name("input");
    _output->name("output");
  }

protected:
  uint32_t const N = 2;
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

class ExpandBroadcastRank4ConstTest1 : public ExpandBroadcastConstRank4Graph, public ::testing::Test
{
public:
  ExpandBroadcastRank4ConstTest1()
  {
    _y->dtype(loco::DataType::FLOAT32);
    _y->shape({N, H, W, 1});
    _y->size<loco::DataType::FLOAT32>(N * H * W * 1);
  }
};

class ExpandBroadcastRank4ConstTest2 : public ExpandBroadcastConstRank4Graph, public ::testing::Test
{
public:
  ExpandBroadcastRank4ConstTest2()
  {
    _y->dtype(loco::DataType::FLOAT32);
    _y->shape({N, 1, W, D});
    _y->size<loco::DataType::FLOAT32>(N * 1 * W * D);
  }
};

} // namespace

TEST_F(ExpandBroadcastRank4ConstTest1, name)
{
  luci::ExpandBroadcastConstPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(ExpandBroadcastRank4ConstTest1, remove_broadcast)
{
  for (uint32_t i = 0; i < N * H * W; ++i)
    _y->at<loco::DataType::FLOAT32>(i) = static_cast<float>(i);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto broadcasted_const = dynamic_cast<luci::CircleConst *>(_add->y());
  ASSERT_NE(broadcasted_const, nullptr);

  EXPECT_EQ(broadcasted_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(broadcasted_const->dim(0).value(), N);
  EXPECT_EQ(broadcasted_const->dim(1).value(), H);
  EXPECT_EQ(broadcasted_const->dim(2).value(), W);
  EXPECT_EQ(broadcasted_const->dim(3).value(), D);
  EXPECT_EQ(broadcasted_const->size<loco::DataType::FLOAT32>(), N * H * W * D);

  for (uint32_t i = 0; i < N * H * W; ++i)
  {
    for (uint32_t d = 0; d < D; ++d)
    {
      EXPECT_NEAR(broadcasted_const->at<loco::DataType::FLOAT32>(i * D + d), static_cast<float>(i),
                  std::numeric_limits<float>::min());
    }
  }
}

TEST_F(ExpandBroadcastRank4ConstTest1, remove_broadcast_multiple_successors)
{
  auto const circle_sqrt = _g.nodes()->create<luci::CircleSqrt>();
  circle_sqrt->dtype(loco::DataType::FLOAT32);
  circle_sqrt->shape({N, H, W, 1});
  circle_sqrt->x(_y);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto broadcasted_const = dynamic_cast<luci::CircleConst *>(_add->y());
  auto original_const = dynamic_cast<luci::CircleConst *>(circle_sqrt->x());

  ASSERT_NE(broadcasted_const, nullptr);
  EXPECT_EQ(broadcasted_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(broadcasted_const->dim(3).value(), D);
  EXPECT_EQ(broadcasted_const->size<loco::DataType::FLOAT32>(), N * H * W * D);

  // Check if another successor's node was left intact
  ASSERT_NE(original_const, nullptr);
  EXPECT_EQ(original_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(original_const->dim(3).value(), 1);
  EXPECT_EQ(original_const->size<loco::DataType::FLOAT32>(), N * H * W * 1);
}

TEST_F(ExpandBroadcastRank4ConstTest1, broadcast_impossible_NEG)
{
  _y->shape({N, H, W, D + 1});
  _y->size<loco::DataType::FLOAT32>(N * H * W * (D + 1));

  luci::ExpandBroadcastConstPass pass;
  ASSERT_FALSE(pass.run(&_g));
}

TEST_F(ExpandBroadcastRank4ConstTest1, broadcast_diff_rank_NEG)
{
  _y->shape({N, H, W});
  _y->size<loco::DataType::FLOAT32>(N * H * W);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_FALSE(pass.run(&_g));
}

TEST_F(ExpandBroadcastRank4ConstTest2, remove_broadcast)
{
  for (uint32_t i = 0; i < N * W * D; ++i)
    _y->at<loco::DataType::FLOAT32>(i) = static_cast<float>(i);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto broadcasted_const = dynamic_cast<luci::CircleConst *>(_add->y());
  ASSERT_NE(broadcasted_const, nullptr);

  EXPECT_EQ(broadcasted_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(broadcasted_const->dim(0).value(), N);
  EXPECT_EQ(broadcasted_const->dim(1).value(), H);
  EXPECT_EQ(broadcasted_const->dim(2).value(), W);
  EXPECT_EQ(broadcasted_const->dim(3).value(), D);
  EXPECT_EQ(broadcasted_const->size<loco::DataType::FLOAT32>(), N * H * W * D);

  auto const idx = luci::Array4DIndex(N, H, W, D);

  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w)
        for (uint32_t d = 0; d < D; ++d)
          EXPECT_NEAR(broadcasted_const->at<loco::DataType::FLOAT32>(idx(n, h, w, d)),
                      static_cast<float>(n * W * D + w * D + d), std::numeric_limits<float>::min());
}

TEST_F(ExpandBroadcastRank4ConstTest2, remove_broadcast_multiple_successors)
{
  auto const circle_sqrt = _g.nodes()->create<luci::CircleSqrt>();
  circle_sqrt->dtype(loco::DataType::FLOAT32);
  circle_sqrt->shape({N, 1, W, D});
  circle_sqrt->x(_y);

  luci::ExpandBroadcastConstPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto broadcasted_const = dynamic_cast<luci::CircleConst *>(_add->y());
  auto original_const = dynamic_cast<luci::CircleConst *>(circle_sqrt->x());

  ASSERT_NE(broadcasted_const, nullptr);
  EXPECT_EQ(broadcasted_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(broadcasted_const->dim(1).value(), H);
  EXPECT_EQ(broadcasted_const->size<loco::DataType::FLOAT32>(), N * H * W * D);

  // Check if another successor's node was left intact
  ASSERT_NE(original_const, nullptr);
  EXPECT_EQ(original_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(original_const->dim(1).value(), 1);
  EXPECT_EQ(original_const->size<loco::DataType::FLOAT32>(), N * 1 * W * D);
}

TEST_F(ExpandBroadcastRank4ConstTest2, broadcast_impossible_NEG)
{
  _y->shape({N, H, W + 1, D + 1});
  _y->size<loco::DataType::FLOAT32>(N * H * (W + 1) * (D + 1));

  luci::ExpandBroadcastConstPass pass;
  ASSERT_FALSE(pass.run(&_g));
}

TEST_F(ExpandBroadcastRank4ConstTest2, broadcast_diff_rank_NEG)
{
  _y->shape({N, H, W + 4});
  _y->size<loco::DataType::FLOAT32>(N * H * (W + 4));

  luci::ExpandBroadcastConstPass pass;
  ASSERT_FALSE(pass.run(&_g));
}
