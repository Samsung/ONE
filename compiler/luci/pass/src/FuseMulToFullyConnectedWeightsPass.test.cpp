/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseMulToFullyConnectedWeightsPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

template <loco::DataType DT> class FuseMulToFullyConnectedWeightsPassTestGraph : public TestIOGraph
{
public:
  FuseMulToFullyConnectedWeightsPassTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({3, 4}, {3, 6});

    _mul = g()->nodes()->create<luci::CircleMul>();
    _mul_s = g()->nodes()->create<luci::CircleConst>();
    _fc = g()->nodes()->create<luci::CircleFullyConnected>();
    _fc_w = g()->nodes()->create<luci::CircleConst>();
    _fc_b = g()->nodes()->create<luci::CircleConst>();

    _mul->name("mul");
    _mul_s->name("mul_s");
    _fc->name("fc");
    _fc_w->name("fc_w");
    _fc_b->name("fc_b");

    _mul->dtype(DT);
    _fc->dtype(DT);
    _mul->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc->fusedActivationFunction(luci::FusedActFunc::NONE);

    _mul_s->rank(1);
    _mul_s->dim(0) = 3;
    _mul_s->dtype(DT);
    _mul_s->size<DT>(3);
    for (uint32_t i = 0; i < 3; ++i)
    {
      _mul_s->at<DT>(i) = 1.0f;
    }

    _fc_w->rank(2);
    _fc_w->dim(0) = 6;
    _fc_w->dim(1) = 4;
    _fc_w->dtype(DT);
    _fc_w->size<DT>(4 * 6);
    for (uint32_t i = 0; i < 4 * 6; ++i)
    {
      _fc_w->at<DT>(i) = 1.0f;
    }

    _fc_b->rank(1);
    _fc_b->dim(0) = 6;
    _fc_b->dtype(DT);
    _fc_b->size<DT>(6);
    for (uint32_t i = 0; i < 6; ++i)
    {
      _fc_b->at<DT>(i) = 1.0f;
    }

    _mul->x(input());
    _mul->y(_mul_s);
    _fc->input(_mul);
    _fc->weights(_fc_w);
    _fc->bias(_fc_b);

    output()->from(_fc);
  }

  luci::CircleMul *_mul = nullptr;
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleConst *_mul_s = nullptr;
  luci::CircleConst *_fc_w = nullptr;
  luci::CircleConst *_fc_b = nullptr;
};

class FuseMulToFullyConnectedWeightsPassTest : public ::testing::Test
{
public:
  FuseMulToFullyConnectedWeightsPassTest() = default;

protected:
  FuseMulToFullyConnectedWeightsPassTestGraph<loco::DataType::FLOAT32> _graph;
  luci::FuseMulToFullyConnectedWeightsPass _pass;
};

class FuseMulToFullyConnectedWeightsPassS32Test : public ::testing::Test
{
public:
  FuseMulToFullyConnectedWeightsPassS32Test() = default;

protected:
  FuseMulToFullyConnectedWeightsPassTestGraph<loco::DataType::S32> _graph;
  luci::FuseMulToFullyConnectedWeightsPass _pass;
};

} // namespace

TEST_F(FuseMulToFullyConnectedWeightsPassTest, name)
{
  auto const name = _pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FuseMulToFullyConnectedWeightsPassTest, fuse_mul_to_fc_weights)
{
  _graph.init();

  EXPECT_TRUE(_pass.run(_graph.g()));
}

TEST_F(FuseMulToFullyConnectedWeightsPassTest, mul_fused_act_NEG)
{
  _graph.init();

  _graph._mul->fusedActivationFunction(luci::FusedActFunc::RELU);

  EXPECT_FALSE(_pass.run(_graph.g()));
}

TEST_F(FuseMulToFullyConnectedWeightsPassTest, mul_d2_NEG)
{
  _graph.init();

  _graph._mul_s->rank(2);
  _graph._mul_s->dim(0) = 1;
  _graph._mul_s->dim(1) = 3;

  EXPECT_FALSE(_pass.run(_graph.g()));
}

TEST_F(FuseMulToFullyConnectedWeightsPassS32Test, dtype_s32_NEG)
{
  _graph.init();

  EXPECT_FALSE(_pass.run(_graph.g()));
}
