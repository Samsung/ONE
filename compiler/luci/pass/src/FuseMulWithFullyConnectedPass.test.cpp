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

#include "luci/Pass/FuseMulWithFullyConnectedPass.h"
#include "helpers/CreateCircleConst.h"

#include <luci/IR/CircleNodes.h>
#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

#define DIM_ONE 8
#define DIM_TWO 4
#define MUL_VAL 2.0f

namespace
{

using namespace luci::test;

/**
 *  Graph for this test
 *
 *  BEFORE (without extra_fc_successor)
 *
 *         [FC]
 *           |
 *     [Mul w/ Relu]
 *
 *  BEFORE (with extra_fc_successor)
 *
 *         [FC]
 *           |
 *           |-------------------
 *           |                  |
 *           |                  |
 *     [Mul w/ Relu]       [other FC]
 *
 *  AFTER (if pass applied)
 *
 *      [FC w/ Relu] (weights and bias updated)
 *
 */
class FCMulGraphlet
{
public:
  void init(loco::Graph *g, luci::FusedActFunc fc_activation, bool is_mul_scalar, bool use_bias,
            bool extra_successor)
  {
    _fc = g->nodes()->create<luci::CircleFullyConnected>();

    std::vector<float> weights_val(DIM_ONE * DIM_TWO);
    for (uint32_t i = 0; i < DIM_ONE * DIM_TWO; i++)
      weights_val.at(i) = i;

    _fc_f = luci::create_const_node(g, loco::DataType::FLOAT32, {DIM_ONE, DIM_TWO}, weights_val);
    _fc->weights(_fc_f);

    if (use_bias)
    {
      std::vector<float> bias_val(DIM_ONE);
      for (uint32_t i = 0; i < DIM_ONE; i++)
        bias_val.at(i) = i;

      _fc_b = luci::create_const_node(g, loco::DataType::FLOAT32, {DIM_ONE}, bias_val);
    }
    else
    {
      // Create CircleOutputExclude -- no bias
      _fc_b = g->nodes()->create<luci::CircleOutputExclude>();
    }
    _fc->bias(_fc_b);

    _fc->fusedActivationFunction(fc_activation);
    _fc->dtype(loco::DataType::FLOAT32);
    _fc->shape({1, DIM_ONE});
    _fc->name("fc");

    if (extra_successor)
    {
      _extra_succ = g->nodes()->create<luci::CircleFullyConnected>();
      // Set previous FC as input to bump number of successors for it:
      _extra_succ->input(_fc);
      std::vector<float> weights_val(DIM_ONE * DIM_TWO);
      _extra_f =
        luci::create_const_node(g, loco::DataType::FLOAT32, {DIM_ONE, DIM_TWO}, weights_val);
      _extra_succ->weights(_extra_f);
      _extra_succ->bias(nullptr);
      _extra_succ->fusedActivationFunction(luci::FusedActFunc::NONE);
      _extra_succ->dtype(loco::DataType::FLOAT32);
      _extra_succ->shape({1, DIM_ONE});
      _extra_succ->name("extra_fc");
    }

    std::vector<float> mul_values;

    if (is_mul_scalar)
    {
      mul_values.push_back(static_cast<float>(MUL_VAL));
      _mul_c = luci::create_const_node(g, loco::DataType::FLOAT32, {}, mul_values);
    }
    else
    {
      for (uint32_t i = 0; i < DIM_ONE; i++)
      {
        mul_values.push_back(static_cast<float>(i));
      }
      _mul_c = luci::create_const_node(g, loco::DataType::FLOAT32, {1, 1, 1, DIM_ONE}, mul_values);
    }

    _mul = g->nodes()->create<luci::CircleMul>();
    _mul->x(_fc);
    _mul->y(_mul_c);
    _mul->fusedActivationFunction(luci::FusedActFunc::RELU);
    _mul->dtype(loco::DataType::FLOAT32);
    if (is_mul_scalar)
    {
      _mul->shape({1, DIM_ONE});
    }
    else
    {
      _mul->shape({1, 1, 1, DIM_ONE});
    }
    _mul->name("mul");
  }

public:
  luci::CircleFullyConnected *fc() { return _fc; }

  void to_fm_bias(void)
  {
    assert(_fc != nullptr);

    auto new_fc = _fc->graph()->nodes()->create<luci::CircleFullyConnected>();
    _fc->bias(new_fc);
  }

protected:
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_fc_f = nullptr;
  luci::CircleNode *_fc_b = nullptr;
  luci::CircleConst *_mul_c = nullptr;
  luci::CircleFullyConnected *_extra_succ = nullptr;
  luci::CircleConst *_extra_f = nullptr;
};

class FuseMulWithFCTestGraph : public TestIOGraph, public FCMulGraphlet
{
public:
  void init(luci::FusedActFunc fc_activation, bool is_mul_scalar, bool use_bias,
            bool extra_successor)
  {
    TestIOGraph::init({1, DIM_TWO}, {1, DIM_ONE});
    FCMulGraphlet::init(g(), fc_activation, is_mul_scalar, use_bias, extra_successor);

    _fc->input(input());

    output()->from(_mul);
  }
};

class FuseMulWithFullyConnectedPassTest : public ::testing::Test
{
public:
  FuseMulWithFCTestGraph g;
  luci::FuseMulWithFullyConnectedPass pass;
};

} // namespace

TEST_F(FuseMulWithFullyConnectedPassTest, fc_mul_tensor)
{
  g.init(luci::FusedActFunc::NONE, false /* is_mul_scalar */, true /* use_bias */,
         false /* extra_successor */);

  EXPECT_EQ(true, pass.run(g.g()));

  auto fc = dynamic_cast<luci::CircleFullyConnected *>(g.output()->from());
  EXPECT_NE(nullptr, fc);

  auto weights = loco::must_cast<luci::CircleConst *>(g.fc()->weights());
  auto weights_n = weights->dim(0).value();
  auto weights_m = weights->dim(1).value();
  uint32_t offset = 0;
  for (uint32_t i = 0; i < weights_n; i++)
  {
    for (uint32_t j = 0; j < weights_m; j++)
    {
      offset = i * weights_m + j;
      EXPECT_EQ(i * offset, weights->at<loco::DataType::FLOAT32>(offset));
    }
  }

  auto bias = loco::must_cast<luci::CircleConst *>(g.fc()->bias());
  for (uint32_t i = 0; i < bias->size<loco::DataType::FLOAT32>(); i++)
  {
    EXPECT_EQ(i * i, bias->at<loco::DataType::FLOAT32>(i));
  }
}

TEST_F(FuseMulWithFullyConnectedPassTest, fc_mul_scalar)
{
  g.init(luci::FusedActFunc::NONE, true /* is_mul_scalar */, true /* use_bias */,
         false /* extra_successor */);

  EXPECT_EQ(true, pass.run(g.g()));

  auto fc = dynamic_cast<luci::CircleFullyConnected *>(g.output()->from());
  EXPECT_NE(nullptr, fc);

  auto weights = loco::must_cast<luci::CircleConst *>(g.fc()->weights());
  auto weights_n = weights->dim(0).value();
  auto weights_m = weights->dim(1).value();
  uint32_t offset = 0;
  for (uint32_t i = 0; i < weights_n; i++)
  {
    for (uint32_t j = 0; j < weights_m; j++)
    {
      offset = i * weights_m + j;
      EXPECT_EQ(MUL_VAL * offset, weights->at<loco::DataType::FLOAT32>(offset));
    }
  }

  auto bias = loco::must_cast<luci::CircleConst *>(g.fc()->bias());
  for (uint32_t i = 0; i < bias->size<loco::DataType::FLOAT32>(); i++)
  {
    EXPECT_EQ(MUL_VAL * i, bias->at<loco::DataType::FLOAT32>(i));
  }
}

TEST_F(FuseMulWithFullyConnectedPassTest, fc_no_bias)
{
  g.init(luci::FusedActFunc::NONE, false /* is_mul_scalar */, false /* use_bias */,
         false /* extra_successor */);

  EXPECT_EQ(true, pass.run(g.g()));

  auto fc = dynamic_cast<luci::CircleFullyConnected *>(g.output()->from());
  EXPECT_NE(nullptr, fc);
  auto no_bias = dynamic_cast<luci::CircleOutputExclude *>(fc->bias());
  ASSERT_NE(nullptr, no_bias);

  auto weights = loco::must_cast<luci::CircleConst *>(g.fc()->weights());
  auto weights_n = weights->dim(0).value();
  auto weights_m = weights->dim(1).value();
  uint32_t offset = 0;
  for (uint32_t i = 0; i < weights_n; i++)
  {
    for (uint32_t j = 0; j < weights_m; j++)
    {
      offset = i * weights_m + j;
      EXPECT_EQ(i * offset, weights->at<loco::DataType::FLOAT32>(offset));
    }
  }
}

TEST_F(FuseMulWithFullyConnectedPassTest, bias_feature_map_NEG)
{
  g.init(luci::FusedActFunc::NONE, false /* is_mul_scalar */, true /* use_bias */,
         false /* extra_successor */);

  // Bias cannot be fused as it's passed as feature map.
  g.to_fm_bias();

  EXPECT_EQ(false, pass.run(g.g()));
}

TEST_F(FuseMulWithFullyConnectedPassTest, fc_with_activation_NEG)
{
  g.init(luci::FusedActFunc::RELU, false /* is_mul_scalar */, true /* use_bias */,
         false /* extra_successor */);

  EXPECT_EQ(false, pass.run(g.g()));
}

TEST_F(FuseMulWithFullyConnectedPassTest, fc_with_null_weights_NEG)
{
  g.init(luci::FusedActFunc::NONE, false /* is_mul_scalar */, true /* use_bias */,
         false /* extra_successor */);

  g.fc()->weights(nullptr);

  EXPECT_EQ(false, pass.run(g.g()));
}

TEST_F(FuseMulWithFullyConnectedPassTest, fc_with_extra_successor_NEG)
{
  g.init(luci::FusedActFunc::NONE, false /* is_mul_scalar */, true /* use_bias */,
         true /* extra_successor */);

  EXPECT_EQ(false, pass.run(g.g()));
}
