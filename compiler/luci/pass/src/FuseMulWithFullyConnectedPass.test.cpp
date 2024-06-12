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

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class FullyConnectedMulGraphlet
{
public:
  FullyConnectedMulGraphlet() = default;

public:
  void init(loco::Graph *g, bool activation)
  {
    _fc = g->nodes()->create<luci::CircleFullyConnected>();
    _fc_weight = g->nodes()->create<luci::CircleConst>();
    _fc_bias = g->nodes()->create<luci::CircleConst>();
    _mul = g->nodes()->create<luci::CircleMul>();
    _mul_const = g->nodes()->create<luci::CircleConst>();

    if (activation)
    {
      _fc->fusedActivationFunction(luci::FusedActFunc::RELU);
    }
    else
    {
      _fc->fusedActivationFunction(luci::FusedActFunc::NONE);
    }

    _fc->dtype(loco::DataType::FLOAT32);
    _fc_weight->dtype(loco::DataType::FLOAT32);
    _fc_bias->dtype(loco::DataType::FLOAT32);
    _mul->dtype(loco::DataType::FLOAT32);
    _mul_const->dtype(loco::DataType::FLOAT32);

    _fc->name("fc");
    _fc_weight->name("weights");
    _fc_bias->name("bias");
    _mul->name("mul");
    _mul_const->name("mul_const");

    _fc_weight->shape({8, 10});
    _fc_bias->shape({8});

    _mul->fusedActivationFunction(luci::FusedActFunc::NONE);

    _mul_const->shape({1, 1, 8});
    _mul_const->size<loco::DataType::FLOAT32>(8);
    for (uint32_t i = 0; i < 8; i++)
    {
      _mul_const->at<loco::DataType::FLOAT32>(i) = 1.f;
    }

    {
      // initialize bias
      _fc_bias->size<loco::DataType::FLOAT32>(8);
      for (uint32_t i = 0; i < 8; i++)
      {
        _fc_bias->at<loco::DataType::FLOAT32>(i) = 0.f;
      }
    }

    {
      // initialize filter
      _fc_weight->size<loco::DataType::FLOAT32>(8 * 10);
      for (uint32_t i = 0; i < _fc_weight->size<loco::DataType::FLOAT32>(); i++)
      {
        _fc_weight->at<loco::DataType::FLOAT32>(i) = 1.f;
      }
    }

    _fc->weights(_fc_weight);
    _fc->bias(_fc_bias);
    _mul->x(_fc);
    _mul->y(_mul_const);
  }

protected:
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleConst *_fc_weight = nullptr;
  luci::CircleConst *_fc_bias = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_mul_const = nullptr;
};

class FullyConnectedMulGraph : public TestIOGraph, public FullyConnectedMulGraphlet
{
public:
  FullyConnectedMulGraph() = default;

public:
  void init(bool activation)
  {
    TestIOGraph::init({1, 10}, {1, 8});
    FullyConnectedMulGraphlet::init(g(), activation);

    _fc->input(input());
    output()->from(_mul);
  }
};

} // namespace

TEST(FuseMulWithFullyConnectedPass, name_test)
{
  luci::FuseMulWithFullyConnectedPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FuseMulWithFullyConnectedPass, simple_test)
{
  luci::FuseMulWithFullyConnectedPass pass;

  FullyConnectedMulGraph g;
  g.init(false);

  ASSERT_TRUE(pass.run(g.g()));

  // check Mul is removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto mul = dynamic_cast<luci::CircleMul *>(node))
      count++;
  }
  ASSERT_EQ(0, count);
}

TEST(FuseMulWithFullyConnectedPass, activation_blocks_removal_NEG)
{
  luci::FuseMulWithFullyConnectedPass pass;
  FullyConnectedMulGraph g;
  g.init(true);

  ASSERT_FALSE(pass.run(g.g()));

  // check Mul is not removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto mul = dynamic_cast<luci::CircleMul *>(node))
      count++;
  }
  ASSERT_EQ(1, count);
}
