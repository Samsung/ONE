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
#include "luci/Pass/FuseMulWithConvPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class ConvMulGraphlet
{
public:
  ConvMulGraphlet() = default;

public:
  void init(loco::Graph *g, bool activation, bool mul_const_shape)
  {
    _conv = g->nodes()->create<luci::CircleConv2D>();
    _conv_filter = g->nodes()->create<luci::CircleConst>();
    _conv_bias = g->nodes()->create<luci::CircleConst>();
    _mul = g->nodes()->create<luci::CircleMul>();
    _mul_const = g->nodes()->create<luci::CircleConst>();

    if (activation)
    {
      _conv->fusedActivationFunction(luci::FusedActFunc::RELU);
    }
    else
    {
      _conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    }

    _conv->dtype(loco::DataType::FLOAT32);
    _conv_filter->dtype(loco::DataType::FLOAT32);
    _conv_bias->dtype(loco::DataType::FLOAT32);
    _mul->dtype(loco::DataType::FLOAT32);
    _mul_const->dtype(loco::DataType::FLOAT32);

    _conv->name("conv");
    _conv_filter->name("conv_filter");
    _conv_bias->name("conv_bias");
    _mul->name("mul");
    _mul_const->name("mul_const");

    _conv_filter->shape({_output_channels, 1, 1, _input_channels});
    _conv_bias->shape({_output_channels});
    if (mul_const_shape)
    {
      _mul_const->shape({1, 1, _input_dim, _output_channels});
    }
    else
    {
      _mul_const->shape({1, 1, 1, _output_channels});
      // initialize _mul_const for positive test
      _mul_const->size<loco::DataType::FLOAT32>(_output_channels);
      for (uint32_t i = 0; i < _output_channels; i++)
      {
        _mul_const->at<loco::DataType::FLOAT32>(i) = 1.f;
      }
    }

    {
      // initialize bias
      _conv_bias->size<loco::DataType::FLOAT32>(_output_channels);
      for (uint32_t i = 0; i < _output_channels; i++)
      {
        _conv_bias->at<loco::DataType::FLOAT32>(i) = 0.f;
      }
    }

    {
      // initialize filter
      _conv_filter->size<loco::DataType::FLOAT32>(_output_channels * _input_channels);
      for (uint32_t i = 0; i < _conv_filter->size<loco::DataType::FLOAT32>(); i++)
      {
        _conv_filter->at<loco::DataType::FLOAT32>(i) = 1.f;
      }
    }

    _conv->filter(_conv_filter);
    _conv->bias(_conv_bias);
    _conv->padding(luci::Padding::VALID);
    _conv->stride()->h(1);
    _conv->stride()->w(1);
    _mul->x(_mul_const);
    _mul->y(_conv);
  }

protected:
  luci::CircleConv2D *_conv = nullptr;
  luci::CircleConst *_conv_filter = nullptr;
  luci::CircleConst *_conv_bias = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_mul_const = nullptr;

  const uint32_t _input_channels = 32;
  const uint32_t _output_channels = 64;
  const uint32_t _input_dim = 64;
  const ShapeU32 _input_shape = {1, _input_dim, _input_dim, _input_channels};
  const ShapeU32 _output_shape = {1, _input_dim, _input_dim, _output_channels};
};

class ConvMulGraph : public TestIOGraph, public ConvMulGraphlet
{
public:
  ConvMulGraph() = default;

public:
  void init(bool activation, bool mul_const_shape)
  {
    TestIOGraph::init(_input_shape, _output_shape);
    ConvMulGraphlet::init(g(), activation, mul_const_shape);

    _conv->input(input());
    output()->from(_mul);
  }
};

} // namespace

TEST(FuseMulWithConvPass, name_test)
{
  luci::FuseMulWithConvPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FuseMulWithConvPass, simple_test)
{
  luci::FuseMulWithConvPass pass;

  ConvMulGraph g;
  g.init(false, false);

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

TEST(FuseMulWithConvPass, not_removed_NEG)
{
  luci::FuseMulWithConvPass pass;
  ConvMulGraph g;
  g.init(false, true);

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

TEST(FuseMulWithConvPass, activation_blocks_removal_NEG)
{
  luci::FuseMulWithConvPass pass;
  ConvMulGraph g;
  g.init(true, false);

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
