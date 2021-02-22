/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseActivationFunctionPass.h"

#include <luci/IR/CircleNodes.h>

#include "test/TestIOGraph.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *     [CircleConv2D]
 *           |
 *      [CircleRelu]
 *           |
 *     [CircleConv2D]
 *
 *  AFTER
 *
 *     [CircleConv2D]
 *           |
 *     [CircleConv2D]
 *
 */
class ConvReluConvGraphlet
{
public:
  ConvReluConvGraphlet() = default;

  void init(loco::Graph *g)
  {
    _conv1 = g->nodes()->create<luci::CircleConv2D>();
    _conv2 = g->nodes()->create<luci::CircleConv2D>();
    _relu = g->nodes()->create<luci::CircleRelu>();
    _conv1_f = g->nodes()->create<luci::CircleConst>();
    _conv1_b = g->nodes()->create<luci::CircleConst>();
    _conv2_f = g->nodes()->create<luci::CircleConst>();
    _conv2_b = g->nodes()->create<luci::CircleConst>();

    _conv1->fusedActivationFunction(luci::FusedActFunc::NONE);

    _conv1->name("conv1");
    _conv2->name("conv2");
    _relu->name("relu");
    _conv1_f->name("conv1f");
    _conv1_b->name("conv1b");
    _conv2_f->name("conv2f");
    _conv2_b->name("conv2b");
  }

public:
  luci::CircleRelu *relu() { return _relu; }
  luci::CircleConv2D *conv1() { return _conv1; }
  luci::CircleConv2D *conv2() { return _conv2; }

protected:
  luci::CircleConv2D *_conv1 = nullptr;
  luci::CircleConv2D *_conv2 = nullptr;
  luci::CircleRelu *_relu = nullptr;
  luci::CircleConst *_conv1_f = nullptr;
  luci::CircleConst *_conv1_b = nullptr;
  luci::CircleConst *_conv2_f = nullptr;
  luci::CircleConst *_conv2_b = nullptr;
};

class FuseActTestGraph : public TestIOGraph, public ConvReluConvGraphlet
{
public:
  FuseActTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    ConvReluConvGraphlet::init(g());

    _conv1->input(input());
    _conv1->filter(_conv1_f);
    _conv1->bias(_conv1_b);

    _relu->features(_conv1);

    _conv2->input(_relu);
    _conv2->filter(_conv2_f);
    _conv2->bias(_conv2_b);

    output()->from(_conv2);
  }
};

class ConvHasMultiSuccGraph : public TestIOGraph, public ConvReluConvGraphlet
{
public:
  ConvHasMultiSuccGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    ConvReluConvGraphlet::init(g());

    _conv1->input(input());
    _conv1->filter(_conv1_f);
    _conv1->bias(_conv1_b);

    _relu->features(_conv1);

    _conv2->input(_conv1);
    _conv2->filter(_conv2_f);
    _conv2->bias(_conv2_b);

    output()->from(_relu); // We need to check from relu
  }
};

// TODO use ::testing::Test

} // namespace

TEST(FuseActivationFunctionPassTest, name)
{
  luci::FuseActivationFunctionPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FusePreActivationBatchNorm, fuse_activation_function)
{
  FuseActTestGraph g;
  luci::FuseActivationFunctionPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
  EXPECT_EQ(g.conv1(), g.conv2()->input());
}

TEST(FusePreActivationBatchNorm, fuse_activation_function_dup_relu)
{
  FuseActTestGraph g;
  luci::FuseActivationFunctionPass pass;

  g.init();
  g.conv1()->fusedActivationFunction(luci::FusedActFunc::RELU);

  EXPECT_TRUE(pass.run(g.g()));
  EXPECT_EQ(g.conv1(), g.conv2()->input());
}

TEST(FusePreActivationBatchNorm, fuse_activation_function_mulsucc_NEG)
{
  ConvHasMultiSuccGraph g;
  luci::FuseActivationFunctionPass pass;

  g.init();

  // Relu input Conv2D has multiple successors
  EXPECT_FALSE(pass.run(g.g()));
}

TEST(FusePreActivationBatchNorm, fuse_activation_function_tanh_NEG)
{
  FuseActTestGraph g;
  luci::FuseActivationFunctionPass pass;

  g.init();
  g.conv1()->fusedActivationFunction(luci::FusedActFunc::TANH);

  // Relu input Conv2D already has activation function
  EXPECT_FALSE(pass.run(g.g()));
}
