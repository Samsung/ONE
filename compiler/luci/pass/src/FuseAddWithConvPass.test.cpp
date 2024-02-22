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

#include "luci/Pass/FuseAddWithConvPass.h"

#include "helpers/CreateCircleConst.h"

#include <luci/IR/CircleNodes.h>
#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using namespace luci::test;

namespace
{

#define FILTER_O 4
#define FILTER_H 1
#define FILTER_W 1
#define FILTER_I 6

class Conv2DAddGraphlet
{
public:
  Conv2DAddGraphlet() = default;

  void init(loco::Graph *g)
  {
    const ShapeU32 filter_shape = {FILTER_O, FILTER_H, FILTER_W, FILTER_I};
    const ShapeU32 bias_shape = {FILTER_O};

    _conv_f = luci::create_const_node(g, loco::DataType::FLOAT32, filter_shape, 0.5f);
    _conv_b = luci::create_const_node(g, loco::DataType::FLOAT32, bias_shape, 0.5f);
    _conv_f->name("conv_f");
    _conv_b->name("conv_b");

    _conv = g->nodes()->create<luci::CircleConv2D>();
    _conv->filter(_conv_f);
    _conv->bias(_conv_b);
    _conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv->dtype(loco::DataType::FLOAT32);
    _conv->shape({1, 3, 3, FILTER_O});
    _conv->name("conv");

    const ShapeU32 add_shape = {1, 1, 1, FILTER_O};
    _add_y = luci::create_const_node(g, loco::DataType::FLOAT32, add_shape, 0.5f);
    _add_y->name("add_y");

    _add = g->nodes()->create<luci::CircleAdd>();
    _add->x(_conv);
    _add->y(_add_y);
    _add->fusedActivationFunction(luci::FusedActFunc::RELU);
    _add->dtype(loco::DataType::FLOAT32);
    _add->shape({1, 3, 3, FILTER_O});
    _add->name("add");

    // for negative test
    const ShapeU32 add_shape_2 = {FILTER_O, FILTER_I};
    _add_y_2 = luci::create_const_node(g, loco::DataType::FLOAT32, add_shape_2, 0.5f);
    _add_y_2->name("add_y_2");
  }

protected:
  luci::CircleConv2D *_conv = nullptr;
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_conv_f = nullptr;
  luci::CircleConst *_conv_b = nullptr;
  luci::CircleConst *_add_y = nullptr;
  luci::CircleConst *_add_y_2 = nullptr;
};

class FuseAddWithConvTestGraph : public TestIOGraph, public Conv2DAddGraphlet
{
public:
  FuseAddWithConvTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 3, 3, FILTER_I}, {1, 3, 3, FILTER_O});
    Conv2DAddGraphlet::init(g());

    _conv->input(input());
    output()->from(_add);
  }

  void add_use_2()
  {
    // set to not compatible shape
    _add->y(_add_y_2);
  }
};

class FuseAddWithConvPassTest : public ::testing::Test, public FuseAddWithConvTestGraph
{
public:
  luci::FuseAddWithConvPass pass;
};

} // namespace

TEST_F(FuseAddWithConvPassTest, simple_test)
{
  init();

  // Add should exist
  auto add = dynamic_cast<luci::CircleAdd *>(output()->from());
  EXPECT_NE(nullptr, add);

  EXPECT_TRUE(pass.run(g()));

  // expect Add is fused into Conv
  auto conv = dynamic_cast<luci::CircleConv2D *>(output()->from());
  EXPECT_NE(nullptr, conv);
}

TEST_F(FuseAddWithConvPassTest, wrong_add_shape_NEG)
{
  init();
  add_use_2();

  // Add const shape is not compatible
  EXPECT_FALSE(pass.run(g()));
}
