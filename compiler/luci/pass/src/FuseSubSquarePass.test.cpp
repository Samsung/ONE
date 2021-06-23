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

#include "luci/Pass/FuseSubSquarePass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class SubSquareGraphlet
{
public:
  SubSquareGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    std::vector<uint32_t> shape_out_v = shape_out;

    _sub = g->nodes()->create<luci::CircleSub>();
    _square = g->nodes()->create<luci::CircleSquare>();

    _sub->fusedActivationFunction(luci::FusedActFunc::NONE);

    _sub->dtype(loco::DataType::FLOAT32);
    _sub->shape(shape_in);

    _square->dtype(loco::DataType::FLOAT32);
    _square->shape(shape_out);

    _sub->name("sub");
    _square->name("square");
  }

  void sub_act_func(luci::FusedActFunc func)
  {
    assert(_sub != nullptr);
    _sub->fusedActivationFunction(func);
  }

protected:
  luci::CircleSub *_sub = nullptr;
  luci::CircleSquare *_square = nullptr;
};

class SubSquareGraph : public TestIsGraphlet<2>, public TestOGraphlet, public SubSquareGraphlet
{
public:
  SubSquareGraph() = default;

public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIsGraphlet<2>::init(g(), {shape_in, shape_in});
    TestOGraphlet::init(g(), shape_out);
    SubSquareGraphlet::init(g(), shape_in, shape_out);

    // connect network
    _sub->x(input(0));
    _sub->y(input(1));
    _square->x(_sub);

    output()->from(_square);
  }
};

class FuseSubSquarePassTest : public ::testing::Test
{
public:
  FuseSubSquarePassTest() = default;

  void run_pass(void)
  {
    while (_pass.run(_graph.g()))
      ;
  }

protected:
  SubSquareGraph _graph;
  luci::FuseSubSquarePass _pass;
};

} // namespace

TEST(FuseSubSquarePassNameTest, name)
{
  luci::FuseSubSquarePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FuseSubSquarePassTest, run)
{
  _graph.init({3, 3}, {3, 3});

  run_pass();

  auto squdiff = dynamic_cast<luci::CircleSquaredDifference *>(_graph.output()->from());
  ASSERT_NE(nullptr, squdiff);
}

TEST_F(FuseSubSquarePassTest, run_wront_act_NEG)
{
  _graph.init({3, 3}, {3, 3});

  _graph.sub_act_func(luci::FusedActFunc::RELU);

  run_pass();

  auto squdiff = dynamic_cast<luci::CircleSquaredDifference *>(_graph.output()->from());
  ASSERT_EQ(nullptr, squdiff);
  auto square = dynamic_cast<luci::CircleSquare *>(_graph.output()->from());
  ASSERT_NE(nullptr, square);
}
