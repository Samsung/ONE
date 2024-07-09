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

#include "luci/Pass/TransformSqrtDivToRsqrtMulPass.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class SqrtDivGraphlet
{
public:
  SqrtDivGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    _div = g->nodes()->create<luci::CircleDiv>();
    _div->name("div");

    _sqrt = g->nodes()->create<luci::CircleSqrt>();
    _sqrt->name("sqrt");
  }

protected:
  luci::CircleDiv *_div = nullptr;
  luci::CircleSqrt *_sqrt = nullptr;
};

class SqrtDivGraph : public TestIOGraph, public SqrtDivGraphlet
{
public:
  SqrtDivGraph() = default;

public:
  void init(void)
  {
    TestIOGraph::init({1, 2, 3}, {1, 2, 3});
    SqrtDivGraphlet::init(g());

    _div->x(input());
    _div->y(_sqrt);

    _sqrt->x(input());

    output()->from(_div);
  }
};

// For negative test: Div input order does not match
class SqrtDivOrderGraph : public TestIOGraph, public SqrtDivGraphlet
{
public:
  SqrtDivOrderGraph() = default;

public:
  void init(void)
  {
    TestIOGraph::init({1, 2, 3}, {1, 2, 3});
    SqrtDivGraphlet::init(g());

    _div->x(_sqrt);
    _div->y(input());

    _sqrt->x(input());

    output()->from(_div);
  }
};

// For negative test: Div input x is Const
class SqrtDivConstGraph : public TestIOGraph, public SqrtDivGraphlet
{
public:
  SqrtDivConstGraph() = default;

public:
  void init(void)
  {
    TestIOGraph::init({1, 2, 3}, {1, 2, 3});
    SqrtDivGraphlet::init(g());

    _const = g()->nodes()->create<luci::CircleConst>();
    _const->name("const");

    _div->x(_const);
    _div->y(_sqrt);

    _sqrt->x(input());

    output()->from(_div);
  }

protected:
  luci::CircleConst *_const = nullptr;
};

class TransformSqrtDivToRsqrtMulPassTest : public ::testing::Test
{
protected:
  luci::TransformSqrtDivToRsqrtMulPass _pass;
};

} // namespace

TEST_F(TransformSqrtDivToRsqrtMulPassTest, name)
{
  auto const name = _pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(TransformSqrtDivToRsqrtMulPassTest, simple_run)
{
  SqrtDivGraph graph;
  graph.init();

  EXPECT_TRUE(_pass.run(graph.g()));

  // success pass will transform Div to Mul
  auto mul_node = dynamic_cast<luci::CircleMul *>(graph.output()->from());
  ASSERT_NE(nullptr, mul_node);
}

TEST_F(TransformSqrtDivToRsqrtMulPassTest, div_input_order_NEG)
{
  SqrtDivOrderGraph graph;
  graph.init();

  EXPECT_FALSE(_pass.run(graph.g()));
}

TEST_F(TransformSqrtDivToRsqrtMulPassTest, div_input_const_NEG)
{
  SqrtDivConstGraph graph;
  graph.init();

  EXPECT_FALSE(_pass.run(graph.g()));
}
