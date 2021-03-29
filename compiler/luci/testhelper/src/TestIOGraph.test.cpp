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

#include "luci/test/TestIOGraph.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class SqrtGraphlet
{
public:
  SqrtGraphlet() = default;

  void init(loco::Graph *g)
  {
    _sqrt = g->nodes()->create<luci::CircleSqrt>();
    _sqrt->name("sqrt");
  }

protected:
  luci::CircleSqrt *_sqrt = nullptr;
};

class AddGraphlet
{
public:
  AddGraphlet() = default;

  void init(loco::Graph *g)
  {
    _add = g->nodes()->create<luci::CircleAdd>();
    _add->name("add");
  }

protected:
  luci::CircleAdd *_add = nullptr;
};

class ConvGraphlet
{
public:
  ConvGraphlet() = default;

  void init(loco::Graph *g)
  {
    _conv = g->nodes()->create<luci::CircleConv2D>();
    _conv->name("conv");
  }

protected:
  luci::CircleConv2D *_conv = nullptr;
};

} // namespace

namespace
{

class TestOfTestIOGraph : public TestIOGraph, public SqrtGraphlet
{
public:
  TestOfTestIOGraph() = default;

public:
  void init(void)
  {
    TestIOGraph::init({1}, {1});
    SqrtGraphlet::init(g());

    _sqrt->x(input());

    output()->from(_sqrt);
  }
};

class TestOfTestI2OGraph : public TestIsGraphlet<2>, public TestOGraphlet, public AddGraphlet
{
public:
  TestOfTestI2OGraph() = default;

public:
  void init(void)
  {
    TestIsGraphlet<2>::init(g(), {{2, 3}, {2, 3}});
    TestOsGraphlet<1>::init(g(), {{2, 3}});
    AddGraphlet::init(g());

    _add->x(input(0));
    _add->y(input(1));

    output()->from(_add);
  }
};

class TestOfTestI3OGraph : public TestIsGraphlet<3>, public TestOGraphlet, public ConvGraphlet
{
public:
  TestOfTestI3OGraph() = default;

public:
  void init(void)
  {
    TestIsGraphlet<3>::init(g(), {{2, 3, 3, 4}, {1, 1}, {4}});
    TestOsGraphlet<1>::init(g(), {{2, 3, 3, 4}});
    ConvGraphlet::init(g());

    _conv->input(input(0));
    _conv->filter(input(1));
    _conv->bias(input(2));

    output()->from(_conv);
  }
};

class FailOfTestI3OGraph : public TestIsGraphlet<3>, public TestOGraphlet, public ConvGraphlet
{
public:
  FailOfTestI3OGraph() = default;

public:
  void init(void)
  {
    TestIsGraphlet<3>::init(g(), {{2, 3, 3, 4}, {1, 1}});
    TestOsGraphlet<1>::init(g(), {{2, 3, 3, 4}});
    ConvGraphlet::init(g());

    _conv->input(input(0));
    _conv->filter(input(1));
    _conv->bias(input(2));

    output()->from(_conv);
  }
};

} // namespace

TEST(TestIOGraphTest, IOGraph_init)
{
  TestOfTestIOGraph tg;
  tg.init();

  SUCCEED();
}

TEST(TestIOGraphTest, I2OGraph_init)
{
  TestOfTestI2OGraph tg;
  tg.init();

  SUCCEED();
}

TEST(TestIOGraphTest, I3OGraph_init)
{
  TestOfTestI3OGraph tg;
  tg.init();

  SUCCEED();
}

TEST(TestIOGraphTest, I3OGraph_input_number_mismatch_NEG)
{
  FailOfTestI3OGraph fg;
  EXPECT_THROW(fg.init(), std::runtime_error);
}
