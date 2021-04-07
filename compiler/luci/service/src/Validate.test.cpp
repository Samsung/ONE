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

#include "luci/Service/Validate.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/Nodes/CircleAdd.h>
#include <luci/IR/Nodes/CircleSqrt.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class SqrtGraphlet
{
public:
  SqrtGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape)
  {
    _sqrt = g->nodes()->create<luci::CircleSqrt>();
    _sqrt->dtype(loco::DataType::S32);
    _sqrt->name("sqrt");
  }

protected:
  luci::CircleSqrt *_sqrt = nullptr;
};

class SqrtGraph : public TestIOGraph, public SqrtGraphlet
{
public:
  SqrtGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIOGraph::init(shape, shape);
    SqrtGraphlet::init(g(), shape);

    _sqrt->x(input());

    output()->from(_sqrt);

    // set output name to _sqrt: CircleOutput may have duplicate name
    output()->name(_sqrt->name());
  }
};

class Sqrt2xGraphlet
{
public:
  Sqrt2xGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape)
  {
    _sqrt1 = g->nodes()->create<luci::CircleSqrt>();
    _sqrt1->dtype(loco::DataType::S32);
    _sqrt1->name("sqrt");

    _sqrt2 = g->nodes()->create<luci::CircleSqrt>();
    _sqrt2->dtype(loco::DataType::S32);
    _sqrt2->name("sqrt");
  }

protected:
  luci::CircleSqrt *_sqrt1 = nullptr;
  luci::CircleSqrt *_sqrt2 = nullptr;
};

class Sqrt2xGraph : public TestIOGraph, public Sqrt2xGraphlet
{
public:
  Sqrt2xGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIOGraph::init(shape, shape);
    Sqrt2xGraphlet::init(g(), shape);

    _sqrt1->x(input());

    _sqrt2->x(_sqrt1);

    output()->from(_sqrt2);
  }
};

} // namespace

TEST(ValidateTest, non_empty_name)
{
  SqrtGraph g;
  g.init({3, 3});

  ASSERT_TRUE(luci::validate_name(g.g()));
}

TEST(ValidateTest, unique_name)
{
  luci::Module module;

  SqrtGraph g;
  g.init({3, 3});
  g.transfer_to(&module);

  ASSERT_TRUE(luci::validate_unique_name(&module));
}

TEST(ValidateTest, unique_name_NEG)
{
  luci::Module module;

  Sqrt2xGraph g;
  g.init({3, 3});
  g.transfer_to(&module);

  ASSERT_FALSE(luci::validate_unique_name(&module));
}
