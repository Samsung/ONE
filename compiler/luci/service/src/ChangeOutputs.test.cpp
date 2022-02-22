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

#include "luci/Service/ChangeOutputs.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/Nodes/CircleSqrt.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class Sqrt2xGraphlet
{
public:
  Sqrt2xGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape)
  {
    _sqrt1 = g->nodes()->create<luci::CircleSqrt>();
    _sqrt1->dtype(loco::DataType::S32);
    _sqrt1->name("sqrt1");

    _sqrt2 = g->nodes()->create<luci::CircleSqrt>();
    _sqrt2->dtype(loco::DataType::S32);
    _sqrt2->name("sqrt2");
  }

public:
  luci::CircleSqrt *sqrt1(void) const { return _sqrt1; }
  luci::CircleSqrt *sqrt2(void) const { return _sqrt2; }

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

TEST(ChangeOutputsTest, change)
{
  Sqrt2xGraph g;

  g.init({3, 3});

  {
    auto output = luci::output_node(g.g(), 0);
    ASSERT_EQ(g.sqrt2(), output->from());
  }

  std::vector<std::string> names{"sqrt1"};

  EXPECT_NO_THROW(luci::change_outputs(g.g(), names));

  {
    auto output = luci::output_node(g.g(), 0);
    ASSERT_EQ(g.sqrt1(), output->from());
  }
}

TEST(ChangeOutputsTest, name_not_found_NEG)
{
  Sqrt2xGraph g;

  g.init({3, 3});

  std::vector<std::string> names{"sqrt33"};

  EXPECT_ANY_THROW(luci::change_outputs(g.g(), names));
}

TEST(ChangeOutputsTest, number_names_NEG)
{
  Sqrt2xGraph g;

  g.init({3, 3});

  std::vector<std::string> names{"sqrt1", "sqrt2"};

  EXPECT_ANY_THROW(luci::change_outputs(g.g(), names));
}
