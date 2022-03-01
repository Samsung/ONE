/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveRedundantQuantizePass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class QuantizeGraphlet
{
public:
  QuantizeGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    _first_quantize = g->nodes()->create<luci::CircleQuantize>();
    _first_quantize->name("first_quantize");

    _second_quantize = g->nodes()->create<luci::CircleQuantize>();
    _second_quantize->name("second_quantize");
  }

protected:
  luci::CircleQuantize *_first_quantize = nullptr;
  luci::CircleQuantize *_second_quantize = nullptr;
};

class QuantizeGraph : public TestIOGraph, public QuantizeGraphlet
{
public:
  QuantizeGraph() = default;

public:
  void init(void)
  {
    TestIOGraph::init({1}, {1});
    QuantizeGraphlet::init(g());

    input()->dtype(loco::DataType::FLOAT32);

    _first_quantize->input(input());
    _second_quantize->input(_first_quantize);

    output()->from(_second_quantize);
    output()->dtype(loco::DataType::FLOAT32);
  }
};

} // namespace

TEST(RemoveRedundantQuantizePass, name)
{
  luci::RemoveRedundantQuantizePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveRedundantQuantizePass, remove_redundant_quantize)
{
  QuantizeGraph g;
  luci::RemoveRedundantQuantizePass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));

  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (dynamic_cast<luci::CircleQuantize *>(node))
    {
      count++;
    }
  }

  ASSERT_EQ(1, count);
}
