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

#include "luci/Pass/RemoveRedundantDequantizePass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class DequantizeGraphlet
{
public:
  DequantizeGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    _dequantize = g->nodes()->create<luci::CircleDequantize>();
    _dequantize->dtype(loco::DataType::FLOAT32);
    _dequantize->name("dequantize");
  }

protected:
  luci::CircleDequantize *_dequantize = nullptr;
};

class RedundantDequantizeGraph : public TestIOGraph, public DequantizeGraphlet
{
public:
  RedundantDequantizeGraph() = default;

public:
  void init(void)
  {
    TestIOGraph::init({1}, {1});
    DequantizeGraphlet::init(g());

    _dequantize->input(input());

    output()->from(_dequantize);
  }

  void init_u8_input(void)
  {
    TestIOGraph::init({1}, {1});
    DequantizeGraphlet::init(g());

    // Use u8 input (dequantize is not redundant anymore)
    input()->dtype(loco::DataType::U8);
    {
      auto qparam = std::make_unique<luci::CircleQuantParam>();
      qparam->scale = {1};
      qparam->zerop = {1};
      input()->quantparam(std::move(qparam));
    }

    _dequantize->input(input());

    output()->from(_dequantize);
  }
};

} // namespace

TEST(RemoveRedundantDequantizePass, single_redundant_dequantize)
{
  RedundantDequantizeGraph g;
  luci::RemoveRedundantDequantizePass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));

  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (dynamic_cast<luci::CircleDequantize *>(node))
    {
      count++;
    }
  }

  ASSERT_EQ(0, count);
}

TEST(RemoveRedundantDequantizePass, wrong_dtype_NEG)
{
  RedundantDequantizeGraph g;
  luci::RemoveRedundantDequantizePass pass;

  g.init_u8_input();

  EXPECT_FALSE(pass.run(g.g()));
}
