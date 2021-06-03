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

#include "luci/Pass/RemoveQuantDequantSeqPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class QuantDequantGraphlet
{
public:
  QuantDequantGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    _qu = g->nodes()->create<luci::CircleQuantize>();
    _qu->name("qu");

    _de = g->nodes()->create<luci::CircleDequantize>();
    _de->name("de");
  }

protected:
  luci::CircleQuantize *_qu = nullptr;
  luci::CircleDequantize *_de = nullptr;
};

class QuantDequantGraph : public TestIOGraph, public QuantDequantGraphlet
{
public:
  QuantDequantGraph() = default;

public:
  void init(void)
  {
    TestIOGraph::init({1}, {1});
    QuantDequantGraphlet::init(g());

    _qu->input(input());
    _de->input(_qu);

    output()->from(_de);
  }
};

} // namespace

TEST(RemoveQuantDequantSeqPass, name)
{
  luci::RemoveQuantDequantSeqPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveQuantDequantSeqPass, remove_quantdequant)
{
  QuantDequantGraph g;
  luci::RemoveQuantDequantSeqPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));

  auto *node1 = loco::must_cast<luci::CircleNode *>(g.output()->from());
  auto *node2 = loco::must_cast<luci::CircleNode *>(g.input());
  EXPECT_EQ(node1, node2);
}
