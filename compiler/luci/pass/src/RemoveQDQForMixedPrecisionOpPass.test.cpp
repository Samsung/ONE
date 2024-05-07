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

#include "luci/Pass/RemoveQDQForMixedPrecisionOpPass.h"

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
    _fc = g->nodes()->create<luci::CircleFullyConnected>();
    _fc->name("fc");

    _qu = g->nodes()->create<luci::CircleQuantize>();
    _qu->name("qu");

    _de = g->nodes()->create<luci::CircleDequantize>();
    _de->name("de");

    _qu_2 = g->nodes()->create<luci::CircleQuantize>();
    _qu_2->name("qu");

    _de_2 = g->nodes()->create<luci::CircleDequantize>();
    _de_2->name("de");
  }

public:
  luci::CircleFullyConnected *fc(void) { return _fc; }
  luci::CircleQuantize *qu(void) { return _qu; }
  luci::CircleQuantize *qu_2(void) { return _qu_2; }

protected:
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleQuantize *_qu = nullptr;
  luci::CircleDequantize *_de = nullptr;
  luci::CircleQuantize *_qu_2 = nullptr;
  luci::CircleDequantize *_de_2 = nullptr;
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

    _fc->input(input());
    _qu->input(_fc);
    _de->input(_qu);
    _qu_2->input(_de);
    _de_2->input(_qu_2);

    output()->from(_de_2);
  }
};

} // namespace

TEST(RemoveQDQForMixedPrecisionOpPass, remove_qdq_FC)
{
  QuantDequantGraph g;
  luci::RemoveQDQForMixedPrecisionOpPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));

  EXPECT_EQ(g.fc(), g.qu_2()->input());
}

TEST(RemoveQDQForMixedPrecisionOpPass, remove_qdq_wrong_op_NEG)
{
  QuantDequantGraph g;
  luci::RemoveQDQForMixedPrecisionOpPass pass;

  g.init();

  g.qu()->input(g.input());

  EXPECT_FALSE(pass.run(g.g()));
}
