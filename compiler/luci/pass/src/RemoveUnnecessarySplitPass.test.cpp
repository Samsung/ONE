/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveUnnecessarySplitPass.h"

#include <luci/IR/CircleNodes.h>

#include "test/TestIOGraph.h"
#include "test/TestFirstNode.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class SplitGraphlet
{
public:
  SplitGraphlet() = default;

public:
  void init(loco::Graph *g, uint32_t nout)
  {
    assert(nout == 1 || nout == 2);

    _dim = g->nodes()->create<luci::CircleConst>();
    set_shape_vector(_dim, {0});
    _dim->name("dim");

    _split = g->nodes()->create<luci::CircleSplit>();
    _split->num_split(nout);
    _split->name("split");

    _split_out_0 = g->nodes()->create<luci::CircleSplitOut>();
    _split_out_0->index(0);
    _split_out_0->name("split_out_0");

    if (nout == 2)
    {
      _split_out_1 = g->nodes()->create<luci::CircleSplitOut>();
      _split_out_1->index(1);
      _split_out_1->name("split_out_1");
    }
  }

protected:
  luci::CircleSplit *_split = nullptr;
  luci::CircleConst *_dim = nullptr;
  luci::CircleSplitOut *_split_out_0 = nullptr;
  luci::CircleSplitOut *_split_out_1 = nullptr;
};

class SplitOneGraph : public TestIGraphlet, public TestOGraphlet, public SplitGraphlet
{
public:
  SplitOneGraph() = default;

public:
  void init()
  {
    TestIGraphlet::init(g(), {1});
    TestOGraphlet::init(g(), {1});
    SplitGraphlet::init(g(), 1);

    _split->input(input());
    _split->split_dim(_dim);
    _split_out_0->input(_split);

    output()->from(_split_out_0);
  }
};

class SplitTwoGraph : public TestIGraphlet, public TestOsGraphlet<2>, public SplitGraphlet
{
public:
  SplitTwoGraph() = default;

public:
  void init()
  {
    TestIGraphlet::init(g(), {1});
    TestOsGraphlet<2>::init(g(), {1});
    SplitGraphlet::init(g(), 2);

    _split->input(input());
    _split->split_dim(_dim);
    _split_out_0->input(_split);
    _split_out_1->input(_split);

    output(0)->from(_split_out_0);
    output(1)->from(_split_out_1);
  }
};

// TODO use ::testing::Test

} // namespace

TEST(RemoveUnnecessarySplitPass, name)
{
  luci::RemoveUnnecessarySplitPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveUnnecessarySplitPass, create_unnecessary_split)
{
  SplitOneGraph g;

  g.init();

  luci::RemoveUnnecessarySplitPass pass;
  while (pass.run(g.g()))
    ;

  auto split_node = luci::test::first_node<luci::CircleSplit>(g.g());
  // No Split node is in graph.
  ASSERT_EQ(nullptr, split_node);
}

TEST(RemoveUnnecessarySplitPass, create_unnecessary_split_NEG)
{
  SplitTwoGraph g;

  g.init();

  luci::RemoveUnnecessarySplitPass pass;
  while (pass.run(g.g()))
    ;

  auto split_node = luci::test::first_node<luci::CircleSplit>(g.g());
  // Split node is in graph.
  ASSERT_NE(nullptr, split_node);
}
