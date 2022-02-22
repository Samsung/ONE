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

#include "ConnectNode.h"

#include "ConnectNode.test.h"

#include <luci/Service/CircleNodeClone.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class NodeGraphlet : public NodeIsOsGraphletT<luci::CircleWhile>
{
public:
  NodeGraphlet() = default;

public:
  void init(loco::Graph *g, uint32_t n, uint32_t m) override { NodeIsOsGraphletT::init(g, n, m); }
};

class TestNodeGraph : public TestIsOsGraph<1, 1>, public NodeGraphlet
{
public:
  TestNodeGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIsOsGraph<1, 1>::init({shape}, {shape});
    NodeGraphlet::init(g(), 1, 1);

    node()->input(0, input(0));

    output(0)->from(node());
  }
};

} // namespace

TEST(ConnectNodeTest, connect_While)
{
  TestNodeGraph tng;
  tng.init({1});

  ConnectionTestHelper cth;
  cth.prepare_inputs<1, 1>(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleWhile *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  ASSERT_NO_THROW(loco::must_cast<luci::CircleWhile *>(clone));

  cth.clone_connect(node, clone);

  ASSERT_EQ(1, clone->arity());
  ASSERT_EQ(cth.inputs(0), clone->arg(0));
}

TEST(ConnectNodeTest, connect_While_NEG)
{
  TestNodeGraph tng;
  tng.init({1});

  ConnectionTestHelper cth;
  cth.prepare_inputs_miss<1, 1>(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleWhile *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  ASSERT_NO_THROW(loco::must_cast<luci::CircleWhile *>(clone));

  EXPECT_ANY_THROW(cth.clone_connect(node, clone));
}
