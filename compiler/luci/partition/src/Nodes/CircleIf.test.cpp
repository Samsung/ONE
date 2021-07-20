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

class NodeGraphlet : public NodeIsOsGraphletT<luci::CircleIf>
{
public:
  NodeGraphlet() = default;

public:
  void init(loco::Graph *g, uint32_t n, uint32_t m) override
  {
    // cond() will take one input
    NodeIsOsGraphletT::init(g, n - 1, m);
  }
};

class TestNodeGraph : public TestIsOsGraph<3, 1>, public NodeGraphlet
{
public:
  TestNodeGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIsOsGraph<3, 1>::init({shape, shape, shape}, {shape});
    NodeGraphlet::init(g(), 3, 1);

    node()->cond(input(0));
    node()->input(0, input(1));
    node()->input(1, input(2));

    output(0)->from(node());
  }
};

} // namespace

TEST(ConnectNodeTest, connect_If)
{
  TestNodeGraph tng;
  tng.init({2, 3});

  ConnectionTestHelper cth;
  cth.prepare_inputs<3, 1>(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleIf *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  ASSERT_NO_THROW(loco::must_cast<luci::CircleIf *>(clone));

  cth.clone_connect(node, clone);

  // aritiy(3) = cond + input(2)
  ASSERT_EQ(3, clone->arity());
  ASSERT_EQ(cth.inputs(0), clone->arg(0));
  ASSERT_EQ(cth.inputs(1), clone->arg(1));
  ASSERT_EQ(cth.inputs(2), clone->arg(2));
}

TEST(ConnectNodeTest, connect_If_NEG)
{
  TestNodeGraph tng;
  tng.init({2, 3});

  ConnectionTestHelper cth;
  cth.prepare_inputs_miss<3, 1>(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleIf *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  ASSERT_NO_THROW(loco::must_cast<luci::CircleIf *>(clone));

  EXPECT_ANY_THROW(cth.clone_connect(node, clone));
}
