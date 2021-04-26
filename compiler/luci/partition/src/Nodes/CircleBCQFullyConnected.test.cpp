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

class NodeGraphlet : public NodeGraphletT<luci::CircleBCQFullyConnected>
{
public:
  NodeGraphlet() = default;

public:
  void init(loco::Graph *g) override
  {
    NodeGraphletT<luci::CircleBCQFullyConnected>::init(g);

    _node->fusedActivationFunction(luci::FusedActFunc::RELU);
  }
};

class TestNodeGraph : public TestIsOGraph<5>, public NodeGraphlet
{
public:
  TestNodeGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIsOGraph<5>::init({shape, shape, shape, shape, shape}, shape);
    NodeGraphlet::init(g());

    node()->input(input(0));
    node()->weights_scales(input(1));
    node()->weights_binary(input(2));
    node()->bias(input(3));
    node()->weights_clusters(input(4));

    output()->from(node());
  }
};

} // namespace

TEST(ConnectNodeTest, connect_BCQFullyConnected)
{
  TestNodeGraph tng;
  tng.init({2, 3});

  ConnectionTestHelper cth;
  cth.prepare_inputs(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleBCQFullyConnected *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  ASSERT_NO_THROW(loco::must_cast<luci::CircleBCQFullyConnected *>(clone));

  cth.clone_connect(node, clone);

  ASSERT_EQ(5, clone->arity());
  ASSERT_EQ(cth.inputs(0), clone->arg(0));
  ASSERT_EQ(cth.inputs(1), clone->arg(1));
  ASSERT_EQ(cth.inputs(2), clone->arg(2));
  ASSERT_EQ(cth.inputs(3), clone->arg(3));
  ASSERT_EQ(cth.inputs(4), clone->arg(4));
}

TEST(ConnectNodeTest, connect_BCQFullyConnected_NEG)
{
  TestNodeGraph tng;
  tng.init({2, 3});

  ConnectionTestHelper cth;
  cth.prepare_inputs_miss(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleBCQFullyConnected *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  ASSERT_NO_THROW(loco::must_cast<luci::CircleBCQFullyConnected *>(clone));

  EXPECT_ANY_THROW(cth.clone_connect(node, clone));
}
