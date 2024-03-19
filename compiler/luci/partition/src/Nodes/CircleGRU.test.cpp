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

#include "luci/ConnectNode.h"

#include "ConnectNode.test.h"

#include <luci/Service/CircleNodeClone.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class NodeGraphlet : public NodeGraphletT<luci::CircleGRU>
{
public:
  NodeGraphlet() = default;

public:
  void init(loco::Graph *g) override
  {
    NodeGraphletT<luci::CircleGRU>::init(g);

    _node->fusedActivationFunction(luci::FusedActFunc::NONE);
  }
};

class TestNodeGraph : public TestIsOGraph<6>, public NodeGraphlet
{
public:
  TestNodeGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIsOGraph<6>::init({shape, shape, shape, shape, shape, shape}, shape);
    NodeGraphlet::init(g());

    node()->input(input(0));
    node()->hidden_hidden(input(1));
    node()->hidden_hidden_bias(input(2));
    node()->hidden_input(input(3));
    node()->hidden_input_bias(input(4));
    node()->state(input(5));

    output()->from(node());
  }
};

} // namespace

TEST(ConnectNodeTest, connect_CIRCLE_GRU)
{
  TestNodeGraph tng;
  tng.init({10, 1, 4});

  ConnectionTestHelper cth;
  cth.prepare_inputs(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleGRU *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  ASSERT_NO_THROW(loco::must_cast<luci::CircleGRU *>(clone));

  cth.clone_connect(node, clone);

  ASSERT_EQ(6, clone->arity());
  // 24 separate checks is too much
  for (uint32_t i = 0; i < 6; ++i)
    ASSERT_EQ(cth.inputs(i), clone->arg(i));
}

TEST(ConnectNodeTest, connect_CIRCLE_GRU_NEG)
{
  TestNodeGraph tng;
  tng.init({10, 1, 4});

  ConnectionTestHelper cth;
  cth.prepare_inputs_miss(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleGRU *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  ASSERT_NO_THROW(loco::must_cast<luci::CircleGRU *>(clone));

  EXPECT_ANY_THROW(cth.clone_connect(node, clone));
}
