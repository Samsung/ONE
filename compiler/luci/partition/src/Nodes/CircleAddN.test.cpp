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

class NodeGraphlet : public NodeNGraphletT<luci::CircleAddN>
{
public:
  NodeGraphlet() = default;
};

class TestNodeGraph : public TestIsOGraph<3>, public NodeGraphlet
{
public:
  TestNodeGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIsOGraph<3>::init({shape, shape, shape}, shape);
    NodeGraphlet::init(g(), 3);

    for (uint32_t i = 0; i < 3; ++i)
    {
      node()->inputs(i, input(i));
    }

    output()->from(node());
  }
};

} // namespace

TEST(ConnectNodeTest, connect_AddN)
{
  TestNodeGraph tng;
  tng.init({2, 3});

  ConnectionTestHelper cth;
  cth.prepare_inputs(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleAddN *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  cth.clone_connect(node, clone);

  ASSERT_EQ(3, clone->arity());
  for (uint32_t i = 0; i < 3; ++i)
  {
    ASSERT_EQ(cth.inputs(i), clone->arg(i));
  }
}
