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

#include <luci/IR/Nodes/CircleAdd.h>
#include <luci/Service/CircleNodeClone.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class AddGraphlet
{
public:
  AddGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape)
  {
    _add = g->nodes()->create<luci::CircleAdd>();
    _add->dtype(loco::DataType::S32);
    _add->name("add");
    _add->fusedActivationFunction(luci::FusedActFunc::RELU);
  }

  luci::CircleAdd *add(void) const { return _add; }

protected:
  luci::CircleAdd *_add = nullptr;
};

class AddGraph : public TestI2OGraph, public AddGraphlet
{
public:
  AddGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestI2OGraph::init(shape, shape);
    AddGraphlet::init(g(), shape);

    add()->x(input(0));
    add()->y(input(1));

    output()->from(add());
  }
};

} // namespace

TEST(ConnectNodeTest, connect_Add)
{
  AddGraph ag;
  ag.init({2, 3});

  ConnectionTestHelper cth;
  cth.prepare_inputs(&ag);

  auto *node = ag.add();
  auto *clone = luci::clone_node(node, cth.graph_c());
  cth.clone_connect(node, clone);

  ASSERT_EQ(2, clone->arity());
  ASSERT_EQ(cth.inputs(0), clone->arg(0));
  ASSERT_EQ(cth.inputs(1), clone->arg(1));
}
