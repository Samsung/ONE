/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "luci/Pass/RemoveUnnecessaryAddPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class AddGraphlet
{
public:
  AddGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape, bool fill_with_zeros, bool activation)
  {
    // zero Create.
    _zero = g->nodes()->create<luci::CircleConst>();
    _zero->rank(1);
    _zero->dim(0).set(input_shape.size());
    _zero->shape_status(luci::ShapeStatus::VALID);
    _zero->dtype(loco::DataType::FLOAT32);
    _zero->size<loco::DataType::FLOAT32>(input_shape.size());
    for (int i = 0; i < input_shape.size(); ++i)
      _zero->at<loco::DataType::FLOAT32>(i) = fill_with_zeros ? 0 : 1;
    _zero->name("begin");

    // Add Create.
    _add = g->nodes()->create<luci::CircleAdd>();
    _add->y(_zero);
    if (activation)
    {
      _add->fusedActivationFunction(luci::FusedActFunc::RELU);
    }
    else
    {
      _add->fusedActivationFunction(luci::FusedActFunc::NONE);
    }
    _add->dtype(loco::DataType::FLOAT32);
    _add->shape(input_shape);
    _add->name("add");
  }

protected:
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_zero = nullptr;
};

class AddGraph : public TestIOGraph, public AddGraphlet
{
public:
  AddGraph() = default;

public:
  void init(const ShapeU32 shape, bool fill_with_zeros, bool activation)
  {
    TestIOGraph::init(shape, shape);
    AddGraphlet::init(g(), shape, fill_with_zeros, activation);

    _add->x(input());
    output()->from(_add);
  }
};

} // namespace

TEST(RemoveUnnecessaryAddPass, name_test)
{
  luci::RemoveUnnecessaryAddPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveUnnecessaryAddPass, simple_test)
{
  luci::RemoveUnnecessaryAddPass pass;

  AddGraph g;
  g.init({1, 14, 21, 32}, true, false);

  ASSERT_TRUE(pass.run(g.g()));

  // check Add is removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto add = dynamic_cast<luci::CircleAdd *>(node))
      count++;
  }
  ASSERT_EQ(0, count);
}

TEST(RemoveUnnecessaryAddPass, not_removed_NEG)
{
  luci::RemoveUnnecessaryAddPass pass;
  AddGraph g;
  g.init({1, 14, 21, 32}, false, false);

  ASSERT_FALSE(pass.run(g.g()));

  // check Add is not removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto add = dynamic_cast<luci::CircleAdd *>(node))
      count++;
  }
  ASSERT_EQ(1, count);
}

TEST(RemoveUnnecessaryAddPass, activation_blocks_removal_NEG)
{
  luci::RemoveUnnecessaryAddPass pass;
  AddGraph g;
  g.init({1, 14, 21, 32}, true, true);

  ASSERT_FALSE(pass.run(g.g()));

  // check Add is not removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto add = dynamic_cast<luci::CircleAdd *>(node))
      count++;
  }
  ASSERT_EQ(1, count);
}
