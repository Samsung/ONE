/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "luci/Pass/RemoveUnnecessaryMulDivPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class MulDivGraphlet
{
public:
  MulDivGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape, bool fill_with_ones, bool activation)
  {
    // one Create.
    _one = g->nodes()->create<luci::CircleConst>();
    _one->rank(1);
    _one->dim(0).set(input_shape.size());
    _one->shape_status(luci::ShapeStatus::VALID);
    _one->dtype(loco::DataType::FLOAT32);
    _one->size<loco::DataType::FLOAT32>(input_shape.size());
    for (int i = 0; i < input_shape.size(); ++i)
      _one->at<loco::DataType::FLOAT32>(i) = fill_with_ones ? 1.0f : 0.0f;
    _one->name("one");

    // Div Create.
    _div = g->nodes()->create<luci::CircleDiv>();
    _div->y(_one);
    if (activation)
    {
      _div->fusedActivationFunction(luci::FusedActFunc::RELU);
    }
    else
    {
      _div->fusedActivationFunction(luci::FusedActFunc::NONE);
    }
    _div->dtype(loco::DataType::FLOAT32);
    _div->shape(input_shape);
    _div->name("div");

    // Mul Create.
    _mul = g->nodes()->create<luci::CircleMul>();
    _mul->y(_one);
    if (activation)
    {
      _mul->fusedActivationFunction(luci::FusedActFunc::RELU);
    }
    else
    {
      _mul->fusedActivationFunction(luci::FusedActFunc::NONE);
    }
    _mul->dtype(loco::DataType::FLOAT32);
    _mul->shape(input_shape);
    _mul->name("mul");
  }

protected:
  luci::CircleDiv *_div = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_one = nullptr;
};

class DivGraph : public TestIOGraph, public MulDivGraphlet
{
public:
  DivGraph() = default;

public:
  void init(const ShapeU32 shape, bool fill_with_ones, bool activation)
  {
    TestIOGraph::init(shape, shape);
    MulDivGraphlet::init(g(), shape, fill_with_ones, activation);

    _div->x(input());
    output()->from(_div);
  }
};

class MulGraph : public TestIOGraph, public MulDivGraphlet
{
public:
  MulGraph() = default;

public:
  void init(const ShapeU32 shape, bool fill_with_ones, bool activation)
  {
    TestIOGraph::init(shape, shape);
    MulDivGraphlet::init(g(), shape, fill_with_ones, activation);

    _mul->x(input());
    output()->from(_mul);
  }
};

} // namespace

TEST(RemoveUnnecessaryDivPass, name_test)
{
  luci::RemoveUnnecessaryDivPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveUnnecessaryMulPass, name_test)
{
  luci::RemoveUnnecessaryMulPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveUnnecessaryDivPass, simple_div_test)
{
  luci::RemoveUnnecessaryDivPass pass;

  DivGraph g;
  g.init({1, 14, 21, 32}, true, false);

  ASSERT_TRUE(pass.run(g.g()));

  // check Div is removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto div = dynamic_cast<luci::CircleDiv *>(node))
      count++;
  }
  ASSERT_EQ(0, count);
}

TEST(RemoveUnnecessaryMulPass, simple_mul_test)
{
  luci::RemoveUnnecessaryMulPass pass;

  MulGraph g;
  g.init({1, 14, 21, 32}, true, false);

  ASSERT_TRUE(pass.run(g.g()));

  // check Mul is removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto mul = dynamic_cast<luci::CircleMul *>(node))
      count++;
  }
  ASSERT_EQ(0, count);
}

TEST(RemoveUnnecessaryDivPass, div_not_removed_NEG)
{
  luci::RemoveUnnecessaryDivPass pass;
  DivGraph g;
  g.init({1, 14, 21, 32}, false, false);

  ASSERT_FALSE(pass.run(g.g()));

  // check Div is not removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto div = dynamic_cast<luci::CircleDiv *>(node))
      count++;
  }
  ASSERT_EQ(1, count);
}

TEST(RemoveUnnecessaryMulPass, mul_not_removed_NEG)
{
  luci::RemoveUnnecessaryMulPass pass;
  MulGraph g;
  g.init({1, 14, 21, 32}, false, false);

  ASSERT_FALSE(pass.run(g.g()));

  // check Mul is not removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto mul = dynamic_cast<luci::CircleMul *>(node))
      count++;
  }
  ASSERT_EQ(1, count);
}

TEST(RemoveUnnecessaryDivPass, div_activation_blocks_removal_NEG)
{
  luci::RemoveUnnecessaryDivPass pass;
  DivGraph g;
  g.init({1, 14, 21, 32}, true, true);

  ASSERT_FALSE(pass.run(g.g()));

  // check Div is not removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto div = dynamic_cast<luci::CircleDiv *>(node))
      count++;
  }
  ASSERT_EQ(1, count);
}

TEST(RemoveUnnecessaryMulPass, mul_activation_blocks_removal_NEG)
{
  luci::RemoveUnnecessaryMulPass pass;
  MulGraph g;
  g.init({1, 14, 21, 32}, true, true);

  ASSERT_FALSE(pass.run(g.g()));

  // check Mul is not removed
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    if (auto mul = dynamic_cast<luci::CircleMul *>(node))
      count++;
  }
  ASSERT_EQ(1, count);
}
