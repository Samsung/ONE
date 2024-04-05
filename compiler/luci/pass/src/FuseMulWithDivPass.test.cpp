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

#include "luci/Pass/FuseMulWithDivPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class PatternMulDivGraphlet
{
public:
  PatternMulDivGraphlet() = default;

  void init(loco::Graph *g)
  {
    _mul = g->nodes()->create<luci::CircleMul>();
    _mul->fusedActivationFunction(luci::FusedActFunc::NONE);
    _mul_const = g->nodes()->create<luci::CircleConst>();
    _mul_const->rank(1);
    _mul_const->dtype(loco::DataType::FLOAT32);
    _mul_const->size<loco::DataType::FLOAT32>(1);
    _mul_const->at<loco::DataType::FLOAT32>(0) = 1.1f;
    _mul_const->shape_status(luci::ShapeStatus::VALID);

    _div = g->nodes()->create<luci::CircleDiv>();
    _div->fusedActivationFunction(luci::FusedActFunc::NONE);
    _div_const = g->nodes()->create<luci::CircleConst>();
    _div_const->rank(1);
    _div_const->dtype(loco::DataType::FLOAT32);
    _div_const->size<loco::DataType::FLOAT32>(1);
    _div_const->at<loco::DataType::FLOAT32>(0) = 2.2f;
    _div_const->shape_status(luci::ShapeStatus::VALID);

    _mul->name("_mul");
    _mul_const->name("_mul_const");

    _div->name("_div");
    _div_const->name("_div_const");
  }

public:
  luci::CircleMul *mul() { return _mul; }
  luci::CircleConst *mul_const() { return _mul_const; }
  luci::CircleDiv *div() { return _div; }
  luci::CircleConst *div_const() { return _div_const; }

protected:
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_mul_const = nullptr;
  luci::CircleDiv *_div = nullptr;
  luci::CircleConst *_div_const = nullptr;
};

/**
 *  Simple graph for test
 *
 *  BEFORE
 *              [Input]
 *                 |
 *          [Mul, MUL_Scalar_Const]
 *                 |
 *          [Div, DIV_Scalar_Const]
 *                 |
 *             [Output]
 *
 *  AFTER
 *              [Input]
 *                 |
 *          [Div, Scalar_Const_new]
 *                 |
 *             [Output]
 *
 *  WHERE: Scalar_Const_new = DIV_Scalar_Const / MUL_Scalar_Const
 */
class FuseMulDivPatternTestGraph : public TestIOGraph, public PatternMulDivGraphlet
{
public:
  FuseMulDivPatternTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 2, 3}, {1, 2, 3});
    PatternMulDivGraphlet::init(g());

    _mul->x(input());
    _mul->y(_mul_const);

    _div->x(_div_const);
    _div->y(_mul);

    output()->from(_div);
  }
};

class FuseMulDivToMulPatternTestGraph : public TestIOGraph, public PatternMulDivGraphlet
{
public:
  FuseMulDivToMulPatternTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 2, 3}, {1, 2, 3});
    PatternMulDivGraphlet::init(g());

    _mul->x(input());
    _mul->y(_mul_const);

    _div->x(_mul);
    _div->y(_div_const);

    output()->from(_div);
  }
};

} // namespace

TEST(FuseMulWithDivPassTest, fus_mul_div_pattern)
{
  FuseMulDivPatternTestGraph g;
  luci::FuseMulWithDivPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseMulWithDivPassTest, fuse_mul_div_NEG)
{
  FuseMulDivPatternTestGraph g;
  luci::FuseMulWithDivPass pass;

  g.init();

  // Add CircleRelu operation between CircleMean and Mul operations
  auto relu = g.g()->nodes()->create<luci::CircleRelu>();
  relu->name("relu");
  relu->features(g.mul());
  g.div()->y(relu);

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(FuseMulWithDivPassTest, fuse_mul_div_to_mul_pattern)
{
  FuseMulDivToMulPatternTestGraph g;
  luci::FuseMulWithDivPass pass;

  g.init();

  auto div = dynamic_cast<luci::CircleDiv *>(g.output()->from());
  EXPECT_NE(div, nullptr);

  EXPECT_TRUE(pass.run(g.g()));

  auto mul = dynamic_cast<luci::CircleMul *>(g.output()->from());
  EXPECT_NE(mul, nullptr);
}

TEST(FuseMulWithDivPassTest, fuse_mul_div_to_mul_NEG)
{
  FuseMulDivToMulPatternTestGraph g;
  luci::FuseMulWithDivPass pass;

  g.init();

  // Add CircleRelu operation between CircleMul and Div operations
  auto relu = g.g()->nodes()->create<luci::CircleRelu>();
  relu->name("relu");
  relu->features(g.mul());
  g.div()->x(relu);

  EXPECT_FALSE(pass.run(g.g()));
}
