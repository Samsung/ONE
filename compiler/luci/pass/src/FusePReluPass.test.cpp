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

#include "luci/Pass/FusePReluPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class PReluGraphlet
{
public:
  PReluGraphlet() = default;

  void init(loco::Graph *g)
  {
    _abs = g->nodes()->create<luci::CircleAbs>();
    _sub = g->nodes()->create<luci::CircleSub>();
    _mul_alpha = g->nodes()->create<luci::CircleMul>();
    _mul_half = g->nodes()->create<luci::CircleMul>();
    _relu = g->nodes()->create<luci::CircleRelu>();
    _add = g->nodes()->create<luci::CircleAdd>();
    _const_alpha = g->nodes()->create<luci::CircleConst>();
    _const_half = g->nodes()->create<luci::CircleConst>();

    _sub->fusedActivationFunction(luci::FusedActFunc::NONE);
    _mul_alpha->fusedActivationFunction(luci::FusedActFunc::NONE);
    _mul_half->fusedActivationFunction(luci::FusedActFunc::NONE);
    _add->fusedActivationFunction(luci::FusedActFunc::NONE);

    _abs->name("abs");
    _sub->name("sub");
    _mul_alpha->name("mul_alpha");
    _mul_half->name("mul_half");
    _relu->name("relu");
    _add->name("add");
    _const_alpha->name("const_alpha");
    _const_half->name("const_half");

    _const_alpha->dtype(loco::DataType::FLOAT32);
    _const_alpha->size<loco::DataType::FLOAT32>(1);
    _const_alpha->shape({1});
    _const_alpha->at<loco::DataType::FLOAT32>(0) = 0.1;
    _const_alpha->shape_status(luci::ShapeStatus::VALID);

    _const_half->dtype(loco::DataType::FLOAT32);
    _const_half->size<loco::DataType::FLOAT32>(1);
    _const_half->shape({1});
    _const_half->at<loco::DataType::FLOAT32>(0) = 0.5;
    _const_half->shape_status(luci::ShapeStatus::VALID);
  }

  void invalid_half() { _const_half->at<loco::DataType::FLOAT32>(0) = 0.1; }

protected:
  luci::CircleAbs *_abs = nullptr;
  luci::CircleSub *_sub = nullptr;
  luci::CircleMul *_mul_alpha = nullptr;
  luci::CircleMul *_mul_half = nullptr;
  luci::CircleRelu *_relu = nullptr;
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_const_alpha = nullptr;
  luci::CircleConst *_const_half = nullptr;
};

class FusePReluTestGraph : public TestIOGraph, public PReluGraphlet
{
public:
  FusePReluTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    PReluGraphlet::init(g());

    _relu->features(input());
    _abs->x(input());
    _sub->x(input());
    _sub->y(_abs);
    _mul_alpha->x(_sub);
    _mul_alpha->y(_const_alpha);
    _mul_half->x(_mul_alpha);
    _mul_half->y(_const_half);
    _add->x(_relu);
    _add->y(_mul_half);

    output()->from(_add);
  }
};

class FusePReluTestNegGraph : public TestIOGraph, public PReluGraphlet
{
public:
  FusePReluTestNegGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    PReluGraphlet::init(g());

    _relu->features(input());
    _abs->x(input());
    // NOTE x and y are incorrect
    _sub->x(_abs);
    _sub->y(input());
    _mul_alpha->x(_sub);
    _mul_alpha->y(_const_alpha);
    _mul_half->x(_mul_alpha);
    _mul_half->y(_const_half);
    _add->x(_relu);
    _add->y(_mul_half);

    output()->from(_add);
  }
};

} // namespace

TEST(FusePReluPassTest, name)
{
  luci::FusePReluPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FusePReluPassTest, fuse)
{
  FusePReluTestGraph g;
  luci::FusePReluPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FusePReluPassTest, fuse_invalid_half_NEG)
{
  FusePReluTestNegGraph g;
  luci::FusePReluPass pass;

  g.init();
  g.invalid_half();

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(FusePReluPassTest, fuse_NEG)
{
  FusePReluTestNegGraph g;
  luci::FusePReluPass pass;

  g.init();

  EXPECT_FALSE(pass.run(g.g()));
}
