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

#include "luci/Pass/FuseGeluPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <cmath>
#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class GeluGraphlet
{
public:
  GeluGraphlet() = default;

  void init(loco::Graph *g)
  {
    _ifm = g->nodes()->create<luci::CircleAbs>();
    _mul_sqrt = g->nodes()->create<luci::CircleMul>();
    _erf = g->nodes()->create<luci::CircleCustom>(1, 1);
    _erf_out = g->nodes()->create<luci::CircleCustomOut>();
    _add_one = g->nodes()->create<luci::CircleAdd>();
    _mul = g->nodes()->create<luci::CircleMul>();
    _mul_half = g->nodes()->create<luci::CircleMul>();
    _const_sqrt = g->nodes()->create<luci::CircleConst>();
    _const_one = g->nodes()->create<luci::CircleConst>();
    _const_half = g->nodes()->create<luci::CircleConst>();

    _mul->fusedActivationFunction(luci::FusedActFunc::NONE);
    _mul_sqrt->fusedActivationFunction(luci::FusedActFunc::NONE);
    _mul_half->fusedActivationFunction(luci::FusedActFunc::NONE);
    _add_one->fusedActivationFunction(luci::FusedActFunc::NONE);

    _ifm->name("ifm");
    _mul_sqrt->name("mul_sqrt");
    _erf->name("erf");
    _erf_out->name("erf_out");
    _add_one->name("add_one");
    _mul->name("mul");
    _mul_half->name("mul_half");
    _const_one->name("const_one");
    _const_sqrt->name("const_sqrt");
    _const_half->name("const_half");

    _erf->custom_code("Erf");

    _const_sqrt->dtype(loco::DataType::FLOAT32);
    _const_sqrt->size<loco::DataType::FLOAT32>(1);
    _const_sqrt->shape({1});
    _const_sqrt->at<loco::DataType::FLOAT32>(0) = sqrtf(0.5f);
    _const_sqrt->shape_status(luci::ShapeStatus::VALID);

    _const_one->dtype(loco::DataType::FLOAT32);
    _const_one->size<loco::DataType::FLOAT32>(1);
    _const_one->shape({1});
    _const_one->at<loco::DataType::FLOAT32>(0) = 1.0;
    _const_one->shape_status(luci::ShapeStatus::VALID);

    _const_half->dtype(loco::DataType::FLOAT32);
    _const_half->size<loco::DataType::FLOAT32>(1);
    _const_half->shape({1});
    _const_half->at<loco::DataType::FLOAT32>(0) = 0.5;
    _const_half->shape_status(luci::ShapeStatus::VALID);
  }

  void invalid_half() { _const_half->at<loco::DataType::FLOAT32>(0) = 0.1; }
  void invalid_act() { _add_one->fusedActivationFunction(luci::FusedActFunc::RELU); }

protected:
  luci::CircleAbs *_ifm = nullptr;
  luci::CircleMul *_mul_sqrt = nullptr;
  luci::CircleCustom *_erf = nullptr;
  luci::CircleCustomOut *_erf_out = nullptr;
  luci::CircleAdd *_add_one = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleMul *_mul_half = nullptr;
  luci::CircleConst *_const_sqrt = nullptr;
  luci::CircleConst *_const_one = nullptr;
  luci::CircleConst *_const_half = nullptr;
};

class FuseGeluTestGraph : public TestIOGraph, public GeluGraphlet
{
public:
  FuseGeluTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    GeluGraphlet::init(g());

    _ifm->x(input());
    _mul_sqrt->x(_ifm);
    _mul_sqrt->y(_const_sqrt);
    _erf->inputs(0, _mul_sqrt);
    _erf_out->input(_erf);
    _add_one->x(_erf_out);
    _add_one->y(_const_one);
    _mul->x(_ifm);
    _mul->y(_add_one);
    _mul_half->x(_mul);
    _mul_half->y(_const_half);

    output()->from(_mul_half);
  }
};

class FuseGeluTestNegGraph : public TestIOGraph, public GeluGraphlet
{
public:
  FuseGeluTestNegGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    GeluGraphlet::init(g());

    _ifm->x(input());
    _mul_sqrt->x(_ifm);
    // NOTE y is incorrect (should be _const_sqrt)
    _mul_sqrt->y(_ifm);
    _erf->inputs(0, _mul_sqrt);
    _erf_out->input(_erf);
    _add_one->x(_erf_out);
    _add_one->y(_const_one);
    _mul->x(_ifm);
    _mul->y(_add_one);
    _mul_half->x(_mul);
    _mul_half->y(_const_half);

    output()->from(_mul_half);
  }
};

} // namespace

TEST(FuseGeluPassTest, name)
{
  luci::FuseGeluPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FuseGeluPassTest, fuse)
{
  FuseGeluTestGraph g;
  luci::FuseGeluPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseGeluPassTest, fuse_invalid_half_NEG)
{
  FuseGeluTestNegGraph g;
  luci::FuseGeluPass pass;

  g.init();
  g.invalid_half();

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(FuseGeluPassTest, fuse_invalid_act_NEG)
{
  FuseGeluTestNegGraph g;
  luci::FuseGeluPass pass;

  g.init();
  g.invalid_act();

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(FuseGeluPassTest, fuse_NEG)
{
  FuseGeluTestNegGraph g;
  luci::FuseGeluPass pass;

  g.init();

  EXPECT_FALSE(pass.run(g.g()));
}
