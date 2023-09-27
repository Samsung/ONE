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

#include "luci/Pass/ReplaceWithFCGeluFCPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <cmath>
#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class FCGeluFCGraphlet
{
public:
  FCGeluFCGraphlet() = default;

  virtual ~FCGeluFCGraphlet() = default;

  void init(loco::Graph *g)
  {
    _fc1 = g->nodes()->create<luci::CircleFullyConnected>();
    _fc2 = g->nodes()->create<luci::CircleFullyConnected>();
    _fc3 = g->nodes()->create<luci::CircleFullyConnected>();
    _erf = g->nodes()->create<luci::CircleCustom>(1, 1);
    _erf_out = g->nodes()->create<luci::CircleCustomOut>();
    _add_one = g->nodes()->create<luci::CircleAdd>();
    _mul = g->nodes()->create<luci::CircleMul>();
    _const_one = g->nodes()->create<luci::CircleConst>();
    _fc1_w = g->nodes()->create<luci::CircleConst>();
    _fc2_w = g->nodes()->create<luci::CircleConst>();
    _fc3_w = g->nodes()->create<luci::CircleConst>();
    auto no_bias = g->nodes()->create<luci::CircleOutputExclude>();

    _mul->fusedActivationFunction(luci::FusedActFunc::NONE);
    _add_one->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc1->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc2->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc3->fusedActivationFunction(luci::FusedActFunc::NONE);

    _fc1->name("fc1");
    _fc2->name("fc2");
    _fc3->name("fc3");
    _erf->name("erf");
    _erf_out->name("erf_out");
    _add_one->name("add_one");
    _mul->name("mul");
    _const_one->name("const_one");
    _fc1_w->name("fc1_w");
    _fc2_w->name("fc2_w");
    _fc3_w->name("fc3_w");

    _erf->custom_code("Erf");

    _const_one->dtype(loco::DataType::FLOAT32);
    _const_one->size<loco::DataType::FLOAT32>(1);
    _const_one->shape({1});
    _const_one->at<loco::DataType::FLOAT32>(0) = 1.0;
    _const_one->shape_status(luci::ShapeStatus::VALID);

    _fc1_w->dtype(loco::DataType::FLOAT32);
    _fc1_w->size<loco::DataType::FLOAT32>(16);
    _fc1_w->shape({4, 4});
    for (uint32_t i = 0; i < 16; i++)
      _fc1_w->at<loco::DataType::FLOAT32>(i) = 1.0;
    _fc1_w->shape_status(luci::ShapeStatus::VALID);

    _fc2_w->dtype(loco::DataType::FLOAT32);
    _fc2_w->size<loco::DataType::FLOAT32>(16);
    _fc2_w->shape({4, 4});
    for (uint32_t i = 0; i < 16; i++)
      _fc2_w->at<loco::DataType::FLOAT32>(i) = sqrtf(0.5);
    _fc2_w->shape_status(luci::ShapeStatus::VALID);

    _fc3_w->dtype(loco::DataType::FLOAT32);
    _fc3_w->size<loco::DataType::FLOAT32>(16);
    _fc3_w->shape({4, 4});
    for (uint32_t i = 0; i < 16; i++)
      _fc3_w->at<loco::DataType::FLOAT32>(i) = 1.0;
    _fc3_w->shape_status(luci::ShapeStatus::VALID);

    _fc1->dtype(loco::DataType::FLOAT32);
    _fc2->dtype(loco::DataType::FLOAT32);
    _fc3->dtype(loco::DataType::FLOAT32);
    _erf->dtype(loco::DataType::FLOAT32);
    _erf_out->dtype(loco::DataType::FLOAT32);
    _add_one->dtype(loco::DataType::FLOAT32);
    _mul->dtype(loco::DataType::FLOAT32);

    // Connect nodes
    _fc1->weights(_fc1_w);
    _fc1->bias(no_bias);
    _fc2->weights(_fc2_w);
    _fc2->bias(no_bias);
    _erf->inputs(0, _fc2);
    _erf_out->input(_erf);
    _add_one->x(_erf_out);
    _add_one->y(_const_one);
    _mul->x(_fc1);
    _mul->y(_add_one);
    _fc3->input(_mul);
    _fc3->weights(_fc3_w);
    _fc3->bias(no_bias);
  }

protected:
  luci::CircleFullyConnected *_fc1 = nullptr;
  luci::CircleFullyConnected *_fc2 = nullptr;
  luci::CircleFullyConnected *_fc3 = nullptr;
  luci::CircleCustom *_erf = nullptr;
  luci::CircleCustomOut *_erf_out = nullptr;
  luci::CircleAdd *_add_one = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_const_one = nullptr;
  luci::CircleConst *_fc1_w = nullptr;
  luci::CircleConst *_fc2_w = nullptr;
  luci::CircleConst *_fc3_w = nullptr;
};

class ReplaceWithFCGeluFCTestGraph : public TestIOGraph, public FCGeluFCGraphlet
{
public:
  ReplaceWithFCGeluFCTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 4, 4}, {1, 4, 4});
    FCGeluFCGraphlet::init(g());

    _fc1->input(input());
    _fc2->input(input());

    output()->from(_fc3);
  }
};

class ReplaceWithFCGeluFCTestNegGraph : public TestIOGraph, public FCGeluFCGraphlet
{
public:
  ReplaceWithFCGeluFCTestNegGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 4, 4}, {1, 4, 4});
    FCGeluFCGraphlet::init(g());
    _fc1->input(input());
    _fc2->input(_fc1);

    output()->from(_fc3);
  }
};

} // namespace

TEST(ReplaceWithFCGeluFCPassTest, basic)
{
  ReplaceWithFCGeluFCTestGraph g;
  luci::ReplaceWithFCGeluFCPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(ReplaceWithFCGeluFCPassTest, wrong_pattern_NEG)
{
  ReplaceWithFCGeluFCTestNegGraph g;
  luci::ReplaceWithFCGeluFCPass pass;

  g.init();

  EXPECT_FALSE(pass.run(g.g()));
}
