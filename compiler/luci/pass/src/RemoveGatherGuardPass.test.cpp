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
#include "luci/Pass/RemoveGatherGuardPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class GatherGuardGraphlet
{
public:
  GatherGuardGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 params_s, const ShapeU32 indices_s,
            const ShapeU32 output_s)
  {
    std::vector<uint32_t> params_shape{params_s};

    _add_y = g->nodes()->create<luci::CircleConst>();
    _add_y->rank(0);
    _add_y->shape_status(luci::ShapeStatus::VALID);
    _add_y->dtype(loco::DataType::S32);
    _add_y->size<loco::DataType::S32>(1);
    _add_y->at<loco::DataType::S32>(0) = params_shape[0];

    _add = g->nodes()->create<luci::CircleAdd>();
    _add->fusedActivationFunction(luci::FusedActFunc::NONE);
    _add->dtype(loco::DataType::S32);
    _add->shape(indices_s);
    _add->shape_status(luci::ShapeStatus::VALID);

    _fm_y = g->nodes()->create<luci::CircleConst>();
    _fm_y->rank(0);
    _fm_y->shape_status(luci::ShapeStatus::VALID);
    _fm_y->dtype(loco::DataType::S32);
    _fm_y->size<loco::DataType::S32>(1);
    _fm_y->at<loco::DataType::S32>(0) = params_shape[0];

    _fm = g->nodes()->create<luci::CircleFloorMod>();
    _fm->dtype(loco::DataType::S32);
    _fm->shape(indices_s);
    _fm->shape_status(luci::ShapeStatus::VALID);

    _gather = g->nodes()->create<luci::CircleGather>();
    _gather->axis(0);
    _gather->dtype(loco::DataType::FLOAT32);
    _gather->shape(output_s);
    _gather->shape_status(luci::ShapeStatus::VALID);
  }

protected:
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_add_y = nullptr;
  luci::CircleFloorMod *_fm = nullptr;
  luci::CircleConst *_fm_y = nullptr;
  luci::CircleGather *_gather = nullptr;
};

class GatherGuardGraph : public TestIsGraphlet<2>, public TestOGraphlet, public GatherGuardGraphlet
{
public:
  GatherGuardGraph() = default;

public:
  void init(const ShapeU32 params_s, const ShapeU32 indices_s, const ShapeU32 output_s)
  {
    TestIsGraphlet<2>::init(g(), {params_s, indices_s});
    TestOGraphlet::init(g(), output_s);
    GatherGuardGraphlet::init(g(), params_s, indices_s, output_s);

    // connect graph
    _add->x(input(1));
    _add->y(_add_y);
    _fm->x(_add);
    _fm->y(_fm_y);
    _gather->params(input(0));
    _gather->indices(_fm);
    output()->from(_gather);
  }
};

class GatherGuardGraphTest : public ::testing::Test, public GatherGuardGraph
{
protected:
  luci::RemoveGatherGuardPass _pass;

  ShapeU32 _input_s0 = {10, 3};
  ShapeU32 _input_s1 = {5, 4};
  ShapeU32 _output_s = {5, 4, 3};
};

} // namespace

TEST_F(GatherGuardGraphTest, name)
{
  auto const name = _pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(GatherGuardGraphTest, removed)
{
  // test to check pass is working as expected

  init(_input_s0, _input_s1, _output_s);

  auto *indices_before = loco::must_cast<luci::CircleNode *>(_gather->indices());
  EXPECT_NE(input(1), indices_before);

  EXPECT_TRUE(_pass.run(g()));

  auto *indices_after = loco::must_cast<luci::CircleNode *>(_gather->indices());
  EXPECT_EQ(input(1), indices_after);
}

TEST_F(GatherGuardGraphTest, axis_value_NEG)
{
  // test if fails when gather->params->dim(0) != add/floormod rhs value

  init(_input_s0, _input_s1, _output_s);

  _add_y->at<loco::DataType::S32>(0) = 11;
  EXPECT_FALSE(_pass.run(g()));
  _add_y->at<loco::DataType::S32>(0) = 10;

  _fm_y->at<loco::DataType::S32>(0) = 11;
  EXPECT_FALSE(_pass.run(g()));
}

TEST_F(GatherGuardGraphTest, add_act_not_none_NEG)
{
  // test if fails when add activation function is not none

  init(_input_s0, _input_s1, _output_s);

  _add->fusedActivationFunction(luci::FusedActFunc::RELU);

  EXPECT_FALSE(_pass.run(g()));
}
