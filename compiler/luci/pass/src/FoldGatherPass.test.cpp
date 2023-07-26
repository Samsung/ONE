/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FoldGatherPass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *
 *  Graph that has a Gather S64 Op with const inputs
 *
 *    BEFORE
 *    params: [Const] (shape: [3], values: [1, 2, 3])
 *    indices: [Const] (shape: [1], values: [1])
 *
 *     [params]     [indices]
 *        |            |
 *        ---[Gather]---
 *
 *    AFTER
 *    [Const] (shape: [1], values: [2])
 *
 */
class S64FoldGatherSimpleTest : public luci::ConstantFoldingAddTestGraph, public ::testing::Test
{
public:
  S64FoldGatherSimpleTest() : luci::ConstantFoldingAddTestGraph({1}, loco::DataType::S64) {}

  void SetUp() override { init(); }

  loco::Node *createFoldedPattern() override
  {
    _gather = _g.nodes()->create<luci::CircleGather>();
    _params = _g.nodes()->create<luci::CircleConst>();
    _indices = _g.nodes()->create<luci::CircleConst>();

    _gather->dtype(loco::DataType::S64);
    _params->dtype(loco::DataType::S64);
    _indices->dtype(loco::DataType::S64);

    _params->shape({3});
    _indices->shape({1});

    _params->size<loco::DataType::S64>(3);
    _params->at<loco::DataType::S64>(0) = 1;
    _params->at<loco::DataType::S64>(1) = 2;
    _params->at<loco::DataType::S64>(2) = 3;

    _indices->size<loco::DataType::S64>(1);
    _indices->at<loco::DataType::S64>(0) = 1;

    _gather->params(_params);
    _gather->indices(_indices);

    _gather->name("gather");
    _params->name("params");
    _indices->name("indices");

    return _gather;
  }

protected:
  luci::CircleGather *_gather = nullptr;
  luci::CircleConst *_params = nullptr;
  luci::CircleConst *_indices = nullptr;
};

/**
 *
 *  Graph that has a Gather S32 Op with axis = 1 and with const inputs
 *
 *    BEFORE
 *    params: [Const] (shape: [2, 3], values: [0, 1, 2, 3, 4, 5])
 *    indices: [Const] (shape: [2], values: [2, 1])
 *
 *     [params]     [indices]
 *        |            |
 *        ---[Gather]---
 *
 *    AFTER
 *    [Const] (shape: [2, 2], values: [2, 1, 5, 4])
 *
 */

class S32FoldGatherTwoDimsTest : public luci::ConstantFoldingAddTestGraph, public ::testing::Test
{
public:
  S32FoldGatherTwoDimsTest() : luci::ConstantFoldingAddTestGraph({4, 2}, loco::DataType::S32) {}

  void SetUp() override { init(); }

  loco::Node *createFoldedPattern() override
  {
    _gather = _g.nodes()->create<luci::CircleGather>();
    _params = _g.nodes()->create<luci::CircleConst>();
    _indices = _g.nodes()->create<luci::CircleConst>();

    _gather->dtype(loco::DataType::S32);
    _params->dtype(loco::DataType::S32);
    _indices->dtype(loco::DataType::S32);

    _params->shape({2, 3});
    _indices->shape({2});

    _params->size<loco::DataType::S32>(6);
    _params->at<loco::DataType::S32>(0) = 0;
    _params->at<loco::DataType::S32>(1) = 1;
    _params->at<loco::DataType::S32>(2) = 2;
    _params->at<loco::DataType::S32>(3) = 3;
    _params->at<loco::DataType::S32>(4) = 4;
    _params->at<loco::DataType::S32>(5) = 5;

    _indices->size<loco::DataType::S32>(2);
    _indices->at<loco::DataType::S32>(0) = 2;
    _indices->at<loco::DataType::S32>(1) = 1;

    _gather->params(_params);
    _gather->indices(_indices);

    _gather->axis(1);

    _gather->name("gather");
    _params->name("params");
    _indices->name("indices");

    return _gather;
  }

protected:
  luci::CircleGather *_gather = nullptr;
  luci::CircleConst *_params = nullptr;
  luci::CircleConst *_indices = nullptr;
};

} // namespace

TEST(FoldGatherTest, name)
{
  luci::FoldGatherPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(S64FoldGatherSimpleTest, fold_gather_simple)
{
  luci::FoldGatherPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S64, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(1, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->at<loco::DataType::S64>(0));
}

TEST_F(S32FoldGatherTwoDimsTest, fold_gather_with_two_dim)
{
  luci::FoldGatherPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S32, folded_const->dtype());
  EXPECT_EQ(2, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());

  EXPECT_EQ(2, folded_const->at<loco::DataType::S32>(0));
  EXPECT_EQ(1, folded_const->at<loco::DataType::S32>(1));
  EXPECT_EQ(5, folded_const->at<loco::DataType::S32>(2));
  EXPECT_EQ(4, folded_const->at<loco::DataType::S32>(3));
}

TEST_F(S64FoldGatherSimpleTest, illegal_input_NEG)
{
  _indices->dtype(loco::DataType::FLOAT32);

  luci::FoldGatherPass pass;
  EXPECT_ANY_THROW(pass.run(graph()));
}

TEST_F(S64FoldGatherSimpleTest, illegal_axis_NEG)
{
  _gather->axis(1);

  luci::FoldGatherPass pass;
  EXPECT_ANY_THROW(pass.run(graph()));
}
