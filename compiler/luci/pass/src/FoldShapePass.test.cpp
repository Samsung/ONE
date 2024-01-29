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

#include "luci/Pass/FoldShapePass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

template <loco::DataType OutType> class FoldShapeGraph : public luci::ConstantFoldingAddTestGraph
{
public:
  FoldShapeGraph(std::vector<uint32_t> input_shape)
    : luci::ConstantFoldingAddTestGraph(input_shape, OutType)
  {
    _x = _g.nodes()->template create<luci::CircleConst>();
    _x->name("x");
    _x->dtype(loco::DataType::S32);
    _x->rank(input_shape.size());
    for (uint32_t i = 0; i < input_shape.size(); i++)
      _x->dim(i).set(input_shape.at(i));
    _x->shape_status(luci::ShapeStatus::VALID);

    _shape = _g.nodes()->template create<luci::CircleShape>();
    _shape->name("shape");
    _shape->out_type(OutType);
    _shape->input(_x);
    _shape->shape({4});
    _shape->rank(1);
    _shape->dim(0).set(4);
  }

  loco::Node *createFoldedPattern() override { return _shape; }

protected:
  luci::CircleShape *_shape = nullptr;
  luci::CircleConst *_x = nullptr;
};

/**
 *  Graph that has a Shape Op
 *
 *    BEFORE
 *
 *                     [CircleConst]
 *                           |
 *    [CircleInput]    [CircleShape]
 *              \       /
 *             [CircleAdd]
 *                  |
 *            [CircleOutput]
 *
 *    AFTER
 *
 *    [CircleInput]    [CircleConst]
 *              \       /
 *             [CircleAdd]
 *                  |
 *            [CircleOutput]
 *
 */
class FoldShapePassGraphTest : public FoldShapeGraph<loco::DataType::S32>, public ::testing::Test
{
public:
  FoldShapePassGraphTest() : FoldShapeGraph<loco::DataType::S32>({1, 8, 8, 64}) {}

  virtual void SetUp() { init(); }
};

} // namespace

TEST(FoldShapePassTest, name)
{
  luci::FoldShapePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldShapePassGraphTest, fold_shape)
{
  luci::FoldShapePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded shape
  EXPECT_EQ(loco::DataType::S32, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(4, folded_const->dim(0).value());
  EXPECT_EQ(1, folded_const->at<loco::DataType::S32>(0));
  EXPECT_EQ(8, folded_const->at<loco::DataType::S32>(1));
  EXPECT_EQ(8, folded_const->at<loco::DataType::S32>(2));
  EXPECT_EQ(64, folded_const->at<loco::DataType::S32>(3));
}

TEST_F(FoldShapePassGraphTest, undefined_shape_NEG)
{
  _x->shape_status(luci::ShapeStatus::UNDEFINED);

  luci::FoldShapePass pass;
  EXPECT_FALSE(pass.run(graph()));
}

TEST_F(FoldShapePassGraphTest, unallowed_rank_NEG)
{
  _x->rank(0);

  luci::FoldShapePass pass;
  EXPECT_FALSE(pass.run(graph()));
}

TEST_F(FoldShapePassGraphTest, unknown_dimension_NEG)
{
  _x->dim(0).unset();

  luci::FoldShapePass pass;
  EXPECT_FALSE(pass.run(graph()));
}
