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

#include "luci/Pass/FoldCastPass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

template <loco::DataType FromT, loco::DataType ToT>
class FoldCastTest : public luci::ConstantFoldingTestGraph
{
public:
  FoldCastTest(std::initializer_list<uint32_t> shape) : luci::ConstantFoldingTestGraph(shape, ToT)
  {
    cast = g.nodes()->create<luci::CircleCast>();
    x = g.nodes()->create<luci::CircleConst>();

    cast->dtype(ToT);
    x->dtype(FromT);

    cast->shape(shape);
    x->shape(shape);

    uint32_t num_elems = 1;
    for (auto dim = shape.begin(); dim != shape.end(); dim++)
      num_elems *= *dim;

    x->size<FromT>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
      x->at<FromT>(i) = i + 1;

    cast->x(x);
  }

  loco::Node *createFoldedPattern() override { return cast; }

  // NOTE: we're not adding _ prefix as these class members are public
public:
  luci::CircleCast *cast = nullptr;
  luci::CircleConst *x = nullptr;
};

/**
 *  Graph that has a Cast Op with constant input
 *
 *    BEFORE
 *
 *         [CircleConst]
 *               |
 *            [Cast]
 *
 *    AFTER
 *
 *         [CircleConst]
 *
 */
class FoldS64ToS32CastTest : public FoldCastTest<loco::DataType::S64, loco::DataType::S32>,
                             public ::testing::Test
{
public:
  FoldS64ToS32CastTest() : FoldCastTest<loco::DataType::S64, loco::DataType::S32>({3}) {}

  virtual void SetUp() { init(); }
};

} // namespace

TEST_F(FoldS64ToS32CastTest, fold_cast_s64_to_s32)
{
  luci::FoldCastPass pass;
  while (pass.run(&g))
    ;

  auto folded_const = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S32, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(1, folded_const->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, folded_const->at<loco::DataType::S32>(1));
  EXPECT_EQ(3, folded_const->at<loco::DataType::S32>(2));
}

TEST_F(FoldS64ToS32CastTest, value_overflow)
{
  int64_t int32_max = std::numeric_limits<int32_t>::max();
  x->at<loco::DataType::S64>(0) = int32_max + 1;

  luci::FoldCastPass pass;
  while (pass.run(&g))
    ;

  auto folded_const = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S32, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(int32_max, folded_const->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, folded_const->at<loco::DataType::S32>(1));
  EXPECT_EQ(3, folded_const->at<loco::DataType::S32>(2));
}
