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
class FoldCastTest : public luci::ConstantFoldingAddTestGraph
{
public:
  FoldCastTest(std::initializer_list<uint32_t> shape)
    : luci::ConstantFoldingAddTestGraph(shape, ToT)
  {
    _cast = _g.nodes()->template create<luci::CircleCast>();
    _x = _g.nodes()->template create<luci::CircleConst>();

    _cast->dtype(ToT);
    _x->dtype(FromT);

    _cast->shape(shape);
    _x->shape(shape);

    uint32_t num_elems = 1;
    for (auto dim = shape.begin(); dim != shape.end(); dim++)
      num_elems *= *dim;

    _x->size<FromT>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
      _x->at<FromT>(i) = i + 1;

    _cast->x(_x);

    _cast->name("cast");
    _x->name("x");
  }

  loco::Node *createFoldedPattern() override { return _cast; }

protected:
  luci::CircleCast *_cast = nullptr;
  luci::CircleConst *_x = nullptr;
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

TEST(FoldCastPassTest, name)
{
  luci::FoldCastPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldS64ToS32CastTest, fold_cast_s64_to_s32)
{
  luci::FoldCastPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S32, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(1, folded_const->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, folded_const->at<loco::DataType::S32>(1));
  EXPECT_EQ(3, folded_const->at<loco::DataType::S32>(2));
}
