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

#include "luci/Pass/FoldMulPass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Graph has an Mul Op with constant inputs
 *
 *    BEFORE
 *
 *    [CircleConst] [CircleConst]
 *               |    |
 *             [CircleMul]
 *                  |
 *             [CircleNode]
 *    AFTER
 *                             [CircleConst] [CircleConst]
 *                                       |    |
 *             [CircleConst]           [CircleMul]
 *                  |
 *             [CircleNode]
 */

template <loco::DataType T> class FoldMulTest : public luci::ConstantFoldingAddTestGraph
{
public:
  FoldMulTest(std::initializer_list<uint32_t> shape) : luci::ConstantFoldingAddTestGraph(shape, T)
  {
    _mul = _g.nodes()->template create<luci::CircleMul>();
    _x = _g.nodes()->template create<luci::CircleConst>();
    _y = _g.nodes()->template create<luci::CircleConst>();

    _mul->dtype(T);
    _x->dtype(T);
    _y->dtype(T);

    _mul->shape(shape);
    _x->shape(shape);
    _y->shape(shape);

    uint32_t num_elems = 1;
    for (auto dim = shape.begin(); dim != shape.end(); dim++)
      num_elems *= *dim;

    _x->size<T>(num_elems);
    _y->size<T>(num_elems);

    for (uint32_t i = 0; i < num_elems; i++)
    {
      _x->at<T>(i) = i + 1;
      _y->at<T>(i) = i + 1;
    }

    _mul->x(_x);
    _mul->y(_y);
    _mul->name("mul");
    _x->name("x");
    _y->name("y");
  }

  loco::Node *createFoldedPattern() override { return _mul; }

  virtual ~FoldMulTest() = default;

protected:
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_x = nullptr;
  luci::CircleConst *_y = nullptr;
};

class FoldF32MulTest : public FoldMulTest<loco::DataType::FLOAT32>, public ::testing::Test
{
public:
  FoldF32MulTest() : FoldMulTest<loco::DataType::FLOAT32>({3}) {}

  virtual void SetUp() { init(); }
};

} // namespace

TEST_F(FoldF32MulTest, name)
{
  luci::FoldMulPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldF32MulTest, fold_mul)
{
  luci::FoldMulPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(1, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(4, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(9, folded_const->at<loco::DataType::FLOAT32>(2));
}

TEST_F(FoldF32MulTest, input_type_mismatch_NEG)
{
  _x->dtype(loco::DataType::U4);

  luci::FoldMulPass pass;
  EXPECT_FALSE(pass.run(graph()));
}
