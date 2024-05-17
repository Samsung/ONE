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

#include "luci/Pass/FoldReshapePass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

template <loco::DataType DT> class FoldReshapeTest : public luci::ConstantFoldingAddTestGraph
{
public:
  FoldReshapeTest(std::initializer_list<uint32_t> input_shape,
                  std::initializer_list<uint32_t> output_shape)
    : luci::ConstantFoldingAddTestGraph(output_shape, DT)
  {
    _reshape = _g.nodes()->template create<luci::CircleReshape>();
    _x = _g.nodes()->template create<luci::CircleConst>();
    _shape = _g.nodes()->template create<luci::CircleConst>();

    _reshape->dtype(DT);
    _x->dtype(DT);
    _shape->dtype(loco::DataType::S32);

    _reshape->shape(_shape);
    _x->shape(input_shape);
    _shape->shape({static_cast<uint32_t>(output_shape.size())});

    uint32_t num_elems = 1;
    for (auto dim : input_shape)
      num_elems *= dim;

    _x->size<DT>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
      _x->at<DT>(i) = i;

    _shape->size<loco::DataType::S32>(output_shape.size());
    uint32_t i = 0;
    for (auto dim : output_shape)
    {
      _shape->at<loco::DataType::S32>(i++) = static_cast<int32_t>(dim);
    }

    _reshape->tensor(_x);
    _reshape->shape(_shape);

    _reshape->name("reshape");
    _shape->name("shape");
    _x->name("x");
  }

  loco::Node *createFoldedPattern() override { return _reshape; }

public:
  void set_unknown_dim() { _x->dim(0).unset(); }

protected:
  luci::CircleReshape *_reshape = nullptr;
  luci::CircleConst *_x = nullptr;
  luci::CircleConst *_shape = nullptr;
};

/**
 *  Graph that has a Reshape Op with constant input
 *
 *    BEFORE
 *
 *         [CircleConst]
 *               |
 *            [Reshape]
 *
 *    AFTER
 *
 *         [CircleConst]
 *
 */
class FoldFP32ReshapeTest : public FoldReshapeTest<loco::DataType::FLOAT32>, public ::testing::Test
{
public:
  FoldFP32ReshapeTest() : FoldReshapeTest<loco::DataType::FLOAT32>({1, 3}, {3}) {}

  virtual void SetUp() { init(); }
};

class FoldS16ReshapeTest : public FoldReshapeTest<loco::DataType::S16>, public ::testing::Test
{
public:
  FoldS16ReshapeTest() : FoldReshapeTest<loco::DataType::S16>({1, 3}, {3}) {}

  virtual void SetUp() { init(); }
};

} // namespace

TEST_F(FoldFP32ReshapeTest, fold_reshape_fp32)
{
  luci::FoldReshapePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(1, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(2, folded_const->at<loco::DataType::FLOAT32>(2));
}

TEST_F(FoldFP32ReshapeTest, fold_reshape_unkown_dim_NEG)
{
  set_unknown_dim();

  luci::FoldReshapePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(FoldS16ReshapeTest, fold_reshape_s16)
{
  luci::FoldReshapePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S16, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(0, folded_const->at<loco::DataType::S16>(0));
  EXPECT_EQ(1, folded_const->at<loco::DataType::S16>(1));
  EXPECT_EQ(2, folded_const->at<loco::DataType::S16>(2));
}
