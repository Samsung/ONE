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

#include "luci/Pass/FoldSqueezePass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

template <loco::DataType DT> class FoldSqueezeTest : public luci::ConstantFoldingAddTestGraph
{
public:
  FoldSqueezeTest(std::initializer_list<uint32_t> input_shape,
                  std::initializer_list<uint32_t> output_shape)
    : luci::ConstantFoldingAddTestGraph(output_shape, DT)
  {
    _squeeze = _g.nodes()->template create<luci::CircleSqueeze>();
    _x = _g.nodes()->template create<luci::CircleConst>();

    _squeeze->dtype(DT);
    _x->dtype(DT);

    _squeeze->shape(output_shape);
    _x->shape(input_shape);

    _squeeze->squeeze_dims({0});

    uint32_t num_elems = 1;
    for (auto dim = input_shape.begin(); dim != input_shape.end(); dim++)
      num_elems *= *dim;

    _x->size<DT>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
      _x->at<DT>(i) = i;

    _squeeze->input(_x);

    _squeeze->name("squeeze");
    _x->name("x");
  }

  loco::Node *createFoldedPattern() override { return _squeeze; }

public:
  void set_unknown_dim() { _x->dim(0).unset(); }

protected:
  luci::CircleSqueeze *_squeeze = nullptr;
  luci::CircleConst *_x = nullptr;
};

/**
 *  Graph that has a Squeeze Op with constant input
 *
 *    BEFORE
 *
 *         [CircleConst]
 *               |
 *            [Squeeze]
 *
 *    AFTER
 *
 *         [CircleConst]
 *
 */
class FoldFP32SqueezeTest : public FoldSqueezeTest<loco::DataType::FLOAT32>, public ::testing::Test
{
public:
  FoldFP32SqueezeTest() : FoldSqueezeTest<loco::DataType::FLOAT32>({1, 3}, {3}) {}

  virtual void SetUp() { init(); }
};

class FoldS16SqueezeTest : public FoldSqueezeTest<loco::DataType::S16>, public ::testing::Test
{
public:
  FoldS16SqueezeTest() : FoldSqueezeTest<loco::DataType::S16>({1, 3}, {3}) {}

  virtual void SetUp() { init(); }
};

} // namespace

TEST_F(FoldFP32SqueezeTest, fold_squeeze_fp32)
{
  luci::FoldSqueezePass pass;
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

TEST_F(FoldFP32SqueezeTest, fold_squeeze_unkown_dim_NEG)
{
  set_unknown_dim();

  luci::FoldSqueezePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(FoldS16SqueezeTest, fold_squeeze_s16)
{
  luci::FoldSqueezePass pass;
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
