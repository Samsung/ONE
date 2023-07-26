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

#include "luci/Pass/FoldSparseToDensePass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Graph that has a SparseToDense Op with zero-sized indices
 *
 *    BEFORE
 *    - shape of indices: [0,1]
 *    - output_shape: [3]
 *    - default_value: scalar 2
 *
 *     [indices] [output_shape] [values] [default_value]
 *            |         |          |      |
 *            +------[SparseToDense]------+
 *
 *    AFTER
 *
 *            [Const] (shape: [3], values: [2, 2, 2])
 *
 */
class S64SparseToDenseZeroIndicesTest : public luci::ConstantFoldingAddTestGraph,
                                        public ::testing::Test
{
public:
  S64SparseToDenseZeroIndicesTest() : luci::ConstantFoldingAddTestGraph({3}, loco::DataType::S64) {}

  void SetUp() override { init(); }

  loco::Node *createFoldedPattern() override
  {
    _stod = _g.nodes()->create<luci::CircleSparseToDense>();
    _indices = _g.nodes()->create<luci::CircleConst>();
    _output_shape = _g.nodes()->create<luci::CircleConst>();
    _values = _g.nodes()->create<luci::CircleConst>();
    _default_value = _g.nodes()->create<luci::CircleConst>();

    _stod->dtype(loco::DataType::S64);
    _indices->dtype(loco::DataType::S64);
    _output_shape->dtype(loco::DataType::S64);
    _values->dtype(loco::DataType::S64);
    _default_value->dtype(loco::DataType::S64);

    _indices->shape({0, 1});
    _output_shape->shape({1});
    _values->shape({0});
    _default_value->rank(0);

    _indices->size<loco::DataType::S64>(0);
    _output_shape->size<loco::DataType::S64>(1);
    _output_shape->at<loco::DataType::S64>(0) = 3;
    _values->size<loco::DataType::S64>(0);
    _default_value->size<loco::DataType::S64>(1);
    _default_value->at<loco::DataType::S64>(0) = 2;

    _stod->indices(_indices);
    _stod->output_shape(_output_shape);
    _stod->values(_values);
    _stod->default_value(_default_value);

    _stod->name("stod");
    _indices->name("indices");
    _output_shape->name("output_shape");
    _values->name("values");
    _default_value->name("default_value");

    return _stod;
  }

protected:
  luci::CircleSparseToDense *_stod = nullptr;
  luci::CircleConst *_indices = nullptr;
  luci::CircleConst *_output_shape = nullptr;
  luci::CircleConst *_values = nullptr;
  luci::CircleConst *_default_value = nullptr;
};

} // namespace

TEST(FoldSparseToDensePassTest, name)
{
  luci::FoldSparseToDensePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(S64SparseToDenseZeroIndicesTest, fold_stod_with_zero_indices)
{
  luci::FoldSparseToDensePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S64, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->at<loco::DataType::S64>(0));
  EXPECT_EQ(2, folded_const->at<loco::DataType::S64>(1));
  EXPECT_EQ(2, folded_const->at<loco::DataType::S64>(2));
}

TEST_F(S64SparseToDenseZeroIndicesTest, illegal_input_NEG)
{
  _indices->dtype(loco::DataType::S32);

  luci::FoldSparseToDensePass pass;
  EXPECT_ANY_THROW(pass.run(graph()));
}
