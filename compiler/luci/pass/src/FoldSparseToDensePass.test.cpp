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
#include "PassTestUtils.h"

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
class S64SparseToDenseZeroIndicesGraph final : public luci::ConstantFoldingTestGraph
{
public:
  S64SparseToDenseZeroIndicesGraph(std::vector<int32_t> input_shape, loco::DataType input_dtype)
    : luci::ConstantFoldingTestGraph(input_shape, input_dtype)
  {
  }

public:
  loco::Node *createFoldedPattern() override
  {
    stod = g.nodes()->create<luci::CircleSparseToDense>();
    indices = g.nodes()->create<luci::CircleConst>();
    output_shape = g.nodes()->create<luci::CircleConst>();
    values = g.nodes()->create<luci::CircleConst>();
    default_value = g.nodes()->create<luci::CircleConst>();

    stod->dtype(loco::DataType::S64);
    indices->dtype(loco::DataType::S64);
    output_shape->dtype(loco::DataType::S64);
    values->dtype(loco::DataType::S64);
    default_value->dtype(loco::DataType::S64);

    indices->shape({0, 1});
    output_shape->shape({1});
    values->shape({0});
    default_value->rank(0);

    indices->size<loco::DataType::S64>(0);
    output_shape->size<loco::DataType::S64>(1);
    output_shape->at<loco::DataType::S64>(0) = 3;
    values->size<loco::DataType::S64>(0);
    default_value->size<loco::DataType::S64>(1);
    default_value->at<loco::DataType::S64>(0) = 2;

    stod->indices(indices);
    stod->output_shape(output_shape);
    stod->values(values);
    stod->default_value(default_value);

    return stod;
  }

public:
  luci::CircleSparseToDense *stod = nullptr;
  luci::CircleConst *indices = nullptr;
  luci::CircleConst *output_shape = nullptr;
  luci::CircleConst *values = nullptr;
  luci::CircleConst *default_value = nullptr;
};

} // namespace

TEST(FoldSparseToDense, S64Value)
{
  S64SparseToDenseZeroIndicesGraph g({3}, loco::DataType::S64);
  g.init();

  luci::FoldSparseToDensePass pass;
  while (pass.run(&g.g))
    ;

  auto folded_const = dynamic_cast<luci::CircleConst *>(g.add->y());
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S64, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->at<loco::DataType::S64>(0));
  EXPECT_EQ(2, folded_const->at<loco::DataType::S64>(1));
  EXPECT_EQ(2, folded_const->at<loco::DataType::S64>(2));
}

TEST(FoldSparseToDense, Illegal_input_NEG)
{
  S64SparseToDenseZeroIndicesGraph g({3}, loco::DataType::S64);
  g.init();

  g.indices->dtype(loco::DataType::S32);

  luci::FoldSparseToDensePass pass;
  EXPECT_ANY_THROW(pass.run(&g.g));
}
