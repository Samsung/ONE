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

#include "luci/Pass/FoldAddV2Pass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Graph has an AddV2 Op with constant inputs
 *
 *    BEFORE
 *
 *    [CircleConst] [CircleConst]
 *               |   |
 *       [CircleCustom (AddV2)]
 *                 |
 *         [CircleCustomOut]
 *
 *    AFTER
 *
 *           [CircleConst]
 */
template <loco::DataType T> class FoldAddV2Test : public luci::ConstantFoldingTestGraph
{
public:
  FoldAddV2Test(std::initializer_list<uint32_t> shape) : luci::ConstantFoldingTestGraph(shape, T)
  {
    addV2 = g.nodes()->create<luci::CircleCustom>(2);
    x = g.nodes()->create<luci::CircleConst>();
    y = g.nodes()->create<luci::CircleConst>();
    addV2_out = g.nodes()->create<luci::CircleCustomOut>();

    addV2->dtype(T);
    x->dtype(T);
    y->dtype(T);
    addV2_out->dtype(T);

    addV2->shape(shape);
    x->shape(shape);
    y->shape(shape);
    addV2_out->shape(shape);

    uint32_t num_elems = 1;
    for (auto dim = shape.begin(); dim != shape.end(); dim++)
      num_elems *= *dim;

    x->size<T>(num_elems);
    y->size<T>(num_elems);

    for (uint32_t i = 0; i < num_elems; i++)
    {
      x->at<T>(i) = i + 1;
      y->at<T>(i) = i + 1;
    }

    addV2->custom_code("AddV2");
    addV2->inputs(0, x);
    addV2->inputs(1, y);
    addV2_out->input(addV2);
  }

  loco::Node *createFoldedPattern() override { return addV2_out; }

  virtual ~FoldAddV2Test() = default;

  // NOTE: we're not adding _ prefix as these class members are public
public:
  luci::CircleCustom *addV2 = nullptr;
  luci::CircleCustomOut *addV2_out = nullptr;
  luci::CircleConst *x = nullptr;
  luci::CircleConst *y = nullptr;
};

class FoldS64AddV2Test : public FoldAddV2Test<loco::DataType::S64>, public ::testing::Test
{
public:
  FoldS64AddV2Test() : FoldAddV2Test<loco::DataType::S64>({3}) {}

  virtual void SetUp() { init(); }
};

} // namespace

TEST_F(FoldS64AddV2Test, fold_addV2)
{
  luci::FoldAddV2Pass pass;
  while (pass.run(&g))
    ;

  auto folded_const = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S64, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->at<loco::DataType::S64>(0));
  EXPECT_EQ(4, folded_const->at<loco::DataType::S64>(1));
  EXPECT_EQ(6, folded_const->at<loco::DataType::S64>(2));
}

TEST_F(FoldS64AddV2Test, input_type_mismatch_NEG)
{
  x->dtype(loco::DataType::S32);

  luci::FoldAddV2Pass pass;
  EXPECT_FALSE(pass.run(&g));
}
