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
template <loco::DataType T> class FoldAddV2Test : public luci::ConstantFoldingAddTestGraph
{
public:
  FoldAddV2Test(std::initializer_list<uint32_t> shape) : luci::ConstantFoldingAddTestGraph(shape, T)
  {
    _addV2 = _g.nodes()->template create<luci::CircleCustom>(2, 1);
    _x = _g.nodes()->template create<luci::CircleConst>();
    _y = _g.nodes()->template create<luci::CircleConst>();
    _addV2_out = _g.nodes()->template create<luci::CircleCustomOut>();

    _addV2->dtype(T);
    _x->dtype(T);
    _y->dtype(T);
    _addV2_out->dtype(T);

    _addV2->shape(shape);
    _x->shape(shape);
    _y->shape(shape);
    _addV2_out->shape(shape);

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

    _addV2->custom_code("AddV2");
    _addV2->inputs(0, _x);
    _addV2->inputs(1, _y);
    _addV2_out->input(_addV2);

    _addV2->name("addV2");
    _x->name("x");
    _y->name("y");
  }

  loco::Node *createFoldedPattern() override { return _addV2_out; }

  virtual ~FoldAddV2Test() = default;

protected:
  luci::CircleCustom *_addV2 = nullptr;
  luci::CircleCustomOut *_addV2_out = nullptr;
  luci::CircleConst *_x = nullptr;
  luci::CircleConst *_y = nullptr;
};

class FoldS64AddV2Test : public FoldAddV2Test<loco::DataType::S64>, public ::testing::Test
{
public:
  FoldS64AddV2Test() : FoldAddV2Test<loco::DataType::S64>({3}) {}

  virtual void SetUp() { init(); }
};

} // namespace

TEST(FoldAddV2PassTest, name)
{
  luci::FoldAddV2Pass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldS64AddV2Test, fold_addV2)
{
  luci::FoldAddV2Pass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
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
  _x->dtype(loco::DataType::S32);

  luci::FoldAddV2Pass pass;
  EXPECT_FALSE(pass.run(graph()));
}
