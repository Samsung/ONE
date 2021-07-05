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

#include "luci/Pass/FoldNegPass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

template <loco::DataType DT> class FoldNegTest : public luci::ConstantFoldingAddTestGraph
{
public:
  FoldNegTest() : luci::ConstantFoldingAddTestGraph({3}, DT)
  {
    _neg = _g.nodes()->create<luci::CircleNeg>();
    _x = _g.nodes()->create<luci::CircleConst>();

    _neg->dtype(DT);
    _x->dtype(DT);

    _neg->shape({3});
    _x->shape({3});

    _x->size<DT>(3);
    for (uint32_t i = 0; i < 3; i++)
      _x->at<DT>(i) = i;

    _neg->x(_x);

    _neg->name("neg");
    _x->name("x");
  }

  loco::Node *createFoldedPattern() override { return _neg; }

  void set_qparam()
  {
    auto qparam = std::make_unique<luci::CircleQuantParam>();
    for (uint32_t i = 0; i < 3; ++i)
    {
      qparam->zerop.emplace_back(i);
      qparam->scale.emplace_back(i);
    }
    _x->quantparam(std::move(qparam));
  }

protected:
  luci::CircleNeg *_neg = nullptr;
  luci::CircleConst *_x = nullptr;
};

class FoldNegFloat32Test : public FoldNegTest<loco::DataType::FLOAT32>, public ::testing::Test
{
public:
  FoldNegFloat32Test() : FoldNegTest<loco::DataType::FLOAT32>() {}

  virtual void SetUp() { init(); }
};

class FoldNegQuantU8Test : public FoldNegTest<loco::DataType::U8>, public ::testing::Test
{
public:
  FoldNegQuantU8Test() : FoldNegTest<loco::DataType::U8>() {}

  virtual void SetUp()
  {
    set_qparam();
    init();
  }
};

class FoldNegQuantS16Test : public FoldNegTest<loco::DataType::S16>, public ::testing::Test
{
public:
  FoldNegQuantS16Test() : FoldNegTest<loco::DataType::S16>() {}

  virtual void SetUp()
  {
    set_qparam();
    init();
  }
};

} // namespace

TEST(FoldNegTest, name)
{
  luci::FoldNegPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldNegFloat32Test, simple_float32)
{
  luci::FoldNegPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_EQ(0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(-1, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(-2, folded_const->at<loco::DataType::FLOAT32>(2));
}

TEST_F(FoldNegQuantU8Test, simple_quant_u8)
{
  luci::FoldNegPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::U8, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_NE(nullptr, folded_const->quantparam());
  EXPECT_EQ(3, folded_const->quantparam()->scale.size());
  EXPECT_EQ(0, folded_const->quantparam()->scale.at(0));
  EXPECT_EQ(-1, folded_const->quantparam()->scale.at(1));
  EXPECT_EQ(-2, folded_const->quantparam()->scale.at(2));
}

TEST_F(FoldNegQuantS16Test, simple_quant_s16)
{
  luci::FoldNegPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S16, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(3, folded_const->dim(0).value());
  EXPECT_NE(nullptr, folded_const->quantparam());
  EXPECT_EQ(3, folded_const->quantparam()->scale.size());
  EXPECT_EQ(0, folded_const->quantparam()->scale.at(0));
  EXPECT_EQ(-1, folded_const->quantparam()->scale.at(1));
  EXPECT_EQ(-2, folded_const->quantparam()->scale.at(2));
}
