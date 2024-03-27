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

#include "luci/Pass/FoldDequantizePass.h"
#include "PassTestGraphs.h"

#include <gtest/gtest.h>

namespace
{

template <loco::DataType DT>
class FoldDequantizeTest : public luci::ConstantFoldingAddTestGraph, public ::testing::Test
{
public:
  FoldDequantizeTest() : luci::ConstantFoldingAddTestGraph({2, 2, 2}, DT) {}

  virtual void SetUp() { init(); }

  loco::Node *createFoldedPattern() override
  {
    _dequantize = _g.nodes()->template create<luci::CircleDequantize>();
    _input = _g.nodes()->template create<luci::CircleConst>();

    _dequantize->dtype(loco::DataType::FLOAT32);
    _input->dtype(DT);

    _input->shape({2, 2, 2});

    _input->size<DT>(8);
    _input->at<DT>(0) = 0;
    _input->at<DT>(1) = 1;
    _input->at<DT>(2) = 2;
    _input->at<DT>(3) = 3;
    _input->at<DT>(4) = 4;
    _input->at<DT>(5) = 5;
    _input->at<DT>(6) = 6;
    _input->at<DT>(7) = 7;

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    qparam->quantized_dimension = 1;
    qparam->scale.push_back(5.0);
    qparam->scale.push_back(10.0);
    qparam->zerop.push_back(1);
    qparam->zerop.push_back(2);
    _input->quantparam(std::move(qparam));

    _dequantize->input(_input);

    _dequantize->name("dequantize");
    _input->name("input");

    return _dequantize;
  }

  void createScalarPattern()
  {
    _input->rank(0);
    _input->size<DT>(1);
    _input->at<DT>(0) = 1;

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    qparam->quantized_dimension = 0;
    qparam->scale.push_back(1.0);
    qparam->zerop.push_back(0);
    _input->quantparam(std::move(qparam));
  }

  void createNotFoldablePattern() { _input->quantparam(nullptr); }

protected:
  luci::CircleDequantize *_dequantize = nullptr;
  luci::CircleConst *_input = nullptr;
};

class U4FoldDequantizeTest : public FoldDequantizeTest<loco::DataType::U4>
{
};

class S4FoldDequantizeTest : public FoldDequantizeTest<loco::DataType::S4>
{
};

class S8FoldDequantizeTest : public FoldDequantizeTest<loco::DataType::S8>
{
};

class S16FoldDequantizeTest : public FoldDequantizeTest<loco::DataType::S16>
{
};

class S32FoldDequantizeTest : public FoldDequantizeTest<loco::DataType::S32>
{
};

class S64FoldDequantizeTest : public FoldDequantizeTest<loco::DataType::S64>
{
};

class U8FoldDequantizeTest : public FoldDequantizeTest<loco::DataType::U8>
{
};

class F16FoldDequantizeTest : public luci::ConstantFoldingTestGraph, public ::testing::Test
{
public:
  F16FoldDequantizeTest() : ConstantFoldingTestGraph({2, 2}, loco::DataType::FLOAT16) {}

  virtual void SetUp() { init(); }

  loco::Node *createFoldedPattern() override
  {
    const auto DT = loco::DataType::FLOAT16;
    _dequantize = _g.nodes()->create<luci::CircleDequantize>();
    _f16const = _g.nodes()->create<luci::CircleConst>();

    _dequantize->dtype(loco::DataType::FLOAT32);
    _f16const->dtype(DT);

    _f16const->shape({2, 2});

    _f16const->size<loco::DataType::FLOAT16>(4);
    _f16const->at<DT>(0) = 49408; // -2.5f
    _f16const->at<DT>(1) = 47104; // -0.5f
    _f16const->at<DT>(2) = 0;     //  0.0f
    _f16const->at<DT>(3) = 15872; //  1.5f
    // NOTE how to get uint16_t value of float16 ?
    // Use compiler/souschef/src/Gaussian.cpp GaussianFloat16DataChef::generate()
    //   uint16_t value = fp16_ieee_from_fp32_value(-2.5);
    //   printf("-2.5 = %u\r\n", value);

    _dequantize->input(_f16const);

    _dequantize->name("dequantize");
    _f16const->name("input");

    _output->from(_dequantize);

    return _dequantize;
  }

  void createNotFoldablePattern() { _dequantize->input(_input); }

protected:
  luci::CircleConst *getFoldedPattern() override
  {
    return dynamic_cast<luci::CircleConst *>(_output->from());
  }

  void init() override { createFoldedPattern(); }

protected:
  luci::CircleDequantize *_dequantize = nullptr;
  luci::CircleConst *_f16const = nullptr;
};

} // namespace

TEST(FoldDequantizePassTest, name)
{
  luci::FoldDequantizePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(U8FoldDequantizeTest, fold_dequant_basic)
{
  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(3, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(2, folded_const->dim(2).value());
  EXPECT_EQ(-5.0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(2));
  EXPECT_EQ(10.0, folded_const->at<loco::DataType::FLOAT32>(3));
  EXPECT_EQ(15.0, folded_const->at<loco::DataType::FLOAT32>(4));
  EXPECT_EQ(20.0, folded_const->at<loco::DataType::FLOAT32>(5));
  EXPECT_EQ(40.0, folded_const->at<loco::DataType::FLOAT32>(6));
  EXPECT_EQ(50.0, folded_const->at<loco::DataType::FLOAT32>(7));
}

TEST_F(U8FoldDequantizeTest, fold_dequant_basic_NEG)
{
  createNotFoldablePattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(U4FoldDequantizeTest, fold_dequant_basic)
{
  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(3, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(2, folded_const->dim(2).value());
  EXPECT_EQ(-5.0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(2));
  EXPECT_EQ(10.0, folded_const->at<loco::DataType::FLOAT32>(3));
  EXPECT_EQ(15.0, folded_const->at<loco::DataType::FLOAT32>(4));
  EXPECT_EQ(20.0, folded_const->at<loco::DataType::FLOAT32>(5));
  EXPECT_EQ(40.0, folded_const->at<loco::DataType::FLOAT32>(6));
  EXPECT_EQ(50.0, folded_const->at<loco::DataType::FLOAT32>(7));
}

TEST_F(U4FoldDequantizeTest, fold_dequant_basic_NEG)
{
  createNotFoldablePattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(S4FoldDequantizeTest, fold_dequant_basic)
{
  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(3, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(2, folded_const->dim(2).value());
  EXPECT_EQ(-5.0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(2));
  EXPECT_EQ(10.0, folded_const->at<loco::DataType::FLOAT32>(3));
  EXPECT_EQ(15.0, folded_const->at<loco::DataType::FLOAT32>(4));
  EXPECT_EQ(20.0, folded_const->at<loco::DataType::FLOAT32>(5));
  EXPECT_EQ(40.0, folded_const->at<loco::DataType::FLOAT32>(6));
  EXPECT_EQ(50.0, folded_const->at<loco::DataType::FLOAT32>(7));
}

TEST_F(S4FoldDequantizeTest, fold_dequant_basic_NEG)
{
  createNotFoldablePattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(S8FoldDequantizeTest, fold_dequant_basic)
{
  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(3, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(2, folded_const->dim(2).value());
  EXPECT_EQ(-5.0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(2));
  EXPECT_EQ(10.0, folded_const->at<loco::DataType::FLOAT32>(3));
  EXPECT_EQ(15.0, folded_const->at<loco::DataType::FLOAT32>(4));
  EXPECT_EQ(20.0, folded_const->at<loco::DataType::FLOAT32>(5));
  EXPECT_EQ(40.0, folded_const->at<loco::DataType::FLOAT32>(6));
  EXPECT_EQ(50.0, folded_const->at<loco::DataType::FLOAT32>(7));
}

TEST_F(S8FoldDequantizeTest, fold_dequant_basic_NEG)
{
  createNotFoldablePattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(S16FoldDequantizeTest, fold_dequant_basic)
{
  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(3, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(2, folded_const->dim(2).value());
  EXPECT_EQ(-5.0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(2));
  EXPECT_EQ(10.0, folded_const->at<loco::DataType::FLOAT32>(3));
  EXPECT_EQ(15.0, folded_const->at<loco::DataType::FLOAT32>(4));
  EXPECT_EQ(20.0, folded_const->at<loco::DataType::FLOAT32>(5));
  EXPECT_EQ(40.0, folded_const->at<loco::DataType::FLOAT32>(6));
  EXPECT_EQ(50.0, folded_const->at<loco::DataType::FLOAT32>(7));
}

TEST_F(S16FoldDequantizeTest, fold_dequant_basic_NEG)
{
  createNotFoldablePattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(S32FoldDequantizeTest, fold_dequant_basic)
{
  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(3, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(2, folded_const->dim(2).value());
  EXPECT_EQ(-5.0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(2));
  EXPECT_EQ(10.0, folded_const->at<loco::DataType::FLOAT32>(3));
  EXPECT_EQ(15.0, folded_const->at<loco::DataType::FLOAT32>(4));
  EXPECT_EQ(20.0, folded_const->at<loco::DataType::FLOAT32>(5));
  EXPECT_EQ(40.0, folded_const->at<loco::DataType::FLOAT32>(6));
  EXPECT_EQ(50.0, folded_const->at<loco::DataType::FLOAT32>(7));
}

TEST_F(S32FoldDequantizeTest, fold_dequant_basic_NEG)
{
  createNotFoldablePattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(S64FoldDequantizeTest, fold_dequant_basic)
{
  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(3, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(2, folded_const->dim(2).value());
  EXPECT_EQ(-5.0, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(2));
  EXPECT_EQ(10.0, folded_const->at<loco::DataType::FLOAT32>(3));
  EXPECT_EQ(15.0, folded_const->at<loco::DataType::FLOAT32>(4));
  EXPECT_EQ(20.0, folded_const->at<loco::DataType::FLOAT32>(5));
  EXPECT_EQ(40.0, folded_const->at<loco::DataType::FLOAT32>(6));
  EXPECT_EQ(50.0, folded_const->at<loco::DataType::FLOAT32>(7));
}

TEST_F(S64FoldDequantizeTest, fold_dequant_basic_NEG)
{
  createNotFoldablePattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(U8FoldDequantizeTest, fold_dequant_scalar)
{
  createScalarPattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Check type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(0, folded_const->rank());
  EXPECT_EQ(1.0, folded_const->at<loco::DataType::FLOAT32>(0));
}

TEST_F(F16FoldDequantizeTest, fold_dequant_basic)
{
  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(2, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(-2.5, folded_const->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(-0.5, folded_const->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(0.0, folded_const->at<loco::DataType::FLOAT32>(2));
  EXPECT_EQ(1.5, folded_const->at<loco::DataType::FLOAT32>(3));
}

TEST_F(F16FoldDequantizeTest, fold_dequant_basic_NEG)
{
  createNotFoldablePattern();

  luci::FoldDequantizePass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}
