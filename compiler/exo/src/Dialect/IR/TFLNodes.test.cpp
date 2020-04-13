/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TFLNodes.h"

#include "TFLDialect.h"
#include "TFLOpcode.h"

#include <gtest/gtest.h>

TEST(TFLAddTest, constructor)
{
  locoex::TFLAdd add_node;

  ASSERT_EQ(add_node.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(add_node.opcode(), locoex::TFLOpcode::ADD);

  ASSERT_EQ(add_node.x(), nullptr);
  ASSERT_EQ(add_node.y(), nullptr);
}

// TODO TFLAveragePool2D

TEST(TFLConcatTest, constructor)
{
  locoex::TFLConcatenation concat_node(3);

  ASSERT_EQ(concat_node.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(concat_node.opcode(), locoex::TFLOpcode::CONCATENATION);

  ASSERT_EQ(concat_node.numValues(), 3);
  ASSERT_EQ(concat_node.values(0), nullptr);
  ASSERT_EQ(concat_node.values(1), nullptr);
  ASSERT_EQ(concat_node.values(2), nullptr);
  ASSERT_EQ(concat_node.fusedActivationFunction(), locoex::FusedActFunc::UNDEFINED);
}

// TODO TFLConv2D

TEST(TFLDepthwiseConv2DTest, constructor)
{
  locoex::TFLDepthwiseConv2D dw_conv2d_node;

  ASSERT_EQ(dw_conv2d_node.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(dw_conv2d_node.opcode(), locoex::TFLOpcode::DEPTHWISE_CONV_2D);

  ASSERT_EQ(dw_conv2d_node.input(), nullptr);
  ASSERT_EQ(dw_conv2d_node.filter(), nullptr);
  ASSERT_EQ(dw_conv2d_node.bias(), nullptr);
  ASSERT_EQ(dw_conv2d_node.padding(), locoex::Padding::UNDEFINED);
  ASSERT_EQ(dw_conv2d_node.stride()->h(), 1);
  ASSERT_EQ(dw_conv2d_node.stride()->w(), 1);
  ASSERT_EQ(dw_conv2d_node.depthMultiplier(), 0);
  ASSERT_EQ(dw_conv2d_node.fusedActivationFunction(), locoex::FusedActFunc::UNDEFINED);
}

TEST(TFLDivTest, constructor)
{
  locoex::TFLDiv div_node;

  ASSERT_EQ(div_node.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(div_node.opcode(), locoex::TFLOpcode::DIV);

  ASSERT_EQ(div_node.x(), nullptr);
  ASSERT_EQ(div_node.y(), nullptr);
}

// TODO TFLMaxPool2D

TEST(TFLMulTest, constructor)
{
  locoex::TFLMul mul_node;

  ASSERT_EQ(mul_node.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(mul_node.opcode(), locoex::TFLOpcode::MUL);

  ASSERT_EQ(mul_node.x(), nullptr);
  ASSERT_EQ(mul_node.y(), nullptr);
}

TEST(TFLReluTest, constructor)
{
  locoex::TFLRelu relu_node;

  ASSERT_EQ(relu_node.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(relu_node.opcode(), locoex::TFLOpcode::RELU);

  ASSERT_EQ(relu_node.features(), nullptr);
}

// TODO TFLRelu6

TEST(TFLReshapeTest, constructor)
{
  locoex::TFLReshape reshape;

  ASSERT_EQ(reshape.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(reshape.opcode(), locoex::TFLOpcode::RESHAPE);

  ASSERT_EQ(reshape.tensor(), nullptr);
  ASSERT_EQ(reshape.shape(), nullptr);
  ASSERT_EQ(reshape.newShape()->rank(), 0);
}

TEST(TFLReshapeTest, alloc_new_shape)
{
  locoex::TFLReshape reshape;

  reshape.newShape()->rank(2);
  ASSERT_EQ(reshape.newShape()->rank(), 2);

  reshape.newShape()->dim(0) = 0;
  reshape.newShape()->dim(1) = 1;

  auto &const_reshape = const_cast<const locoex::TFLReshape &>(reshape);
  ASSERT_EQ(const_reshape.newShape()->dim(0), 0);
  ASSERT_EQ(const_reshape.newShape()->dim(1), 1);
}

// TODO TFLSoftmax

// TODO TFLSqrt

TEST(TFLSubTest, constructor)
{
  locoex::TFLSub sub_node;

  ASSERT_EQ(sub_node.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(sub_node.opcode(), locoex::TFLOpcode::SUB);

  ASSERT_EQ(sub_node.x(), nullptr);
  ASSERT_EQ(sub_node.y(), nullptr);
}

// TODO TFLTanh

TEST(TFLTransposeTest, constructor)
{
  locoex::TFLTranspose tr_node;

  ASSERT_EQ(tr_node.dialect(), locoex::TFLDialect::get());
  ASSERT_EQ(tr_node.opcode(), locoex::TFLOpcode::TRANSPOSE);

  ASSERT_EQ(tr_node.a(), nullptr);
  ASSERT_EQ(tr_node.perm(), nullptr);
}
