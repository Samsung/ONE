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

  ASSERT_EQ(locoex::TFLDialect::get(), add_node.dialect());
  ASSERT_EQ(locoex::TFLOpcode::ADD, add_node.opcode());

  ASSERT_EQ(nullptr, add_node.x());
  ASSERT_EQ(nullptr, add_node.y());
}

// TODO TFLAveragePool2D

TEST(TFLConcatTest, constructor)
{
  locoex::TFLConcatenation concat_node(3);

  ASSERT_EQ(locoex::TFLDialect::get(), concat_node.dialect());
  ASSERT_EQ(locoex::TFLOpcode::CONCATENATION, concat_node.opcode());

  ASSERT_EQ(3, concat_node.numValues());
  ASSERT_EQ(nullptr, concat_node.values(0));
  ASSERT_EQ(nullptr, concat_node.values(1));
  ASSERT_EQ(nullptr, concat_node.values(2));
  ASSERT_EQ(locoex::FusedActFunc::UNDEFINED, concat_node.fusedActivationFunction());
}

// TODO TFLConv2D

TEST(TFLDepthwiseConv2DTest, constructor)
{
  locoex::TFLDepthwiseConv2D dw_conv2d_node;

  ASSERT_EQ(locoex::TFLDialect::get(), dw_conv2d_node.dialect());
  ASSERT_EQ(locoex::TFLOpcode::DEPTHWISE_CONV_2D, dw_conv2d_node.opcode());

  ASSERT_EQ(nullptr, dw_conv2d_node.input());
  ASSERT_EQ(nullptr, dw_conv2d_node.filter());
  ASSERT_EQ(nullptr, dw_conv2d_node.bias());
  ASSERT_EQ(locoex::Padding::UNDEFINED, dw_conv2d_node.padding());
  ASSERT_EQ(1, dw_conv2d_node.stride()->h());
  ASSERT_EQ(1, dw_conv2d_node.stride()->w());
  ASSERT_EQ(0, dw_conv2d_node.depthMultiplier());
  ASSERT_EQ(locoex::FusedActFunc::UNDEFINED, dw_conv2d_node.fusedActivationFunction());
}

TEST(TFLDivTest, constructor)
{
  locoex::TFLDiv div_node;

  ASSERT_EQ(locoex::TFLDialect::get(), div_node.dialect());
  ASSERT_EQ(locoex::TFLOpcode::DIV, div_node.opcode());

  ASSERT_EQ(nullptr, div_node.x());
  ASSERT_EQ(nullptr, div_node.y());
}

// TODO TFLMaxPool2D

TEST(TFLMulTest, constructor)
{
  locoex::TFLMul mul_node;

  ASSERT_EQ(locoex::TFLDialect::get(), mul_node.dialect());
  ASSERT_EQ(locoex::TFLOpcode::MUL, mul_node.opcode());

  ASSERT_EQ(nullptr, mul_node.x());
  ASSERT_EQ(nullptr, mul_node.y());
}

TEST(TFLReluTest, constructor)
{
  locoex::TFLRelu relu_node;

  ASSERT_EQ(locoex::TFLDialect::get(), relu_node.dialect());
  ASSERT_EQ(locoex::TFLOpcode::RELU, relu_node.opcode());

  ASSERT_EQ(nullptr, relu_node.features());
}

// TODO TFLRelu6

TEST(TFLReshapeTest, constructor)
{
  locoex::TFLReshape reshape;

  ASSERT_EQ(locoex::TFLDialect::get(), reshape.dialect());
  ASSERT_EQ(locoex::TFLOpcode::RESHAPE, reshape.opcode());

  ASSERT_EQ(nullptr, reshape.tensor());
  ASSERT_EQ(nullptr, reshape.shape());
  ASSERT_EQ(0, reshape.newShape()->rank());
}

TEST(TFLReshapeTest, alloc_new_shape)
{
  locoex::TFLReshape reshape;

  reshape.newShape()->rank(2);
  ASSERT_EQ(2, reshape.newShape()->rank());

  reshape.newShape()->dim(0) = 0;
  reshape.newShape()->dim(1) = 1;

  auto &const_reshape = const_cast<const locoex::TFLReshape &>(reshape);
  ASSERT_EQ(0, const_reshape.newShape()->dim(0));
  ASSERT_EQ(1, const_reshape.newShape()->dim(1));
}

// TODO TFLSoftmax

// TODO TFLSqrt

TEST(TFLSubTest, constructor)
{
  locoex::TFLSub sub_node;

  ASSERT_EQ(locoex::TFLDialect::get(), sub_node.dialect());
  ASSERT_EQ(locoex::TFLOpcode::SUB, sub_node.opcode());

  ASSERT_EQ(nullptr, sub_node.x());
  ASSERT_EQ(nullptr, sub_node.y());
}

// TODO TFLTanh

TEST(TFLTransposeTest, constructor)
{
  locoex::TFLTranspose tr_node;

  ASSERT_EQ(locoex::TFLDialect::get(), tr_node.dialect());
  ASSERT_EQ(locoex::TFLOpcode::TRANSPOSE, tr_node.opcode());

  ASSERT_EQ(nullptr, tr_node.a());
  ASSERT_EQ(nullptr, tr_node.perm());
}
