/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleDepthwiseConv2D.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleDepthwiseConv2DTest, constructor_P)
{
  luci::CircleDepthwiseConv2D dw_conv2d_node;

  ASSERT_EQ(luci::CircleDialect::get(), dw_conv2d_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::DEPTHWISE_CONV_2D, dw_conv2d_node.opcode());

  ASSERT_EQ(nullptr, dw_conv2d_node.input());
  ASSERT_EQ(nullptr, dw_conv2d_node.filter());
  ASSERT_EQ(nullptr, dw_conv2d_node.bias());
  ASSERT_EQ(luci::Padding::UNDEFINED, dw_conv2d_node.padding());
  ASSERT_EQ(1, dw_conv2d_node.stride()->h());
  ASSERT_EQ(1, dw_conv2d_node.stride()->w());
  ASSERT_EQ(1, dw_conv2d_node.dilation()->h());
  ASSERT_EQ(1, dw_conv2d_node.dilation()->w());
  ASSERT_EQ(0, dw_conv2d_node.depthMultiplier());
  ASSERT_EQ(luci::FusedActFunc::UNDEFINED, dw_conv2d_node.fusedActivationFunction());
}

TEST(CircleDepthwiseConv2DTest, input_NEG)
{
  luci::CircleDepthwiseConv2D dw_conv2d_node;
  luci::CircleDepthwiseConv2D node;

  dw_conv2d_node.input(&node);
  dw_conv2d_node.filter(&node);
  dw_conv2d_node.bias(&node);
  ASSERT_NE(nullptr, dw_conv2d_node.input());
  ASSERT_NE(nullptr, dw_conv2d_node.filter());
  ASSERT_NE(nullptr, dw_conv2d_node.bias());

  dw_conv2d_node.input(nullptr);
  dw_conv2d_node.filter(nullptr);
  dw_conv2d_node.bias(nullptr);
  ASSERT_EQ(nullptr, dw_conv2d_node.input());
  ASSERT_EQ(nullptr, dw_conv2d_node.filter());
  ASSERT_EQ(nullptr, dw_conv2d_node.bias());

  dw_conv2d_node.padding(luci::Padding::SAME);
  ASSERT_NE(luci::Padding::UNDEFINED, dw_conv2d_node.padding());

  dw_conv2d_node.stride()->h(2);
  dw_conv2d_node.stride()->w(2);
  ASSERT_EQ(2, dw_conv2d_node.stride()->h());
  ASSERT_EQ(2, dw_conv2d_node.stride()->w());

  dw_conv2d_node.dilation()->h(2);
  dw_conv2d_node.dilation()->w(2);
  ASSERT_EQ(2, dw_conv2d_node.dilation()->h());
  ASSERT_EQ(2, dw_conv2d_node.dilation()->w());

  dw_conv2d_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  ASSERT_NE(luci::FusedActFunc::UNDEFINED, dw_conv2d_node.fusedActivationFunction());
}

TEST(CircleDepthwiseConv2DTest, arity_NEG)
{
  luci::CircleDepthwiseConv2D dw_conv2d_node;

  ASSERT_NO_THROW(dw_conv2d_node.arg(2));
  ASSERT_THROW(dw_conv2d_node.arg(3), std::out_of_range);
}

TEST(CircleDepthwiseConv2DTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleDepthwiseConv2D dw_conv2d_node;

  TestVisitor tv;
  ASSERT_THROW(dw_conv2d_node.accept(&tv), std::exception);
}

TEST(CircleDepthwiseConv2DTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleDepthwiseConv2D dw_conv2d_node;

  TestVisitor tv;
  ASSERT_THROW(dw_conv2d_node.accept(&tv), std::exception);
}
