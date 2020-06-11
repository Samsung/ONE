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

#include "luci/IR/Nodes/CircleConv2D.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleConv2Dest, constructor_P)
{
  luci::CircleConv2D conv2d_node;

  ASSERT_EQ(luci::CircleDialect::get(), conv2d_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CONV_2D, conv2d_node.opcode());

  ASSERT_EQ(nullptr, conv2d_node.input());
  ASSERT_EQ(nullptr, conv2d_node.filter());
  ASSERT_EQ(nullptr, conv2d_node.bias());
  ASSERT_EQ(luci::Padding::UNDEFINED, conv2d_node.padding());
  ASSERT_EQ(1, conv2d_node.stride()->h());
  ASSERT_EQ(1, conv2d_node.stride()->w());
  ASSERT_EQ(1, conv2d_node.dilation()->h());
  ASSERT_EQ(1, conv2d_node.dilation()->w());
  ASSERT_EQ(luci::FusedActFunc::UNDEFINED, conv2d_node.fusedActivationFunction());
}

TEST(CircleConv2Dest, input_NEG)
{
  luci::CircleConv2D conv2d_node;
  luci::CircleConv2D node;

  conv2d_node.input(&node);
  conv2d_node.filter(&node);
  conv2d_node.bias(&node);
  ASSERT_NE(nullptr, conv2d_node.input());
  ASSERT_NE(nullptr, conv2d_node.filter());
  ASSERT_NE(nullptr, conv2d_node.bias());

  conv2d_node.input(nullptr);
  conv2d_node.filter(nullptr);
  conv2d_node.bias(nullptr);
  ASSERT_EQ(nullptr, conv2d_node.input());
  ASSERT_EQ(nullptr, conv2d_node.filter());
  ASSERT_EQ(nullptr, conv2d_node.bias());

  conv2d_node.padding(luci::Padding::SAME);
  ASSERT_NE(luci::Padding::UNDEFINED, conv2d_node.padding());

  conv2d_node.stride()->h(2);
  conv2d_node.stride()->w(2);
  ASSERT_EQ(2, conv2d_node.stride()->h());
  ASSERT_EQ(2, conv2d_node.stride()->w());

  conv2d_node.dilation()->h(2);
  conv2d_node.dilation()->w(2);
  ASSERT_EQ(2, conv2d_node.dilation()->h());
  ASSERT_EQ(2, conv2d_node.dilation()->w());

  conv2d_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  ASSERT_NE(luci::FusedActFunc::UNDEFINED, conv2d_node.fusedActivationFunction());
}

TEST(CircleConv2Dest, arity_NEG)
{
  luci::CircleConv2D conv2d_node;

  ASSERT_NO_THROW(conv2d_node.arg(2));
  ASSERT_THROW(conv2d_node.arg(3), std::out_of_range);
}

TEST(CircleConv2Dest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleConv2D conv2d_node;

  TestVisitor tv;
  ASSERT_THROW(conv2d_node.accept(&tv), std::exception);
}

TEST(CircleConv2Dest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleConv2D conv2d_node;

  TestVisitor tv;
  ASSERT_THROW(conv2d_node.accept(&tv), std::exception);
}
