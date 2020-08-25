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

#include "luci/IR/Nodes/CircleTransposeConv.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleTransposeConvTest, constructor_P)
{
  luci::CircleTransposeConv trc_node;

  ASSERT_EQ(luci::CircleDialect::get(), trc_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::TRANSPOSE_CONV, trc_node.opcode());

  ASSERT_EQ(nullptr, trc_node.inputSizes());
  ASSERT_EQ(nullptr, trc_node.filter());
  ASSERT_EQ(nullptr, trc_node.outBackprop());

  ASSERT_EQ(luci::Padding::UNDEFINED, trc_node.padding());
  ASSERT_EQ(1, trc_node.stride()->h());
  ASSERT_EQ(1, trc_node.stride()->w());
}

TEST(CircleTransposeConvTest, input_NEG)
{
  luci::CircleTransposeConv trc_node;
  luci::CircleTransposeConv node;

  trc_node.inputSizes(&node);
  trc_node.filter(&node);
  trc_node.outBackprop(&node);
  ASSERT_NE(nullptr, trc_node.inputSizes());
  ASSERT_NE(nullptr, trc_node.filter());
  ASSERT_NE(nullptr, trc_node.outBackprop());

  trc_node.inputSizes(nullptr);
  trc_node.filter(nullptr);
  trc_node.outBackprop(nullptr);
  ASSERT_EQ(nullptr, trc_node.inputSizes());
  ASSERT_EQ(nullptr, trc_node.filter());
  ASSERT_EQ(nullptr, trc_node.outBackprop());

  trc_node.padding(luci::Padding::SAME);
  ASSERT_NE(luci::Padding::UNDEFINED, trc_node.padding());

  trc_node.stride()->h(2);
  trc_node.stride()->w(2);
  ASSERT_EQ(2, trc_node.stride()->h());
  ASSERT_EQ(2, trc_node.stride()->w());
}

TEST(CircleTransposeConvTest, arity_NEG)
{
  luci::CircleTransposeConv trc_node;

  ASSERT_NO_THROW(trc_node.arg(3));
  ASSERT_THROW(trc_node.arg(4), std::out_of_range);
}

TEST(CircleTransposeConvTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleTransposeConv trc_node;

  TestVisitor tv;
  ASSERT_THROW(trc_node.accept(&tv), std::exception);
}

TEST(CircleTransposeConvTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleTransposeConv trc_node;

  TestVisitor tv;
  ASSERT_THROW(trc_node.accept(&tv), std::exception);
}
