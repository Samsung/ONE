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

#include "luci/IR/Nodes/CircleSlice.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSliceTest, constructor)
{
  luci::CircleSlice s_node;

  ASSERT_EQ(luci::CircleDialect::get(), s_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SLICE, s_node.opcode());

  ASSERT_EQ(nullptr, s_node.input());
  ASSERT_EQ(nullptr, s_node.begin());
  ASSERT_EQ(nullptr, s_node.size());
}

TEST(CircleSliceTest, input_NEG)
{
  luci::CircleSlice s_node;
  luci::CircleSlice node;

  s_node.input(&node);
  s_node.begin(&node);
  s_node.size(&node);
  ASSERT_NE(nullptr, s_node.input());
  ASSERT_NE(nullptr, s_node.begin());
  ASSERT_NE(nullptr, s_node.size());

  s_node.input(nullptr);
  s_node.begin(nullptr);
  s_node.size(nullptr);
  ASSERT_EQ(nullptr, s_node.input());
  ASSERT_EQ(nullptr, s_node.begin());
  ASSERT_EQ(nullptr, s_node.size());
}

TEST(CircleSliceTest, arity_NEG)
{
  luci::CircleSlice s_node;

  ASSERT_NO_THROW(s_node.arg(2));
  ASSERT_THROW(s_node.arg(3), std::out_of_range);
}

TEST(CircleSliceTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSlice s_node;

  TestVisitor tv;
  ASSERT_THROW(s_node.accept(&tv), std::exception);
}

TEST(CircleSliceTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSlice s_node;

  TestVisitor tv;
  ASSERT_THROW(s_node.accept(&tv), std::exception);
}
