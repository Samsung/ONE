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

#include "luci/IR/Nodes/CircleGreater.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleGreaterTest, constructor_P)
{
  luci::CircleGreater greater_node;

  ASSERT_EQ(luci::CircleDialect::get(), greater_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::GREATER, greater_node.opcode());

  ASSERT_EQ(nullptr, greater_node.x());
  ASSERT_EQ(nullptr, greater_node.y());
}

TEST(CircleGreaterTest, input_NEG)
{
  luci::CircleGreater greater_node;
  luci::CircleGreater node;

  greater_node.x(&node);
  greater_node.y(&node);
  ASSERT_NE(nullptr, greater_node.x());
  ASSERT_NE(nullptr, greater_node.y());

  greater_node.x(nullptr);
  greater_node.y(nullptr);
  ASSERT_EQ(nullptr, greater_node.x());
  ASSERT_EQ(nullptr, greater_node.y());
}

TEST(CircleGreaterTest, arity_NEG)
{
  luci::CircleGreater greater_node;

  ASSERT_NO_THROW(greater_node.arg(1));
  ASSERT_THROW(greater_node.arg(2), std::out_of_range);
}

TEST(CircleGreaterTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleGreater greater_node;

  TestVisitor tv;
  ASSERT_THROW(greater_node.accept(&tv), std::exception);
}

TEST(CircleGreaterTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleGreater greater_node;

  TestVisitor tv;
  ASSERT_THROW(greater_node.accept(&tv), std::exception);
}
