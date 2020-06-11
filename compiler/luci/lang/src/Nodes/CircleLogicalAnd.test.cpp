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

#include "luci/IR/Nodes/CircleLogicalAnd.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleLogicalAndTest, constructor_P)
{
  luci::CircleLogicalAnd and_node;

  ASSERT_EQ(luci::CircleDialect::get(), and_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::LOGICAL_AND, and_node.opcode());

  ASSERT_EQ(nullptr, and_node.x());
  ASSERT_EQ(nullptr, and_node.y());
}

TEST(CircleLogicalAndTest, input_NEG)
{
  luci::CircleLogicalAnd and_node;
  luci::CircleLogicalAnd node;

  and_node.x(&node);
  and_node.y(&node);
  ASSERT_NE(nullptr, and_node.x());
  ASSERT_NE(nullptr, and_node.y());

  and_node.x(nullptr);
  and_node.y(nullptr);
  ASSERT_EQ(nullptr, and_node.x());
  ASSERT_EQ(nullptr, and_node.y());
}

TEST(CircleLogicalAndTest, arity_NEG)
{
  luci::CircleLogicalAnd and_node;

  ASSERT_NO_THROW(and_node.arg(1));
  ASSERT_THROW(and_node.arg(2), std::out_of_range);
}

TEST(CircleLogicalAndTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleLogicalAnd and_node;

  TestVisitor tv;
  ASSERT_THROW(and_node.accept(&tv), std::exception);
}

TEST(CircleLogicalAndTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleLogicalAnd and_node;

  TestVisitor tv;
  ASSERT_THROW(and_node.accept(&tv), std::exception);
}
