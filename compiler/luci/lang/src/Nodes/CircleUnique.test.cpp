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

#include "luci/IR/Nodes/CircleUnique.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleUniqueTest, constructor)
{
  luci::CircleUnique unique_node;

  ASSERT_EQ(luci::CircleDialect::get(), unique_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::UNIQUE, unique_node.opcode());

  ASSERT_EQ(nullptr, unique_node.input());
}

TEST(CircleUniqueTest, input_NEG)
{
  luci::CircleUnique unique_node;
  luci::CircleUnique node;

  unique_node.input(&node);
  ASSERT_NE(nullptr, unique_node.input());

  unique_node.input(nullptr);
  ASSERT_EQ(nullptr, unique_node.input());
}

TEST(CircleUniqueTest, arity_NEG)
{
  luci::CircleUnique unique_node;

  ASSERT_NO_THROW(unique_node.arg(0));
  ASSERT_THROW(unique_node.arg(1), std::out_of_range);
}

TEST(CircleUniqueTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleUnique unique_node;

  TestVisitor tv;
  ASSERT_THROW(unique_node.accept(&tv), std::exception);
}

TEST(CircleUniqueTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleUnique unique_node;

  TestVisitor tv;
  ASSERT_THROW(unique_node.accept(&tv), std::exception);
}
