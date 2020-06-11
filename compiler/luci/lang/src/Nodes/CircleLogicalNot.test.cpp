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

#include "luci/IR/Nodes/CircleLogicalNot.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleLogicalNotTest, constructor_P)
{
  luci::CircleLogicalNot not_node;

  ASSERT_EQ(luci::CircleDialect::get(), not_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::LOGICAL_NOT, not_node.opcode());

  ASSERT_EQ(nullptr, not_node.x());
}

TEST(CircleLogicalNotTest, input_NEG)
{
  luci::CircleLogicalNot not_node;
  luci::CircleLogicalNot node;

  not_node.x(&node);
  ASSERT_NE(nullptr, not_node.x());

  not_node.x(nullptr);
  ASSERT_EQ(nullptr, not_node.x());
}

TEST(CircleLogicalNotTest, arity_NEG)
{
  luci::CircleLogicalNot not_node;

  ASSERT_NO_THROW(not_node.arg(0));
  ASSERT_THROW(not_node.arg(1), std::out_of_range);
}

TEST(CircleLogicalNotTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleLogicalNot not_node;

  TestVisitor tv;
  ASSERT_THROW(not_node.accept(&tv), std::exception);
}

TEST(CircleLogicalNotTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleLogicalNot not_node;

  TestVisitor tv;
  ASSERT_THROW(not_node.accept(&tv), std::exception);
}
