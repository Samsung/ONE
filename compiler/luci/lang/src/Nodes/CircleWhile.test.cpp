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

#include "luci/IR/Nodes/CircleWhile.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleWhileTest, constructor)
{
  luci::CircleWhile while_node(2, 2);

  ASSERT_EQ(luci::CircleDialect::get(), while_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::WHILE, while_node.opcode());

  ASSERT_EQ(2, while_node.input_count());
  ASSERT_EQ(2, while_node.output_count());

  ASSERT_EQ(nullptr, while_node.input(0));
  ASSERT_EQ(nullptr, while_node.input(1));

  ASSERT_EQ(-1, while_node.cond_branch());
  ASSERT_EQ(-1, while_node.body_branch());
}

TEST(CircleWhileTestDeath, invalid_arity_NEG)
{
  ASSERT_DEBUG_DEATH(luci::CircleWhile very_long_name_while_node(0, 1), "");

  SUCCEED();
}

TEST(CircleWhileTestDeath, invalid_output_count_NEG)
{
  ASSERT_DEBUG_DEATH(luci::CircleWhile while_node(2, 0), "");

  SUCCEED();
}

TEST(CircleWhileTestDeath, invalid_input_get_index_NEG)
{
  luci::CircleWhile while_node(2, 2);

  EXPECT_ANY_THROW(while_node.input(100));
}

TEST(CircleWhileTestDeath, invalid_input_set_index_NEG)
{
  luci::CircleWhile while_node(2, 2);

  EXPECT_ANY_THROW(while_node.input(100, nullptr));
}

TEST(CircleWhileTestDeath, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleWhile while_node(2, 2);

  TestVisitor tv;
  ASSERT_THROW(while_node.accept(&tv), std::exception);
}

TEST(CircleWhileTestDeath, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleWhile while_node(2, 2);

  TestVisitor tv;
  ASSERT_THROW(while_node.accept(&tv), std::exception);
}
