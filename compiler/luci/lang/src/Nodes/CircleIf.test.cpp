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

#include "luci/IR/Nodes/CircleIf.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleIfTest, constructor)
{
  luci::CircleIf if_node(2, 2);

  ASSERT_EQ(luci::CircleDialect::get(), if_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::IF, if_node.opcode());

  ASSERT_EQ(2, if_node.input_count());
  ASSERT_EQ(2, if_node.output_count());

  ASSERT_EQ(nullptr, if_node.input(0));
  ASSERT_EQ(nullptr, if_node.input(1));

  ASSERT_EQ(-1, if_node.then_branch());
  ASSERT_EQ(-1, if_node.else_branch());
}

TEST(CircleIfTestDeath, invalid_arity_NEG)
{
  ASSERT_DEBUG_DEATH(luci::CircleIf very_long_name_if_node(0, 1), "");

  SUCCEED();
}

TEST(CircleIfTestDeath, invalid_output_count_NEG)
{
  ASSERT_DEBUG_DEATH(luci::CircleIf if_node(2, 0), "");

  SUCCEED();
}

TEST(CircleIfTestDeath, invalid_input_get_index_NEG)
{
  luci::CircleIf if_node(2, 2);

  EXPECT_ANY_THROW(if_node.input(100));
}

TEST(CircleIfTestDeath, invalid_input_set_index_NEG)
{
  luci::CircleIf if_node(2, 2);

  EXPECT_ANY_THROW(if_node.input(100, nullptr));
}

TEST(CircleIfTestDeath, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleIf if_node(2, 2);

  TestVisitor tv;
  ASSERT_THROW(if_node.accept(&tv), std::exception);
}

TEST(CircleIfTestDeath, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleIf if_node(2, 2);

  TestVisitor tv;
  ASSERT_THROW(if_node.accept(&tv), std::exception);
}
