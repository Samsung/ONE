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

#include <gtest/gtest.h>

TEST(CircleIfTest, constructor)
{
  luci::CircleIf if_node(2, 2);

  ASSERT_EQ(if_node.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(if_node.opcode(), luci::CircleOpcode::IF);

  ASSERT_EQ(if_node.input_count(), 2);
  ASSERT_EQ(if_node.output_count(), 2);

  ASSERT_EQ(if_node.input(0), nullptr);
  ASSERT_EQ(if_node.input(1), nullptr);

  ASSERT_EQ(if_node.then_branch(), -1);
  ASSERT_EQ(if_node.else_branch(), -1);
}

TEST(CircleIfTestDeath, invalid_arity_NEG)
{
  ASSERT_DEBUG_DEATH(luci::CircleIf very_long_name_if_node(0, 1), "");
}

TEST(CircleIfTestDeath, invalid_output_count_NEG)
{
  ASSERT_DEBUG_DEATH(luci::CircleIf if_node(2, 0), "");
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
