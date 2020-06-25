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

#include "luci/IR/Nodes/CircleWhere.h"
#include "luci/IR/Nodes/CircleInput.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleWhereTest, constructor_with_xy_P)
{
  luci::CircleWhere where_node(/* provide xy inputs */ true);

  ASSERT_EQ(luci::CircleDialect::get(), where_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::WHERE, where_node.opcode());

  ASSERT_EQ(3, where_node.arity());
  ASSERT_EQ(nullptr, where_node.cond());
  ASSERT_EQ(nullptr, where_node.x());
  ASSERT_EQ(nullptr, where_node.y());
}

TEST(CircleWhereTest, constructor_without_xy_P)
{
  luci::CircleWhere where_node(/* provide xy inputs */ false);

  ASSERT_EQ(luci::CircleDialect::get(), where_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::WHERE, where_node.opcode());

  ASSERT_EQ(1, where_node.arity());
  ASSERT_EQ(nullptr, where_node.cond());
  ASSERT_EQ(nullptr, where_node.x());
  ASSERT_EQ(nullptr, where_node.y());
}

TEST(CircleWhereTest, constructor_without_x_NEG)
{
  luci::CircleWhere where_node(/* provide xy inputs */ false);
  luci::CircleInput input_node;

  ASSERT_EQ(1, where_node.arity());

  ASSERT_THROW(where_node.x(&input_node), std::exception);
}

TEST(CircleWhereTest, constructor_without_y_NEG)
{
  luci::CircleWhere where_node(/* provide xy inputs */ false);
  luci::CircleInput input_node;

  ASSERT_EQ(1, where_node.arity());

  ASSERT_THROW(where_node.y(&input_node), std::exception);
}
