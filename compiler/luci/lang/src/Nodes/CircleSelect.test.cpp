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

#include "luci/IR/Nodes/CircleSelect.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSelectTest, constructor)
{
  luci::CircleSelect select_node;

  ASSERT_EQ(luci::CircleDialect::get(), select_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SELECT, select_node.opcode());

  ASSERT_EQ(nullptr, select_node.condition());
  ASSERT_EQ(nullptr, select_node.t());
  ASSERT_EQ(nullptr, select_node.e());
}

TEST(CircleSelectTest, input_NEG)
{
  luci::CircleSelect select_node;
  luci::CircleSelect node;

  select_node.condition(&node);
  select_node.t(&node);
  select_node.e(&node);
  ASSERT_NE(nullptr, select_node.condition());
  ASSERT_NE(nullptr, select_node.t());
  ASSERT_NE(nullptr, select_node.e());

  select_node.condition(nullptr);
  select_node.t(nullptr);
  select_node.e(nullptr);
  ASSERT_EQ(nullptr, select_node.condition());
  ASSERT_EQ(nullptr, select_node.t());
  ASSERT_EQ(nullptr, select_node.e());
}

TEST(CircleSelectTest, arity_NEG)
{
  luci::CircleSelect select_node;

  ASSERT_NO_THROW(select_node.arg(2));
  ASSERT_THROW(select_node.arg(3), std::out_of_range);
}

TEST(CircleSelectTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSelect select_node;

  TestVisitor tv;
  ASSERT_THROW(select_node.accept(&tv), std::exception);
}

TEST(CircleSelectTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSelect select_node;

  TestVisitor tv;
  ASSERT_THROW(select_node.accept(&tv), std::exception);
}
