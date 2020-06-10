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

#include "luci/IR/Nodes/CircleAdd.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleAddTest, constructor_P)
{
  luci::CircleAdd add_node;

  ASSERT_EQ(luci::CircleDialect::get(), add_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ADD, add_node.opcode());

  ASSERT_EQ(nullptr, add_node.x());
  ASSERT_EQ(nullptr, add_node.y());
}

TEST(CircleAddTest, input_NEG)
{
  luci::CircleAdd add_node;
  luci::CircleAdd node;

  add_node.x(&node);
  add_node.y(&node);
  ASSERT_NE(nullptr, add_node.x());
  ASSERT_NE(nullptr, add_node.y());

  add_node.x(nullptr);
  add_node.y(nullptr);
  ASSERT_EQ(nullptr, add_node.x());
  ASSERT_EQ(nullptr, add_node.y());
}

TEST(CircleAddTest, arity_NEG)
{
  luci::CircleAdd add_node;

  ASSERT_NO_THROW(add_node.arg(1));
  ASSERT_THROW(add_node.arg(2), std::out_of_range);
}

TEST(CircleAddTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleAdd add_node;

  TestVisitor tv;
  ASSERT_THROW(add_node.accept(&tv), std::exception);
}

TEST(CircleAddTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleAdd add_node;

  TestVisitor tv;
  ASSERT_THROW(add_node.accept(&tv), std::exception);
}
