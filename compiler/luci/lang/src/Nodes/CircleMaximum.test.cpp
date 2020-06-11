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

#include "luci/IR/Nodes/CircleMaximum.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMaximumTest, constructor_P)
{
  luci::CircleMaximum max_node;

  ASSERT_EQ(luci::CircleDialect::get(), max_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MAXIMUM, max_node.opcode());

  ASSERT_EQ(nullptr, max_node.x());
  ASSERT_EQ(nullptr, max_node.y());
}

TEST(CircleMaximumTest, input_NEG)
{
  luci::CircleMaximum max_node;
  luci::CircleMaximum node;

  max_node.x(&node);
  max_node.y(&node);
  ASSERT_NE(nullptr, max_node.x());
  ASSERT_NE(nullptr, max_node.y());

  max_node.x(nullptr);
  max_node.y(nullptr);
  ASSERT_EQ(nullptr, max_node.x());
  ASSERT_EQ(nullptr, max_node.y());
}

TEST(CircleMaximumTest, arity_NEG)
{
  luci::CircleMaximum max_node;

  ASSERT_NO_THROW(max_node.arg(1));
  ASSERT_THROW(max_node.arg(2), std::out_of_range);
}

TEST(CircleMaximumTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMaximum max_node;

  TestVisitor tv;
  ASSERT_THROW(max_node.accept(&tv), std::exception);
}

TEST(CircleMaximumTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMaximum max_node;

  TestVisitor tv;
  ASSERT_THROW(max_node.accept(&tv), std::exception);
}
