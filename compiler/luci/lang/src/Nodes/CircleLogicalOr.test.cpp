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

#include "luci/IR/Nodes/CircleLogicalOr.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleLogicalOrTest, constructor_P)
{
  luci::CircleLogicalOr or_node;

  ASSERT_EQ(luci::CircleDialect::get(), or_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::LOGICAL_OR, or_node.opcode());

  ASSERT_EQ(nullptr, or_node.x());
  ASSERT_EQ(nullptr, or_node.y());
}

TEST(CircleLogicalOrTest, input_NEG)
{
  luci::CircleLogicalOr or_node;
  luci::CircleLogicalOr node;

  or_node.x(&node);
  or_node.y(&node);
  ASSERT_NE(nullptr, or_node.x());
  ASSERT_NE(nullptr, or_node.y());

  or_node.x(nullptr);
  or_node.y(nullptr);
  ASSERT_EQ(nullptr, or_node.x());
  ASSERT_EQ(nullptr, or_node.y());
}

TEST(CircleLogicalOrTest, arity_NEG)
{
  luci::CircleLogicalOr or_node;

  ASSERT_NO_THROW(or_node.arg(1));
  ASSERT_THROW(or_node.arg(2), std::out_of_range);
}

TEST(CircleLogicalOrTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleLogicalOr or_node;

  TestVisitor tv;
  ASSERT_THROW(or_node.accept(&tv), std::exception);
}

TEST(CircleLogicalOrTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleLogicalOr or_node;

  TestVisitor tv;
  ASSERT_THROW(or_node.accept(&tv), std::exception);
}
