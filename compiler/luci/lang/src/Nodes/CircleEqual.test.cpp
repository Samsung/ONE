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

#include "luci/IR/Nodes/CircleEqual.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleEqualTest, constructor_P)
{
  luci::CircleEqual equ_node;

  ASSERT_EQ(luci::CircleDialect::get(), equ_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::EQUAL, equ_node.opcode());

  ASSERT_EQ(nullptr, equ_node.x());
  ASSERT_EQ(nullptr, equ_node.y());
}

TEST(CircleEqualTest, input_NEG)
{
  luci::CircleEqual equ_node;
  luci::CircleEqual node;

  equ_node.x(&node);
  equ_node.y(&node);
  ASSERT_NE(nullptr, equ_node.x());
  ASSERT_NE(nullptr, equ_node.y());

  equ_node.x(nullptr);
  equ_node.y(nullptr);
  ASSERT_EQ(nullptr, equ_node.x());
  ASSERT_EQ(nullptr, equ_node.y());
}

TEST(CircleEqualTest, arity_NEG)
{
  luci::CircleEqual equ_node;

  ASSERT_NO_THROW(equ_node.arg(1));
  ASSERT_THROW(equ_node.arg(2), std::out_of_range);
}

TEST(CircleEqualTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleEqual equ_node;

  TestVisitor tv;
  ASSERT_THROW(equ_node.accept(&tv), std::exception);
}

TEST(CircleEqualTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleEqual equ_node;

  TestVisitor tv;
  ASSERT_THROW(equ_node.accept(&tv), std::exception);
}
