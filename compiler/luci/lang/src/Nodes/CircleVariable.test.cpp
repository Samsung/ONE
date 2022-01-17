/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleVariable.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleVariableTest, constructor)
{
  luci::CircleVariable var_node;

  ASSERT_EQ(luci::CircleDialect::get(), var_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CIRCLEVARIABLE, var_node.opcode());
}

TEST(CircleVariableTest, arity_NEG)
{
  luci::CircleVariable var_node;

  ASSERT_THROW(var_node.arg(0), std::out_of_range);
}

TEST(CircleVariableTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleVariable var_node;

  TestVisitor tv;
  ASSERT_THROW(var_node.accept(&tv), std::exception);
}

TEST(CircleVariableTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleVariable var_node;

  TestVisitor tv;
  ASSERT_THROW(var_node.accept(&tv), std::exception);
}
