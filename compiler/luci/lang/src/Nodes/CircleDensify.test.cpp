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

#include "luci/IR/Nodes/CircleDensify.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleDensifyTest, constructor)
{
  luci::CircleDensify densify_node;

  ASSERT_EQ(luci::CircleDialect::get(), densify_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::DENSIFY, densify_node.opcode());

  ASSERT_EQ(nullptr, densify_node.input());
}

TEST(CircleDensifyTest, input_NEG)
{
  luci::CircleDensify densify_node;
  luci::CircleDensify node;

  densify_node.input(&node);
  ASSERT_NE(nullptr, densify_node.input());

  densify_node.input(nullptr);
  ASSERT_EQ(nullptr, densify_node.input());
}

TEST(CircleDensifyTest, arity_NEG)
{
  luci::CircleDensify densify_node;

  ASSERT_NO_THROW(densify_node.arg(0));
  ASSERT_THROW(densify_node.arg(1), std::out_of_range);
}

TEST(CircleDensifyTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleDensify densify_node;

  TestVisitor tv;
  ASSERT_THROW(densify_node.accept(&tv), std::exception);
}

TEST(CircleDensifyTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleDensify densify_node;

  TestVisitor tv;
  ASSERT_THROW(densify_node.accept(&tv), std::exception);
}
