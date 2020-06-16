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

#include "luci/IR/Nodes/CircleCeil.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleCeilTest, constructor)
{
  luci::CircleCeil ceil_node;

  ASSERT_EQ(luci::CircleDialect::get(), ceil_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CEIL, ceil_node.opcode());

  ASSERT_EQ(nullptr, ceil_node.x());
}

TEST(CircleCeilTest, input_NEG)
{
  luci::CircleCeil ceil_node;
  luci::CircleCeil node;

  ceil_node.x(&node);
  ASSERT_NE(nullptr, ceil_node.x());

  ceil_node.x(nullptr);
  ASSERT_EQ(nullptr, ceil_node.x());
}

TEST(CircleCeilTest, arity_NEG)
{
  luci::CircleCeil ceil_node;

  ASSERT_NO_THROW(ceil_node.arg(0));
  ASSERT_THROW(ceil_node.arg(1), std::out_of_range);
}

TEST(CircleCeilTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleCeil ceil_node;

  TestVisitor tv;
  ASSERT_THROW(ceil_node.accept(&tv), std::exception);
}

TEST(CircleCeilTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleCeil ceil_node;

  TestVisitor tv;
  ASSERT_THROW(ceil_node.accept(&tv), std::exception);
}
