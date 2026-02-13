/*
 * Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleSign.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSignTest, constructor)
{
  luci::CircleSign sign_node;

  ASSERT_EQ(luci::CircleDialect::get(), sign_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SIGN, sign_node.opcode());

  ASSERT_EQ(nullptr, sign_node.x());
}

TEST(CircleSignTest, input_NEG)
{
  luci::CircleSign sign_node;
  luci::CircleSign node;

  sign_node.x(&node);
  ASSERT_NE(nullptr, sign_node.x());

  sign_node.x(nullptr);
  ASSERT_EQ(nullptr, sign_node.x());
}

TEST(CircleSignTest, arity_NEG)
{
  luci::CircleSign sign_node;

  ASSERT_NO_THROW(sign_node.arg(0));
  ASSERT_THROW(sign_node.arg(1), std::out_of_range);
}

TEST(CircleSignTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSign sign_node;

  TestVisitor tv;
  ASSERT_THROW(sign_node.accept(&tv), std::exception);
}

TEST(CircleSignTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSign sign_node;

  TestVisitor tv;
  ASSERT_THROW(sign_node.accept(&tv), std::exception);
}
