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

#include "luci/IR/Nodes/CircleSin.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSinTest, constructor)
{
  luci::CircleSin sin_node;

  ASSERT_EQ(luci::CircleDialect::get(), sin_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SIN, sin_node.opcode());

  ASSERT_EQ(nullptr, sin_node.x());
}

TEST(CircleSinTest, input_NEG)
{
  luci::CircleSin sin_node;
  luci::CircleSin node;

  sin_node.x(&node);
  ASSERT_NE(nullptr, sin_node.x());

  sin_node.x(nullptr);
  ASSERT_EQ(nullptr, sin_node.x());
}

TEST(CircleSinTest, arity_NEG)
{
  luci::CircleSin sin_node;

  ASSERT_NO_THROW(sin_node.arg(0));
  ASSERT_THROW(sin_node.arg(1), std::out_of_range);
}

TEST(CircleSinTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSin sin_node;

  TestVisitor tv;
  ASSERT_THROW(sin_node.accept(&tv), std::exception);
}

TEST(CircleSinTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSin sin_node;

  TestVisitor tv;
  ASSERT_THROW(sin_node.accept(&tv), std::exception);
}
