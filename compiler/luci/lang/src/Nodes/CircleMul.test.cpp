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

#include "luci/IR/Nodes/CircleMul.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMulTest, constructor_P)
{
  luci::CircleMul mul_node;

  ASSERT_EQ(luci::CircleDialect::get(), mul_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MUL, mul_node.opcode());

  ASSERT_EQ(nullptr, mul_node.x());
  ASSERT_EQ(nullptr, mul_node.y());
}

TEST(CircleMulTest, input_NEG)
{
  luci::CircleMul mul_node;
  luci::CircleMul node;

  mul_node.x(&node);
  mul_node.y(&node);
  ASSERT_NE(nullptr, mul_node.x());
  ASSERT_NE(nullptr, mul_node.y());

  mul_node.x(nullptr);
  mul_node.y(nullptr);
  ASSERT_EQ(nullptr, mul_node.x());
  ASSERT_EQ(nullptr, mul_node.y());
}

TEST(CircleMulTest, arity_NEG)
{
  luci::CircleMul mul_node;

  ASSERT_NO_THROW(mul_node.arg(1));
  ASSERT_THROW(mul_node.arg(2), std::out_of_range);
}

TEST(CircleMulTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMul mul_node;

  TestVisitor tv;
  ASSERT_THROW(mul_node.accept(&tv), std::exception);
}

TEST(CircleMulTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMul mul_node;

  TestVisitor tv;
  ASSERT_THROW(mul_node.accept(&tv), std::exception);
}
