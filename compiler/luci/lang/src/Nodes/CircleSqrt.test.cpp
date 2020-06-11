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

#include "luci/IR/Nodes/CircleSqrt.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSqrtTest, constructor_P)
{
  luci::CircleSqrt sqrt_node;

  ASSERT_EQ(luci::CircleDialect::get(), sqrt_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SQRT, sqrt_node.opcode());

  ASSERT_EQ(nullptr, sqrt_node.x());
}

TEST(CircleSqrtTest, input_NEG)
{
  luci::CircleSqrt sqrt_node;
  luci::CircleSqrt node;

  sqrt_node.x(&node);
  ASSERT_NE(nullptr, sqrt_node.x());

  sqrt_node.x(nullptr);
  ASSERT_EQ(nullptr, sqrt_node.x());
}

TEST(CircleSqrtTest, arity_NEG)
{
  luci::CircleSqrt sqrt_node;

  ASSERT_NO_THROW(sqrt_node.arg(0));
  ASSERT_THROW(sqrt_node.arg(1), std::out_of_range);
}

TEST(CircleSqrtTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSqrt sqrt_node;

  TestVisitor tv;
  ASSERT_THROW(sqrt_node.accept(&tv), std::exception);
}

TEST(CircleSqrtTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSqrt sqrt_node;

  TestVisitor tv;
  ASSERT_THROW(sqrt_node.accept(&tv), std::exception);
}
