/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleCumSum.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleCumSumTest, constructor_P)
{
  luci::CircleCumSum node;

  ASSERT_EQ(luci::CircleDialect::get(), node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CUMSUM, node.opcode());

  ASSERT_EQ(nullptr, node.input());
}

TEST(CircleCumSumTest, input_NEG)
{
  luci::CircleCumSum node;
  luci::CircleCumSum input;

  node.input(&input);
  ASSERT_NE(nullptr, node.input());

  node.input(nullptr);
  ASSERT_EQ(nullptr, node.input());
}

// FIXME
TEST(CircleCumSumTest, arity_NEG)
{
  luci::CircleCumSum node;

  ASSERT_NO_THROW(node.arg(0));
  ASSERT_NO_THROW(node.arg(1));
  ASSERT_THROW(node.arg(2), std::out_of_range);
}

TEST(CircleCumSumTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleCumSum node;

  TestVisitor tv;
  ASSERT_THROW(node.accept(&tv), std::exception);
}

TEST(CircleCumSumTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleCumSum node;

  TestVisitor tv;
  ASSERT_THROW(node.accept(&tv), std::exception);
}
