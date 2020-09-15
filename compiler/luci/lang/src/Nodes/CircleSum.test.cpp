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

#include "luci/IR/Nodes/CircleSum.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSumTest, constructor_P)
{
  luci::CircleSum sum_node;

  ASSERT_EQ(luci::CircleDialect::get(), sum_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SUM, sum_node.opcode());

  ASSERT_EQ(nullptr, sum_node.input());
  ASSERT_EQ(nullptr, sum_node.reduction_indices());
  ASSERT_FALSE(sum_node.keep_dims());
}

TEST(CircleSumTest, input_NEG)
{
  luci::CircleSum sum_node;
  luci::CircleSum node;

  sum_node.input(&node);
  sum_node.reduction_indices(&node);
  ASSERT_NE(nullptr, sum_node.input());
  ASSERT_NE(nullptr, sum_node.reduction_indices());

  sum_node.input(nullptr);
  sum_node.reduction_indices(nullptr);
  ASSERT_EQ(nullptr, sum_node.input());
  ASSERT_EQ(nullptr, sum_node.reduction_indices());

  sum_node.keep_dims(true);
  ASSERT_TRUE(sum_node.keep_dims());
}

TEST(CircleSumTest, arity_NEG)
{
  luci::CircleSum sum_node;

  ASSERT_NO_THROW(sum_node.arg(1));
  ASSERT_THROW(sum_node.arg(2), std::out_of_range);
}

TEST(CircleSumTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSum sum_node;

  TestVisitor tv;
  ASSERT_THROW(sum_node.accept(&tv), std::exception);
}

TEST(CircleSumTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSum sum_node;

  TestVisitor tv;
  ASSERT_THROW(sum_node.accept(&tv), std::exception);
}
