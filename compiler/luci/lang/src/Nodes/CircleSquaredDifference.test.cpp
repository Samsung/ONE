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

#include "luci/IR/Nodes/CircleSquaredDifference.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSquaredDifferenceTest, constructor_P)
{
  luci::CircleSquaredDifference sd_node;

  ASSERT_EQ(luci::CircleDialect::get(), sd_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SQUARED_DIFFERENCE, sd_node.opcode());

  ASSERT_EQ(nullptr, sd_node.x());
  ASSERT_EQ(nullptr, sd_node.y());
}

TEST(CircleSquaredDifferenceTest, input_NEG)
{
  luci::CircleSquaredDifference sd_node;
  luci::CircleSquaredDifference node;

  sd_node.x(&node);
  sd_node.y(&node);
  ASSERT_NE(nullptr, sd_node.x());
  ASSERT_NE(nullptr, sd_node.y());

  sd_node.x(nullptr);
  sd_node.y(nullptr);
  ASSERT_EQ(nullptr, sd_node.x());
  ASSERT_EQ(nullptr, sd_node.y());
}

TEST(CircleSquaredDifferenceTest, arity_NEG)
{
  luci::CircleSquaredDifference sd_node;

  ASSERT_NO_THROW(sd_node.arg(1));
  ASSERT_THROW(sd_node.arg(2), std::out_of_range);
}

TEST(CircleSquaredDifferenceTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSquaredDifference sd_node;

  TestVisitor tv;
  ASSERT_THROW(sd_node.accept(&tv), std::exception);
}

TEST(CircleSquaredDifferenceTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSquaredDifference sd_node;

  TestVisitor tv;
  ASSERT_THROW(sd_node.accept(&tv), std::exception);
}
