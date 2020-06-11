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

#include "luci/IR/Nodes/CircleMinimum.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMinimumTest, constructor_P)
{
  luci::CircleMinimum min_node;

  ASSERT_EQ(luci::CircleDialect::get(), min_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MINIMUM, min_node.opcode());

  ASSERT_EQ(nullptr, min_node.x());
  ASSERT_EQ(nullptr, min_node.y());
}

TEST(CircleMinimumTest, input_NEG)
{
  luci::CircleMinimum min_node;
  luci::CircleMinimum node;

  min_node.x(&node);
  min_node.y(&node);
  ASSERT_NE(nullptr, min_node.x());
  ASSERT_NE(nullptr, min_node.y());

  min_node.x(nullptr);
  min_node.y(nullptr);
  ASSERT_EQ(nullptr, min_node.x());
  ASSERT_EQ(nullptr, min_node.y());
}

TEST(CircleMinimumTest, arity_NEG)
{
  luci::CircleMinimum min_node;

  ASSERT_NO_THROW(min_node.arg(1));
  ASSERT_THROW(min_node.arg(2), std::out_of_range);
}

TEST(CircleMinimumTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMinimum min_node;

  TestVisitor tv;
  ASSERT_THROW(min_node.accept(&tv), std::exception);
}

TEST(CircleMinimumTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMinimum min_node;

  TestVisitor tv;
  ASSERT_THROW(min_node.accept(&tv), std::exception);
}
