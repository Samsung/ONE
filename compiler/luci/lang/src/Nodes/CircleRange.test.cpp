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

#include "luci/IR/Nodes/CircleRange.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleRangeTest, constructor)
{
  luci::CircleRange range_node;

  ASSERT_EQ(luci::CircleDialect::get(), range_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::RANGE, range_node.opcode());

  ASSERT_EQ(nullptr, range_node.start());
  ASSERT_EQ(nullptr, range_node.limit());
  ASSERT_EQ(nullptr, range_node.delta());
}

TEST(CircleRangeTest, input_NEG)
{
  luci::CircleRange range_node;
  luci::CircleRange node;

  range_node.start(&node);
  range_node.limit(&node);
  range_node.delta(&node);
  ASSERT_NE(nullptr, range_node.start());
  ASSERT_NE(nullptr, range_node.limit());
  ASSERT_NE(nullptr, range_node.delta());

  range_node.start(nullptr);
  range_node.limit(nullptr);
  range_node.delta(nullptr);
  ASSERT_EQ(nullptr, range_node.start());
  ASSERT_EQ(nullptr, range_node.limit());
  ASSERT_EQ(nullptr, range_node.delta());
}

TEST(CircleRangeTest, arity_NEG)
{
  luci::CircleRange range_node;

  ASSERT_NO_THROW(range_node.arg(2));
  ASSERT_THROW(range_node.arg(3), std::out_of_range);
}

TEST(CircleRangeTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleRange range_node;

  TestVisitor tv;
  ASSERT_THROW(range_node.accept(&tv), std::exception);
}

TEST(CircleRangeTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleRange range_node;

  TestVisitor tv;
  ASSERT_THROW(range_node.accept(&tv), std::exception);
}
