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

#include "luci/IR/Nodes/CircleSplit.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSplitTest, constructor)
{
  luci::CircleSplit split_node;

  ASSERT_EQ(luci::CircleDialect::get(), split_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SPLIT, split_node.opcode());

  ASSERT_EQ(nullptr, split_node.input());
  ASSERT_EQ(nullptr, split_node.split_dim());
  ASSERT_EQ(0, split_node.num_split());
}

TEST(CircleSplitTest, input_NEG)
{
  luci::CircleSplit split_node;
  luci::CircleSplit node;

  split_node.input(&node);
  split_node.split_dim(&node);
  ASSERT_NE(nullptr, split_node.input());
  ASSERT_NE(nullptr, split_node.split_dim());

  split_node.input(nullptr);
  split_node.split_dim(nullptr);
  ASSERT_EQ(nullptr, split_node.input());
  ASSERT_EQ(nullptr, split_node.split_dim());

  split_node.num_split(100);
  ASSERT_NE(0, split_node.num_split());
}

TEST(CircleSplitTest, arity_NEG)
{
  luci::CircleSplit split_node;

  ASSERT_NO_THROW(split_node.arg(1));
  ASSERT_THROW(split_node.arg(2), std::out_of_range);
}

TEST(CircleSplitTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSplit split_node;

  TestVisitor tv;
  ASSERT_THROW(split_node.accept(&tv), std::exception);
}

TEST(CircleSplitTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSplit split_node;

  TestVisitor tv;
  ASSERT_THROW(split_node.accept(&tv), std::exception);
}
