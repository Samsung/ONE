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

#include "luci/IR/Nodes/CircleGather.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleGatherTest, constructor)
{
  luci::CircleGather gather_node;

  ASSERT_EQ(luci::CircleDialect::get(), gather_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::GATHER, gather_node.opcode());

  ASSERT_EQ(nullptr, gather_node.params());
  ASSERT_EQ(nullptr, gather_node.indices());
  ASSERT_EQ(0, gather_node.axis());
}

TEST(CircleGatherTest, input_NEG)
{
  luci::CircleGather gather_node;
  luci::CircleGather node;

  gather_node.params(&node);
  gather_node.indices(&node);
  ASSERT_NE(nullptr, gather_node.params());
  ASSERT_NE(nullptr, gather_node.indices());

  gather_node.params(nullptr);
  gather_node.indices(nullptr);
  ASSERT_EQ(nullptr, gather_node.params());
  ASSERT_EQ(nullptr, gather_node.indices());

  gather_node.axis(1);
  ASSERT_NE(0, gather_node.axis());
}

TEST(CircleGatherTest, arity_NEG)
{
  luci::CircleGather gather_node;

  ASSERT_NO_THROW(gather_node.arg(1));
  ASSERT_THROW(gather_node.arg(2), std::out_of_range);
}

TEST(CircleGatherTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleGather gather_node;

  TestVisitor tv;
  ASSERT_THROW(gather_node.accept(&tv), std::exception);
}

TEST(CircleGatherTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleGather gather_node;

  TestVisitor tv;
  ASSERT_THROW(gather_node.accept(&tv), std::exception);
}
