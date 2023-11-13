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

#include "luci/IR/Nodes/CircleBroadcastTo.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleBroadcastToTest, constructor)
{
  luci::CircleBroadcastTo broadcast_to_node;

  ASSERT_EQ(luci::CircleDialect::get(), broadcast_to_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::BROADCAST_TO, broadcast_to_node.opcode());

  ASSERT_EQ(nullptr, broadcast_to_node.input());

  ASSERT_EQ(nullptr, broadcast_to_node.shape());
}

TEST(CircleBroadcastToTest, input_NEG)
{
  luci::CircleBroadcastTo broadcast_to_node;
  luci::CircleBroadcastTo node;

  broadcast_to_node.input(&node);
  broadcast_to_node.shape(&node);
  ASSERT_NE(nullptr, broadcast_to_node.input());
  ASSERT_NE(nullptr, broadcast_to_node.shape());

  broadcast_to_node.input(nullptr);
  broadcast_to_node.shape(nullptr);
  ASSERT_EQ(nullptr, broadcast_to_node.input());
  ASSERT_EQ(nullptr, broadcast_to_node.shape());
}

TEST(CircleBroadcastToTest, arity_NEG)
{
  luci::CircleBroadcastTo broadcast_to_node;

  ASSERT_NO_THROW(broadcast_to_node.arg(1));
  ASSERT_THROW(broadcast_to_node.arg(2), std::out_of_range);
}

TEST(CircleBroadcastToTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleBroadcastTo broadcast_to_node;

  TestVisitor tv;
  ASSERT_THROW(broadcast_to_node.accept(&tv), std::exception);
}

TEST(CircleBroadcastToTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleBroadcastTo broadcast_to_node;

  TestVisitor tv;
  ASSERT_THROW(broadcast_to_node.accept(&tv), std::exception);
}
