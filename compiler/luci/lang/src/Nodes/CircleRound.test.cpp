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

#include "luci/IR/Nodes/CircleRound.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleRoundTest, constructor_P)
{
  luci::CircleRound round_node;

  ASSERT_EQ(luci::CircleDialect::get(), round_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ROUND, round_node.opcode());

  ASSERT_EQ(nullptr, round_node.x());
}

TEST(CircleRoundTest, input_NEG)
{
  luci::CircleRound round_node;
  luci::CircleRound node;

  round_node.x(&node);
  ASSERT_NE(nullptr, round_node.x());

  round_node.x(nullptr);
  ASSERT_EQ(nullptr, round_node.x());
}

TEST(CircleRoundTest, arity_NEG)
{
  luci::CircleRound round_node;

  ASSERT_NO_THROW(round_node.arg(0));
  ASSERT_THROW(round_node.arg(1), std::out_of_range);
}

TEST(CircleRoundTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleRound round_node;

  TestVisitor tv;
  ASSERT_THROW(round_node.accept(&tv), std::exception);
}

TEST(CircleRoundTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleRound round_node;

  TestVisitor tv;
  ASSERT_THROW(round_node.accept(&tv), std::exception);
}
