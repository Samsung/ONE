/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleRoPE.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleRoPETest, constructor)
{
  luci::CircleRoPE rope;

  ASSERT_EQ(luci::CircleDialect::get(), rope.dialect());
  ASSERT_EQ(luci::CircleOpcode::ROPE, rope.opcode());

  ASSERT_EQ(nullptr, rope.input());
  ASSERT_EQ(nullptr, rope.sin_table());
  ASSERT_EQ(nullptr, rope.cos_table());

  ASSERT_EQ(luci::RoPEMode::GPT_NEOX, rope.mode());
}

TEST(CircleRoPETest, input_NEG)
{
  luci::CircleRoPE rope;
  luci::CircleRoPE node;

  rope.input(&node);
  rope.sin_table(&node);
  rope.cos_table(&node);
  ASSERT_NE(nullptr, rope.input());
  ASSERT_NE(nullptr, rope.sin_table());
  ASSERT_NE(nullptr, rope.cos_table());

  rope.input(nullptr);
  rope.sin_table(nullptr);
  rope.cos_table(nullptr);
  ASSERT_EQ(nullptr, rope.input());
  ASSERT_EQ(nullptr, rope.sin_table());
  ASSERT_EQ(nullptr, rope.cos_table());

  rope.mode(luci::RoPEMode::GPT_J);
  ASSERT_NE(luci::RoPEMode::GPT_NEOX, rope.mode());
}

TEST(CircleRoPETest, arity_NEG)
{
  luci::CircleRoPE rope;

  ASSERT_NO_THROW(rope.arg(2));
  ASSERT_THROW(rope.arg(3), std::out_of_range);
}

TEST(CircleRoPETest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleRoPE rope;

  TestVisitor tv;
  ASSERT_THROW(rope.accept(&tv), std::exception);
}

TEST(CircleRoPETest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleRoPE rope;

  TestVisitor tv;
  ASSERT_THROW(rope.accept(&tv), std::exception);
}
