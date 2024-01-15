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

#include "luci/IR/Nodes/CircleGRU.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleGRUTest, constructor_P)
{
  luci::CircleGRU gru_node;

  ASSERT_EQ(luci::CircleDialect::get(), gru_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CIR_GRU, gru_node.opcode());

  ASSERT_EQ(nullptr, gru_node.input());
  ASSERT_EQ(nullptr, gru_node.hidden_hidden());
  ASSERT_EQ(nullptr, gru_node.hidden_hidden_bias());
  ASSERT_EQ(nullptr, gru_node.hidden_input());
  ASSERT_EQ(nullptr, gru_node.hidden_input_bias());
  ASSERT_EQ(nullptr, gru_node.state());
}

TEST(CircleGRUTest, input_NEG)
{
  luci::CircleGRU gru_node;
  luci::CircleGRU node;

  gru_node.input(&node);
  ASSERT_NE(nullptr, gru_node.input());

  gru_node.input(nullptr);
  ASSERT_EQ(nullptr, gru_node.input());
}

TEST(CircleGRUTest, arity_NEG)
{
  luci::CircleGRU gru_node;

  ASSERT_NO_THROW(gru_node.arg(0));
  ASSERT_NO_THROW(gru_node.arg(1));
  ASSERT_NO_THROW(gru_node.arg(2));
  ASSERT_NO_THROW(gru_node.arg(3));
  ASSERT_NO_THROW(gru_node.arg(4));
  ASSERT_NO_THROW(gru_node.arg(5));
  ASSERT_THROW(gru_node.arg(6), std::out_of_range);
}

TEST(CircleGRUTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleGRU gru_node;

  TestVisitor tv;
  ASSERT_THROW(gru_node.accept(&tv), std::exception);
}

TEST(CircleGRUTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleGRU gru_node;

  TestVisitor tv;
  ASSERT_THROW(gru_node.accept(&tv), std::exception);
}
