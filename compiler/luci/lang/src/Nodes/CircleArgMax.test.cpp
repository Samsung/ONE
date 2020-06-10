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

#include "luci/IR/Nodes/CircleArgMax.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleArgMaxTest, constructor_P)
{
  luci::CircleArgMax argmax_node;

  ASSERT_EQ(luci::CircleDialect::get(), argmax_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ARG_MAX, argmax_node.opcode());

  ASSERT_EQ(nullptr, argmax_node.input());
  ASSERT_EQ(nullptr, argmax_node.dimension());
}

TEST(CircleArgMaxTest, input_NEG)
{
  luci::CircleArgMax argmax_node;
  luci::CircleArgMax node;

  argmax_node.input(&node);
  argmax_node.dimension(&node);
  ASSERT_NE(nullptr, argmax_node.input());
  ASSERT_NE(nullptr, argmax_node.dimension());

  argmax_node.input(nullptr);
  argmax_node.dimension(nullptr);
  ASSERT_EQ(nullptr, argmax_node.input());
  ASSERT_EQ(nullptr, argmax_node.dimension());
}

TEST(CircleArgMaxTest, arity_NEG)
{
  luci::CircleArgMax argmax_node;

  ASSERT_NO_THROW(argmax_node.arg(1));
  ASSERT_THROW(argmax_node.arg(2), std::out_of_range);
}

TEST(CircleArgMaxTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleArgMax argmax_node;

  TestVisitor tv;
  ASSERT_THROW(argmax_node.accept(&tv), std::exception);
}

TEST(CircleArgMaxTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleArgMax argmax_node;

  TestVisitor tv;
  ASSERT_THROW(argmax_node.accept(&tv), std::exception);
}
