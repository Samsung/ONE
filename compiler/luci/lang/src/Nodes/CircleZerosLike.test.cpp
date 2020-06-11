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

#include "luci/IR/Nodes/CircleZerosLike.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleZerosLikeTest, constructor_P)
{
  luci::CircleZerosLike node;

  ASSERT_EQ(luci::CircleDialect::get(), node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ZEROS_LIKE, node.opcode());

  ASSERT_EQ(nullptr, node.input());
}

TEST(CircleZerosLikeTest, input_NEG)
{
  luci::CircleZerosLike zeros_node;
  luci::CircleZerosLike node;

  zeros_node.input(&node);
  ASSERT_NE(nullptr, zeros_node.input());

  zeros_node.input(nullptr);
  ASSERT_EQ(nullptr, zeros_node.input());
}

TEST(CircleZerosLikeTest, arity_NEG)
{
  luci::CircleZerosLike zeros_node;

  ASSERT_NO_THROW(zeros_node.arg(0));
  ASSERT_THROW(zeros_node.arg(1), std::out_of_range);
}

TEST(CircleZerosLikeTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleZerosLike zeros_node;

  TestVisitor tv;
  ASSERT_THROW(zeros_node.accept(&tv), std::exception);
}

TEST(CircleZerosLikeTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleZerosLike zeros_node;

  TestVisitor tv;
  ASSERT_THROW(zeros_node.accept(&tv), std::exception);
}
