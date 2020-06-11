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

#include "luci/IR/Nodes/CircleSquare.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSquareTest, constructor_P)
{
  luci::CircleSquare square_node;

  ASSERT_EQ(luci::CircleDialect::get(), square_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SQUARE, square_node.opcode());

  ASSERT_EQ(nullptr, square_node.x());
}

TEST(CircleSquareTest, input_NEG)
{
  luci::CircleSquare square_node;
  luci::CircleSquare node;

  square_node.x(&node);
  ASSERT_NE(nullptr, square_node.x());

  square_node.x(nullptr);
  ASSERT_EQ(nullptr, square_node.x());
}

TEST(CircleSquareTest, arity_NEG)
{
  luci::CircleSquare square_node;

  ASSERT_NO_THROW(square_node.arg(0));
  ASSERT_THROW(square_node.arg(1), std::out_of_range);
}

TEST(CircleSquareTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSquare square_node;

  TestVisitor tv;
  ASSERT_THROW(square_node.accept(&tv), std::exception);
}

TEST(CircleSquareTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSquare square_node;

  TestVisitor tv;
  ASSERT_THROW(square_node.accept(&tv), std::exception);
}
