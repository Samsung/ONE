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

#include "luci/IR/Nodes/CircleFloorDiv.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleFloorDivTest, constructor_P)
{
  luci::CircleFloorDiv floordiv_node;

  ASSERT_EQ(luci::CircleDialect::get(), floordiv_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::FLOOR_DIV, floordiv_node.opcode());

  ASSERT_EQ(nullptr, floordiv_node.x());
  ASSERT_EQ(nullptr, floordiv_node.y());
}

TEST(CircleFloorDivTest, input_NEG)
{
  luci::CircleFloorDiv floordiv_node;
  luci::CircleFloorDiv node;

  floordiv_node.x(&node);
  floordiv_node.y(&node);
  ASSERT_NE(nullptr, floordiv_node.x());
  ASSERT_NE(nullptr, floordiv_node.y());

  floordiv_node.x(nullptr);
  floordiv_node.y(nullptr);
  ASSERT_EQ(nullptr, floordiv_node.x());
  ASSERT_EQ(nullptr, floordiv_node.y());
}

TEST(CircleFloorDivTest, arity_NEG)
{
  luci::CircleFloorDiv floordiv_node;

  ASSERT_NO_THROW(floordiv_node.arg(1));
  ASSERT_THROW(floordiv_node.arg(2), std::out_of_range);
}

TEST(CircleFloorDivTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleFloorDiv floordiv_node;

  TestVisitor tv;
  ASSERT_THROW(floordiv_node.accept(&tv), std::exception);
}

TEST(CircleFloorDivTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleFloorDiv floordiv_node;

  TestVisitor tv;
  ASSERT_THROW(floordiv_node.accept(&tv), std::exception);
}
