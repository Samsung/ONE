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

#include "luci/IR/Nodes/CircleFloor.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleFloorTest, constructor)
{
  luci::CircleFloor floor_node;

  ASSERT_EQ(luci::CircleDialect::get(), floor_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::FLOOR, floor_node.opcode());

  ASSERT_EQ(nullptr, floor_node.x());
}

TEST(CircleFloorTest, input_NEG)
{
  luci::CircleFloor floor_node;
  luci::CircleFloor node;

  floor_node.x(&node);
  ASSERT_NE(nullptr, floor_node.x());

  floor_node.x(nullptr);
  ASSERT_EQ(nullptr, floor_node.x());
}

TEST(CircleFloorTest, arity_NEG)
{
  luci::CircleFloor floor_node;

  ASSERT_NO_THROW(floor_node.arg(0));
  ASSERT_THROW(floor_node.arg(1), std::out_of_range);
}

TEST(CircleFloorTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleFloor floor_node;

  TestVisitor tv;
  ASSERT_THROW(floor_node.accept(&tv), std::exception);
}

TEST(CircleFloorTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleFloor floor_node;

  TestVisitor tv;
  ASSERT_THROW(floor_node.accept(&tv), std::exception);
}
