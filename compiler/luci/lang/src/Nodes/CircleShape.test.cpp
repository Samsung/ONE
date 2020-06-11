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

#include "luci/IR/Nodes/CircleShape.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleShapeTest, constructor)
{
  luci::CircleShape shape_node;

  ASSERT_EQ(luci::CircleDialect::get(), shape_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SHAPE, shape_node.opcode());

  ASSERT_EQ(nullptr, shape_node.input());
  ASSERT_EQ(loco::DataType::S32, shape_node.out_type());
}

TEST(CircleShapeTest, input_NEG)
{
  luci::CircleShape shape_node;
  luci::CircleShape node;

  shape_node.input(&node);
  ASSERT_NE(nullptr, shape_node.input());

  shape_node.input(nullptr);
  ASSERT_EQ(nullptr, shape_node.input());

  shape_node.out_type(loco::DataType::U8);
  ASSERT_NE(loco::DataType::S32, shape_node.out_type());
}

TEST(CircleShapeTest, arity_NEG)
{
  luci::CircleShape shape_node;

  ASSERT_NO_THROW(shape_node.arg(0));
  ASSERT_THROW(shape_node.arg(1), std::out_of_range);
}

TEST(CircleShapeTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleShape shape_node;

  TestVisitor tv;
  ASSERT_THROW(shape_node.accept(&tv), std::exception);
}

TEST(CircleShapeTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleShape shape_node;

  TestVisitor tv;
  ASSERT_THROW(shape_node.accept(&tv), std::exception);
}
