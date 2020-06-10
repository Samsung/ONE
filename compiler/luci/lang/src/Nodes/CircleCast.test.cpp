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

#include "luci/IR/Nodes/CircleCast.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleCastTest, constructor)
{
  luci::CircleCast cast_node;

  ASSERT_EQ(luci::CircleDialect::get(), cast_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CAST, cast_node.opcode());

  ASSERT_EQ(nullptr, cast_node.x());
  ASSERT_EQ(loco::DataType::FLOAT32, cast_node.in_data_type());
  ASSERT_EQ(loco::DataType::FLOAT32, cast_node.out_data_type());
}

TEST(CircleCastTest, input_NEG)
{
  luci::CircleCast cast_node;
  luci::CircleCast node;

  cast_node.x(&node);
  ASSERT_NE(nullptr, cast_node.x());

  cast_node.x(nullptr);
  ASSERT_EQ(nullptr, cast_node.x());
}

TEST(CircleCastTest, arity_NEG)
{
  luci::CircleCast cast_node;

  ASSERT_NO_THROW(cast_node.arg(0));
  ASSERT_THROW(cast_node.arg(1), std::out_of_range);
}

TEST(CircleCastTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleCast cast_node;

  TestVisitor tv;
  ASSERT_THROW(cast_node.accept(&tv), std::exception);
}

TEST(CircleCastTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleCast cast_node;

  TestVisitor tv;
  ASSERT_THROW(cast_node.accept(&tv), std::exception);
}
