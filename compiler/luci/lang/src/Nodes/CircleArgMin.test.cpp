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

#include "luci/IR/Nodes/CircleArgMin.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleArgMinTest, constructor_P)
{
  luci::CircleArgMin argmin_node;

  ASSERT_EQ(luci::CircleDialect::get(), argmin_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ARG_MIN, argmin_node.opcode());

  ASSERT_EQ(nullptr, argmin_node.input());
  ASSERT_EQ(nullptr, argmin_node.dimension());
}

TEST(CircleArgMinTest, input_NEG)
{
  luci::CircleArgMin argmin_node;
  luci::CircleArgMin node;

  argmin_node.input(&node);
  argmin_node.dimension(&node);
  ASSERT_NE(nullptr, argmin_node.input());
  ASSERT_NE(nullptr, argmin_node.dimension());

  argmin_node.input(nullptr);
  argmin_node.dimension(nullptr);
  ASSERT_EQ(nullptr, argmin_node.input());
  ASSERT_EQ(nullptr, argmin_node.dimension());
}

TEST(CircleArgMinTest, arity_NEG)
{
  luci::CircleArgMin argmin_node;

  ASSERT_NO_THROW(argmin_node.arg(1));
  ASSERT_THROW(argmin_node.arg(2), std::out_of_range);
}

TEST(CircleArgMinTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleArgMin argmin_node;

  TestVisitor tv;
  ASSERT_THROW(argmin_node.accept(&tv), std::exception);
}

TEST(CircleArgMinTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleArgMin argmin_node;

  TestVisitor tv;
  ASSERT_THROW(argmin_node.accept(&tv), std::exception);
}
