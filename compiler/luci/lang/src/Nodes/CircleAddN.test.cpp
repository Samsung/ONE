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

#include "luci/IR/Nodes/CircleAddN.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleAddNTest, constructor)
{
  luci::CircleAddN add_node(3);

  ASSERT_EQ(luci::CircleDialect::get(), add_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ADD_N, add_node.opcode());

  ASSERT_EQ(nullptr, add_node.inputs(0));
  ASSERT_EQ(nullptr, add_node.inputs(1));
  ASSERT_EQ(nullptr, add_node.inputs(2));
}

TEST(CircleAddNTest, input_NEG)
{
  luci::CircleAddN add_node(3);
  luci::CircleAddN node(2);

  add_node.inputs(0, &node);
  add_node.inputs(1, &node);
  add_node.inputs(2, &node);
  ASSERT_NE(nullptr, add_node.inputs(0));
  ASSERT_NE(nullptr, add_node.inputs(1));
  ASSERT_NE(nullptr, add_node.inputs(2));

  add_node.inputs(0, nullptr);
  add_node.inputs(1, nullptr);
  add_node.inputs(2, nullptr);
  ASSERT_EQ(nullptr, add_node.inputs(0));
  ASSERT_EQ(nullptr, add_node.inputs(1));
  ASSERT_EQ(nullptr, add_node.inputs(2));
}

TEST(CircleAddNTest, arity_NEG)
{
  luci::CircleAddN add_node(3);
  luci::CircleAddN node(2);

  ASSERT_NO_THROW(add_node.inputs(2, &node));
  ASSERT_NO_THROW(add_node.inputs(2, nullptr));
  ASSERT_THROW(add_node.inputs(3, &node), std::out_of_range);

  ASSERT_NO_THROW(add_node.arg(2));
  ASSERT_THROW(add_node.arg(3), std::out_of_range);
}

TEST(CircleAddNTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleAddN add_node(2);

  TestVisitor tv;
  ASSERT_THROW(add_node.accept(&tv), std::exception);
}

TEST(CircleAddNTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleAddN add_node(2);

  TestVisitor tv;
  ASSERT_THROW(add_node.accept(&tv), std::exception);
}
