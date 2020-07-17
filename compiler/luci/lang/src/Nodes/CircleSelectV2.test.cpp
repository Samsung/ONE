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

#include "luci/IR/Nodes/CircleSelectV2.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSelectV2Test, constructor)
{
  luci::CircleSelectV2 select_node;

  ASSERT_EQ(luci::CircleDialect::get(), select_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SELECT_V2, select_node.opcode());

  ASSERT_EQ(nullptr, select_node.condition());
  ASSERT_EQ(nullptr, select_node.t());
  ASSERT_EQ(nullptr, select_node.e());
}

TEST(CircleSelectV2Test, input_NEG)
{
  luci::CircleSelectV2 select_v2_node;
  luci::CircleSelectV2 node;

  select_v2_node.condition(&node);
  select_v2_node.t(&node);
  select_v2_node.e(&node);
  ASSERT_NE(nullptr, select_v2_node.condition());
  ASSERT_NE(nullptr, select_v2_node.t());
  ASSERT_NE(nullptr, select_v2_node.e());

  select_v2_node.condition(nullptr);
  select_v2_node.t(nullptr);
  select_v2_node.e(nullptr);
  ASSERT_EQ(nullptr, select_v2_node.condition());
  ASSERT_EQ(nullptr, select_v2_node.t());
  ASSERT_EQ(nullptr, select_v2_node.e());
}

TEST(CircleSelectV2Test, arity_NEG)
{
  luci::CircleSelectV2 select_v2_node;

  ASSERT_NO_THROW(select_v2_node.arg(2));
  ASSERT_THROW(select_v2_node.arg(3), std::out_of_range);
}

TEST(CircleSelectV2Test, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSelectV2 select_v2_node;

  TestVisitor tv;
  ASSERT_THROW(select_v2_node.accept(&tv), std::exception);
}

TEST(CircleSelectV2Test, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSelectV2 select_v2_node;

  TestVisitor tv;
  ASSERT_THROW(select_v2_node.accept(&tv), std::exception);
}
