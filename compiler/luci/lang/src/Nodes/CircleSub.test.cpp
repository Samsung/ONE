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

#include "luci/IR/Nodes/CircleSub.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSubTest, constructor_P)
{
  luci::CircleSub sub_node;

  ASSERT_EQ(luci::CircleDialect::get(), sub_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SUB, sub_node.opcode());

  ASSERT_EQ(nullptr, sub_node.x());
  ASSERT_EQ(nullptr, sub_node.y());
}

TEST(CircleSubTest, input_NEG)
{
  luci::CircleSub sub_node;
  luci::CircleSub node;

  sub_node.x(&node);
  sub_node.y(&node);
  ASSERT_NE(nullptr, sub_node.x());
  ASSERT_NE(nullptr, sub_node.y());

  sub_node.x(nullptr);
  sub_node.y(nullptr);
  ASSERT_EQ(nullptr, sub_node.x());
  ASSERT_EQ(nullptr, sub_node.y());
}

TEST(CircleSubTest, arity_NEG)
{
  luci::CircleSub sub_node;

  ASSERT_NO_THROW(sub_node.arg(1));
  ASSERT_THROW(sub_node.arg(2), std::out_of_range);
}

TEST(CircleSubTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSub sub_node;

  TestVisitor tv;
  ASSERT_THROW(sub_node.accept(&tv), std::exception);
}

TEST(CircleSubTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSub sub_node;

  TestVisitor tv;
  ASSERT_THROW(sub_node.accept(&tv), std::exception);
}
