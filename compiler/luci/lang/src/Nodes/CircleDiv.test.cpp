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

#include "luci/IR/Nodes/CircleDiv.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleDivTest, constructor_P)
{
  luci::CircleDiv div_node;

  ASSERT_EQ(luci::CircleDialect::get(), div_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::DIV, div_node.opcode());

  ASSERT_EQ(nullptr, div_node.x());
  ASSERT_EQ(nullptr, div_node.y());
}

TEST(CircleDivTest, input_NEG)
{
  luci::CircleDiv div_node;
  luci::CircleDiv node;

  div_node.x(&node);
  div_node.y(&node);
  ASSERT_NE(nullptr, div_node.x());
  ASSERT_NE(nullptr, div_node.y());

  div_node.x(nullptr);
  div_node.y(nullptr);
  ASSERT_EQ(nullptr, div_node.x());
  ASSERT_EQ(nullptr, div_node.y());
}

TEST(CircleDivTest, arity_NEG)
{
  luci::CircleDiv div_node;

  ASSERT_NO_THROW(div_node.arg(1));
  ASSERT_THROW(div_node.arg(2), std::out_of_range);
}

TEST(CircleDivTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleDiv div_node;

  TestVisitor tv;
  ASSERT_THROW(div_node.accept(&tv), std::exception);
}

TEST(CircleDivTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleDiv div_node;

  TestVisitor tv;
  ASSERT_THROW(div_node.accept(&tv), std::exception);
}
