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

#include "luci/IR/Nodes/CircleExp.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleExpTest, constructor)
{
  luci::CircleExp exp_node;

  ASSERT_EQ(luci::CircleDialect::get(), exp_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::EXP, exp_node.opcode());

  ASSERT_EQ(nullptr, exp_node.x());
}

TEST(CircleExpTest, input_NEG)
{
  luci::CircleExp exp_node;
  luci::CircleExp node;

  exp_node.x(&node);
  ASSERT_NE(nullptr, exp_node.x());

  exp_node.x(nullptr);
  ASSERT_EQ(nullptr, exp_node.x());
}

TEST(CircleExpTest, arity_NEG)
{
  luci::CircleExp exp_node;

  ASSERT_NO_THROW(exp_node.arg(0));
  ASSERT_THROW(exp_node.arg(1), std::out_of_range);
}

TEST(CircleExpTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleExp exp_node;

  TestVisitor tv;
  ASSERT_THROW(exp_node.accept(&tv), std::exception);
}

TEST(CircleExpTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleExp exp_node;

  TestVisitor tv;
  ASSERT_THROW(exp_node.accept(&tv), std::exception);
}
