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

#include "luci/IR/Nodes/CircleTanh.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleTanhTest, constructor)
{
  luci::CircleTanh tanh_node;

  ASSERT_EQ(luci::CircleDialect::get(), tanh_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::TANH, tanh_node.opcode());

  ASSERT_EQ(nullptr, tanh_node.x());
}

TEST(CircleTanhTest, input_NEG)
{
  luci::CircleTanh neg_node;
  luci::CircleTanh node;

  neg_node.x(&node);
  ASSERT_NE(nullptr, neg_node.x());

  neg_node.x(nullptr);
  ASSERT_EQ(nullptr, neg_node.x());
}

TEST(CircleTanhTest, arity_NEG)
{
  luci::CircleTanh neg_node;

  ASSERT_NO_THROW(neg_node.arg(0));
  ASSERT_THROW(neg_node.arg(1), std::out_of_range);
}

TEST(CircleTanhTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleTanh neg_node;

  TestVisitor tv;
  ASSERT_THROW(neg_node.accept(&tv), std::exception);
}

TEST(CircleTanhTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleTanh neg_node;

  TestVisitor tv;
  ASSERT_THROW(neg_node.accept(&tv), std::exception);
}
