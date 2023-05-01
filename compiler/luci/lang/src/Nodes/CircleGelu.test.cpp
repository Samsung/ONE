/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleGelu.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleGeluTest, constructor)
{
  luci::CircleGelu gelu_node;

  ASSERT_EQ(luci::CircleDialect::get(), gelu_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::GELU, gelu_node.opcode());

  ASSERT_EQ(nullptr, gelu_node.features());

  ASSERT_EQ(false, gelu_node.approximate());
}

TEST(CircleGeluTest, input_NEG)
{
  luci::CircleGelu gelu_node;
  luci::CircleGelu node;

  gelu_node.features(&node);
  ASSERT_NE(nullptr, gelu_node.features());

  gelu_node.features(nullptr);
  ASSERT_EQ(nullptr, gelu_node.features());

  gelu_node.approximate(true);
  ASSERT_NE(false, gelu_node.approximate());
}

TEST(CircleGeluTest, arity_NEG)
{
  luci::CircleGelu gelu_node;

  ASSERT_NO_THROW(gelu_node.arg(0));
  ASSERT_THROW(gelu_node.arg(1), std::out_of_range);
}

TEST(CircleGeluTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleGelu gelu_node;

  TestVisitor tv;
  ASSERT_THROW(gelu_node.accept(&tv), std::exception);
}

TEST(CircleGeluTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleGelu gelu_node;

  TestVisitor tv;
  ASSERT_THROW(gelu_node.accept(&tv), std::exception);
}
