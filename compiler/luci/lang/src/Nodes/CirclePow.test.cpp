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

#include "luci/IR/Nodes/CirclePow.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CirclePowTest, constructor_P)
{
  luci::CirclePow pow_node;

  ASSERT_EQ(luci::CircleDialect::get(), pow_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::POW, pow_node.opcode());

  ASSERT_EQ(nullptr, pow_node.x());
  ASSERT_EQ(nullptr, pow_node.y());
}

TEST(CirclePowTest, input_NEG)
{
  luci::CirclePow pow_node;
  luci::CirclePow node;

  pow_node.x(&node);
  pow_node.y(&node);
  ASSERT_NE(nullptr, pow_node.x());
  ASSERT_NE(nullptr, pow_node.y());

  pow_node.x(nullptr);
  pow_node.y(nullptr);
  ASSERT_EQ(nullptr, pow_node.x());
  ASSERT_EQ(nullptr, pow_node.y());
}

TEST(CirclePowTest, arity_NEG)
{
  luci::CirclePow pow_node;

  ASSERT_NO_THROW(pow_node.arg(1));
  ASSERT_THROW(pow_node.arg(2), std::out_of_range);
}

TEST(CirclePowTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CirclePow pow_node;

  TestVisitor tv;
  ASSERT_THROW(pow_node.accept(&tv), std::exception);
}

TEST(CirclePowTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CirclePow pow_node;

  TestVisitor tv;
  ASSERT_THROW(pow_node.accept(&tv), std::exception);
}
