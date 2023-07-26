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

#include "luci/IR/Nodes/CircleHardSwish.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleHardSwishTest, constructor_P)
{
  luci::CircleHardSwish hard_swish_node;

  ASSERT_EQ(luci::CircleDialect::get(), hard_swish_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::HARD_SWISH, hard_swish_node.opcode());

  ASSERT_EQ(nullptr, hard_swish_node.features());
}

TEST(CircleHardSwishTest, input_NEG)
{
  luci::CircleHardSwish hard_swish_node;
  luci::CircleHardSwish node;

  hard_swish_node.features(&node);
  ASSERT_NE(nullptr, hard_swish_node.features());

  hard_swish_node.features(nullptr);
  ASSERT_EQ(nullptr, hard_swish_node.features());
}

TEST(CircleHardSwishTest, arity_NEG)
{
  luci::CircleHardSwish hard_swish_node;

  ASSERT_NO_THROW(hard_swish_node.arg(0));
  ASSERT_THROW(hard_swish_node.arg(1), std::out_of_range);
}

TEST(CircleHardSwishTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleHardSwish hard_swish_node;

  TestVisitor tv;
  ASSERT_THROW(hard_swish_node.accept(&tv), std::exception);
}

TEST(CircleHardSwishTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleHardSwish hard_swish_node;

  TestVisitor tv;
  ASSERT_THROW(hard_swish_node.accept(&tv), std::exception);
}
