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

#include "luci/IR/Nodes/CircleCos.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleCosTest, constructor_P)
{
  luci::CircleCos cos_node;

  ASSERT_EQ(luci::CircleDialect::get(), cos_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::COS, cos_node.opcode());

  ASSERT_EQ(nullptr, cos_node.x());
}

TEST(CircleCosTest, input_NEG)
{
  luci::CircleCos cos_node;
  luci::CircleCos node;

  cos_node.x(&node);
  ASSERT_NE(nullptr, cos_node.x());

  cos_node.x(nullptr);
  ASSERT_EQ(nullptr, cos_node.x());
}

TEST(CircleCosTest, arity_NEG)
{
  luci::CircleCos cos_node;

  ASSERT_NO_THROW(cos_node.arg(0));
  ASSERT_THROW(cos_node.arg(1), std::out_of_range);
}

TEST(CircleCosTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleCos cos_node;

  TestVisitor tv;
  ASSERT_THROW(cos_node.accept(&tv), std::exception);
}

TEST(CircleCosTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleCos cos_node;

  TestVisitor tv;
  ASSERT_THROW(cos_node.accept(&tv), std::exception);
}
