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

#include "luci/IR/Nodes/CircleRsqrt.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleRsqrtTest, constructor)
{
  luci::CircleRsqrt rsqrt_node;

  ASSERT_EQ(luci::CircleDialect::get(), rsqrt_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::RSQRT, rsqrt_node.opcode());

  ASSERT_EQ(nullptr, rsqrt_node.x());
}

TEST(CircleRsqrtTest, input_NEG)
{
  luci::CircleRsqrt rsqrt_node;
  luci::CircleRsqrt node;

  rsqrt_node.x(&node);
  ASSERT_NE(nullptr, rsqrt_node.x());

  rsqrt_node.x(nullptr);
  ASSERT_EQ(nullptr, rsqrt_node.x());
}

TEST(CircleRsqrtTest, arity_NEG)
{
  luci::CircleRsqrt rsqrt_node;

  ASSERT_NO_THROW(rsqrt_node.arg(0));
  ASSERT_THROW(rsqrt_node.arg(1), std::out_of_range);
}

TEST(CircleRsqrtTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleRsqrt rsqrt_node;

  TestVisitor tv;
  ASSERT_THROW(rsqrt_node.accept(&tv), std::exception);
}

TEST(CircleRsqrtTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleRsqrt rsqrt_node;

  TestVisitor tv;
  ASSERT_THROW(rsqrt_node.accept(&tv), std::exception);
}
