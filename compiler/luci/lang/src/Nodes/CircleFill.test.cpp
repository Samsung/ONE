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

#include "luci/IR/Nodes/CircleFill.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleFillTest, constructor_P)
{
  luci::CircleFill fill;

  ASSERT_EQ(fill.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(fill.opcode(), luci::CircleOpcode::FILL);

  ASSERT_EQ(nullptr, fill.dims());
  ASSERT_EQ(nullptr, fill.value());
}

TEST(CircleFillTest, input_NEG)
{
  luci::CircleFill fill_node;
  luci::CircleFill node;

  fill_node.dims(&node);
  fill_node.value(&node);
  ASSERT_NE(nullptr, fill_node.dims());
  ASSERT_NE(nullptr, fill_node.value());

  fill_node.dims(nullptr);
  fill_node.value(nullptr);
  ASSERT_EQ(nullptr, fill_node.dims());
  ASSERT_EQ(nullptr, fill_node.value());
}

TEST(CircleFillTest, arity_NEG)
{
  luci::CircleFill fill_node;

  ASSERT_NO_THROW(fill_node.arg(1));
  ASSERT_THROW(fill_node.arg(2), std::out_of_range);
}

TEST(CircleFillTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleFill fill_node;

  TestVisitor tv;
  ASSERT_THROW(fill_node.accept(&tv), std::exception);
}

TEST(CircleFillTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleFill fill_node;

  TestVisitor tv;
  ASSERT_THROW(fill_node.accept(&tv), std::exception);
}
