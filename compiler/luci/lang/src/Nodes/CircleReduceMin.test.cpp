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

#include "luci/IR/Nodes/CircleReduceMin.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleReduceMinTest, constructor_P)
{
  luci::CircleReduceMin reduce_min_node;

  ASSERT_EQ(luci::CircleDialect::get(), reduce_min_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::REDUCE_MIN, reduce_min_node.opcode());

  ASSERT_EQ(nullptr, reduce_min_node.input());
  ASSERT_EQ(nullptr, reduce_min_node.axis());

  ASSERT_FALSE(reduce_min_node.keep_dims());
}

TEST(CircleReduceMinTest, input_NEG)
{
  luci::CircleReduceMin reduce_min_node;
  luci::CircleReduceMin node;

  reduce_min_node.input(&node);
  reduce_min_node.axis(&node);
  ASSERT_NE(nullptr, reduce_min_node.input());
  ASSERT_NE(nullptr, reduce_min_node.axis());

  reduce_min_node.input(nullptr);
  reduce_min_node.axis(nullptr);
  ASSERT_EQ(nullptr, reduce_min_node.input());
  ASSERT_EQ(nullptr, reduce_min_node.axis());

  reduce_min_node.keep_dims(true);
  ASSERT_TRUE(reduce_min_node.keep_dims());
}

TEST(CircleReduceMinTest, arity_NEG)
{
  luci::CircleReduceMin reduce_min_node;

  ASSERT_NO_THROW(reduce_min_node.arg(1));
  ASSERT_THROW(reduce_min_node.arg(2), std::out_of_range);
}

TEST(CircleReduceMinTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleReduceMin reduce_min_node;

  TestVisitor tv;
  ASSERT_THROW(reduce_min_node.accept(&tv), std::exception);
}

TEST(CircleReduceMinTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleReduceMin reduce_min_node;

  TestVisitor tv;
  ASSERT_THROW(reduce_min_node.accept(&tv), std::exception);
}
