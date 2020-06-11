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

#include "luci/IR/Nodes/CircleOneHot.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleOneHotTest, constructor)
{
  luci::CircleOneHot one_hot_node;

  ASSERT_EQ(luci::CircleDialect::get(), one_hot_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ONE_HOT, one_hot_node.opcode());

  ASSERT_EQ(nullptr, one_hot_node.indices());
  ASSERT_EQ(nullptr, one_hot_node.depth());
  ASSERT_EQ(nullptr, one_hot_node.on_value());
  ASSERT_EQ(nullptr, one_hot_node.off_value());
  ASSERT_EQ(-1, one_hot_node.axis());
}

TEST(CircleOneHotTest, input_NEG)
{
  luci::CircleOneHot one_hot_node;
  luci::CircleOneHot node;

  one_hot_node.indices(&node);
  one_hot_node.depth(&node);
  one_hot_node.on_value(&node);
  one_hot_node.off_value(&node);
  ASSERT_NE(nullptr, one_hot_node.indices());
  ASSERT_NE(nullptr, one_hot_node.depth());
  ASSERT_NE(nullptr, one_hot_node.on_value());
  ASSERT_NE(nullptr, one_hot_node.off_value());

  one_hot_node.indices(nullptr);
  one_hot_node.depth(nullptr);
  one_hot_node.on_value(nullptr);
  one_hot_node.off_value(nullptr);
  ASSERT_EQ(nullptr, one_hot_node.indices());
  ASSERT_EQ(nullptr, one_hot_node.depth());
  ASSERT_EQ(nullptr, one_hot_node.on_value());
  ASSERT_EQ(nullptr, one_hot_node.off_value());

  one_hot_node.axis(1);
  ASSERT_NE(-1, one_hot_node.axis());
}

TEST(CircleOneHotTest, arity_NEG)
{
  luci::CircleOneHot one_hot_node;

  ASSERT_NO_THROW(one_hot_node.arg(3));
  ASSERT_THROW(one_hot_node.arg(4), std::out_of_range);
}

TEST(CircleOneHotTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleOneHot one_hot_node;

  TestVisitor tv;
  ASSERT_THROW(one_hot_node.accept(&tv), std::exception);
}

TEST(CircleOneHotTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleOneHot one_hot_node;

  TestVisitor tv;
  ASSERT_THROW(one_hot_node.accept(&tv), std::exception);
}
