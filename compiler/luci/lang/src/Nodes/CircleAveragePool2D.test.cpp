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

#include "luci/IR/Nodes/CircleAveragePool2D.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleAveragePool2DTest, constructor_P)
{
  luci::CircleAveragePool2D average_pool_2d_node;

  ASSERT_EQ(luci::CircleDialect::get(), average_pool_2d_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::AVERAGE_POOL_2D, average_pool_2d_node.opcode());

  ASSERT_EQ(nullptr, average_pool_2d_node.value());
  ASSERT_EQ(luci::Padding::UNDEFINED, average_pool_2d_node.padding());
  ASSERT_EQ(1, average_pool_2d_node.filter()->h());
  ASSERT_EQ(1, average_pool_2d_node.filter()->w());
  ASSERT_EQ(1, average_pool_2d_node.stride()->h());
  ASSERT_EQ(1, average_pool_2d_node.stride()->w());
}

TEST(CircleAveragePool2DTest, input_NEG)
{
  luci::CircleAveragePool2D avgpool_node;
  luci::CircleAveragePool2D node;

  avgpool_node.value(&node);
  ASSERT_NE(nullptr, avgpool_node.value());

  avgpool_node.value(nullptr);
  ASSERT_EQ(nullptr, avgpool_node.value());

  avgpool_node.filter()->h(2);
  avgpool_node.filter()->w(2);
  avgpool_node.stride()->h(2);
  avgpool_node.stride()->w(2);
  ASSERT_NE(1, avgpool_node.filter()->h());
  ASSERT_NE(1, avgpool_node.filter()->w());
  ASSERT_NE(1, avgpool_node.stride()->h());
  ASSERT_NE(1, avgpool_node.stride()->w());
}

TEST(CircleAveragePool2DTest, arity_NEG)
{
  luci::CircleAveragePool2D avgpool_node;

  ASSERT_NO_THROW(avgpool_node.arg(0));
  ASSERT_THROW(avgpool_node.arg(1), std::out_of_range);
}

TEST(CircleAveragePool2DTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleAveragePool2D avgpool_node;

  TestVisitor tv;
  ASSERT_THROW(avgpool_node.accept(&tv), std::exception);
}

TEST(CircleAveragePool2DTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleAveragePool2D avgpool_node;

  TestVisitor tv;
  ASSERT_THROW(avgpool_node.accept(&tv), std::exception);
}
