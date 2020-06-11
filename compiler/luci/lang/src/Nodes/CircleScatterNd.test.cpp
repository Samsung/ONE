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

#include "luci/IR/Nodes/CircleScatterNd.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleScatterNdTest, constructor_P)
{
  luci::CircleScatterNd scatter_nd_node;

  ASSERT_EQ(luci::CircleDialect::get(), scatter_nd_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SCATTER_ND, scatter_nd_node.opcode());

  ASSERT_EQ(nullptr, scatter_nd_node.indices());
  ASSERT_EQ(nullptr, scatter_nd_node.updates());
  ASSERT_EQ(nullptr, scatter_nd_node.shape());
}

TEST(CircleScatterNdTest, input_NEG)
{
  luci::CircleScatterNd scatter_nd_node;
  luci::CircleScatterNd node;

  scatter_nd_node.indices(&node);
  scatter_nd_node.updates(&node);
  scatter_nd_node.shape(&node);
  ASSERT_NE(nullptr, scatter_nd_node.indices());
  ASSERT_NE(nullptr, scatter_nd_node.updates());
  ASSERT_NE(nullptr, scatter_nd_node.shape());

  scatter_nd_node.indices(nullptr);
  scatter_nd_node.updates(nullptr);
  scatter_nd_node.shape(nullptr);
  ASSERT_EQ(nullptr, scatter_nd_node.indices());
  ASSERT_EQ(nullptr, scatter_nd_node.updates());
  ASSERT_EQ(nullptr, scatter_nd_node.shape());
}

TEST(CircleScatterNdTest, arity_NEG)
{
  luci::CircleScatterNd scatter_nd_node;

  ASSERT_NO_THROW(scatter_nd_node.arg(2));
  ASSERT_THROW(scatter_nd_node.arg(3), std::out_of_range);
}

TEST(CircleScatterNdTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleScatterNd scatter_nd_node;

  TestVisitor tv;
  ASSERT_THROW(scatter_nd_node.accept(&tv), std::exception);
}

TEST(CircleScatterNdTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleScatterNd scatter_nd_node;

  TestVisitor tv;
  ASSERT_THROW(scatter_nd_node.accept(&tv), std::exception);
}
