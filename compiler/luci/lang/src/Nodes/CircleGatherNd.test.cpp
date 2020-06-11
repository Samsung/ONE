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

#include "luci/IR/Nodes/CircleGatherNd.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleGatherNdTest, constructor)
{
  luci::CircleGatherNd gather_nd_node;

  ASSERT_EQ(luci::CircleDialect::get(), gather_nd_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::GATHER_ND, gather_nd_node.opcode());

  ASSERT_EQ(nullptr, gather_nd_node.params());
  ASSERT_EQ(nullptr, gather_nd_node.indices());
}

TEST(CircleGatherNdTest, input_NEG)
{
  luci::CircleGatherNd gather_nd_node;
  luci::CircleGatherNd node;

  gather_nd_node.params(&node);
  gather_nd_node.indices(&node);
  ASSERT_NE(nullptr, gather_nd_node.params());
  ASSERT_NE(nullptr, gather_nd_node.indices());

  gather_nd_node.params(nullptr);
  gather_nd_node.indices(nullptr);
  ASSERT_EQ(nullptr, gather_nd_node.params());
  ASSERT_EQ(nullptr, gather_nd_node.indices());
}

TEST(CircleGatherNdTest, arity_NEG)
{
  luci::CircleGatherNd gather_nd_node;

  ASSERT_NO_THROW(gather_nd_node.arg(1));
  ASSERT_THROW(gather_nd_node.arg(2), std::out_of_range);
}

TEST(CircleGatherNdTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleGatherNd gather_nd_node;

  TestVisitor tv;
  ASSERT_THROW(gather_nd_node.accept(&tv), std::exception);
}

TEST(CircleGatherNdTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleGatherNd gather_nd_node;

  TestVisitor tv;
  ASSERT_THROW(gather_nd_node.accept(&tv), std::exception);
}
