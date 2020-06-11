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

#include "luci/IR/Nodes/CircleDepthToSpace.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleDepthToSpaceTest, constructor_P)
{
  luci::CircleDepthToSpace std_node;

  ASSERT_EQ(luci::CircleDialect::get(), std_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::DEPTH_TO_SPACE, std_node.opcode());

  ASSERT_EQ(nullptr, std_node.input());
  ASSERT_EQ(0, std_node.block_size());
}

TEST(CircleDepthToSpaceTest, input_NEG)
{
  luci::CircleDepthToSpace std_node;
  luci::CircleDepthToSpace node;

  std_node.input(&node);
  ASSERT_NE(nullptr, std_node.input());

  std_node.input(nullptr);
  ASSERT_EQ(nullptr, std_node.input());

  std_node.block_size(2);
  ASSERT_EQ(2, std_node.block_size());
}

TEST(CircleDepthToSpaceTest, arity_NEG)
{
  luci::CircleDepthToSpace std_node;

  ASSERT_NO_THROW(std_node.arg(0));
  ASSERT_THROW(std_node.arg(1), std::out_of_range);
}

TEST(CircleDepthToSpaceTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleDepthToSpace std_node;

  TestVisitor tv;
  ASSERT_THROW(std_node.accept(&tv), std::exception);
}

TEST(CircleDepthToSpaceTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleDepthToSpace std_node;

  TestVisitor tv;
  ASSERT_THROW(std_node.accept(&tv), std::exception);
}
