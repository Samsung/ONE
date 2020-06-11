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

#include "luci/IR/Nodes/CircleSpaceToBatchND.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSpaceToBatchNDTest, constructor)
{
  luci::CircleSpaceToBatchND stb_node;

  ASSERT_EQ(luci::CircleDialect::get(), stb_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SPACE_TO_BATCH_ND, stb_node.opcode());

  ASSERT_EQ(nullptr, stb_node.input());
  ASSERT_EQ(nullptr, stb_node.block_shape());
  ASSERT_EQ(nullptr, stb_node.paddings());
}

TEST(CircleSpaceToBatchNDTest, input_NEG)
{
  luci::CircleSpaceToBatchND stb_node;
  luci::CircleSpaceToBatchND node;

  stb_node.input(&node);
  stb_node.block_shape(&node);
  stb_node.paddings(&node);
  ASSERT_NE(nullptr, stb_node.input());
  ASSERT_NE(nullptr, stb_node.block_shape());
  ASSERT_NE(nullptr, stb_node.paddings());

  stb_node.input(nullptr);
  stb_node.block_shape(nullptr);
  stb_node.paddings(nullptr);
  ASSERT_EQ(nullptr, stb_node.input());
  ASSERT_EQ(nullptr, stb_node.block_shape());
  ASSERT_EQ(nullptr, stb_node.paddings());
}

TEST(CircleSpaceToBatchNDTest, arity_NEG)
{
  luci::CircleSpaceToBatchND stb_node;

  ASSERT_NO_THROW(stb_node.arg(2));
  ASSERT_THROW(stb_node.arg(3), std::out_of_range);
}

TEST(CircleSpaceToBatchNDTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSpaceToBatchND stb_node;

  TestVisitor tv;
  ASSERT_THROW(stb_node.accept(&tv), std::exception);
}

TEST(CircleSpaceToBatchNDTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSpaceToBatchND stb_node;

  TestVisitor tv;
  ASSERT_THROW(stb_node.accept(&tv), std::exception);
}
