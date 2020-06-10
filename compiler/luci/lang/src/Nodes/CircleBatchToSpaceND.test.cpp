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

#include "luci/IR/Nodes/CircleBatchToSpaceND.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleBatchToSpaceNDTest, constructor)
{
  luci::CircleBatchToSpaceND bts_node;

  ASSERT_EQ(luci::CircleDialect::get(), bts_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::BATCH_TO_SPACE_ND, bts_node.opcode());

  ASSERT_EQ(nullptr, bts_node.input());
  ASSERT_EQ(nullptr, bts_node.block_shape());
  ASSERT_EQ(nullptr, bts_node.crops());
}

TEST(CircleBatchToSpaceNDTest, input_NEG)
{
  luci::CircleBatchToSpaceND bts_node;
  luci::CircleBatchToSpaceND node;

  bts_node.input(&node);
  bts_node.block_shape(&node);
  bts_node.crops(&node);
  ASSERT_NE(nullptr, bts_node.input());
  ASSERT_NE(nullptr, bts_node.block_shape());
  ASSERT_NE(nullptr, bts_node.crops());

  bts_node.input(nullptr);
  bts_node.block_shape(nullptr);
  bts_node.crops(nullptr);
  ASSERT_EQ(nullptr, bts_node.input());
  ASSERT_EQ(nullptr, bts_node.block_shape());
  ASSERT_EQ(nullptr, bts_node.crops());
}

TEST(CircleBatchToSpaceNDTest, arity_NEG)
{
  luci::CircleBatchToSpaceND bts_node;

  ASSERT_NO_THROW(bts_node.arg(2));
  ASSERT_THROW(bts_node.arg(3), std::out_of_range);
}

TEST(CircleBatchToSpaceNDTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleBatchToSpaceND bts_node;

  TestVisitor tv;
  ASSERT_THROW(bts_node.accept(&tv), std::exception);
}

TEST(CircleBatchToSpaceNDTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleBatchToSpaceND bts_node;

  TestVisitor tv;
  ASSERT_THROW(bts_node.accept(&tv), std::exception);
}
