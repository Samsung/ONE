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

#include "luci/IR/Nodes/CircleBatchMatMul.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleBatchMatMulTest, constructor)
{
  luci::CircleBatchMatMul batchmatmul_node;

  ASSERT_EQ(luci::CircleDialect::get(), batchmatmul_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::BATCH_MATMUL, batchmatmul_node.opcode());

  ASSERT_EQ(nullptr, batchmatmul_node.x());
  ASSERT_EQ(nullptr, batchmatmul_node.y());

  ASSERT_FALSE(batchmatmul_node.adj_x());
  ASSERT_FALSE(batchmatmul_node.adj_y());
}

TEST(CircleBatchMatMulTest, input_NEG)
{
  luci::CircleBatchMatMul batchmatmul_node;
  luci::CircleBatchMatMul node;

  batchmatmul_node.x(&node);
  batchmatmul_node.y(&node);
  ASSERT_NE(nullptr, batchmatmul_node.x());
  ASSERT_NE(nullptr, batchmatmul_node.y());

  batchmatmul_node.x(nullptr);
  batchmatmul_node.y(nullptr);
  ASSERT_EQ(nullptr, batchmatmul_node.x());
  ASSERT_EQ(nullptr, batchmatmul_node.y());
}

TEST(CircleBatchMatMulTest, arity_NEG)
{
  luci::CircleBatchMatMul batchmatmul_node;

  ASSERT_NO_THROW(batchmatmul_node.arg(1));
  ASSERT_THROW(batchmatmul_node.arg(2), std::out_of_range);
}

TEST(CircleBatchMatMulTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleBatchMatMul batchmatmul_node;

  TestVisitor tv;
  ASSERT_THROW(batchmatmul_node.accept(&tv), std::exception);
}

TEST(CircleBatchMatMulTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleBatchMatMul batchmatmul_node;

  TestVisitor tv;
  ASSERT_THROW(batchmatmul_node.accept(&tv), std::exception);
}
