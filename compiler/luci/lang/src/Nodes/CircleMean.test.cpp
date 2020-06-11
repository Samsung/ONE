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

#include "luci/IR/Nodes/CircleMean.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMeanTest, constructor)
{
  luci::CircleMean mean_node;

  ASSERT_EQ(luci::CircleDialect::get(), mean_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MEAN, mean_node.opcode());

  ASSERT_EQ(nullptr, mean_node.input());
  ASSERT_EQ(nullptr, mean_node.reduction_indices());

  ASSERT_FALSE(mean_node.keep_dims());
}

TEST(CircleMeanTest, input_NEG)
{
  luci::CircleMean mean_node;
  luci::CircleMean node;

  mean_node.input(&node);
  mean_node.reduction_indices(&node);
  ASSERT_NE(nullptr, mean_node.input());
  ASSERT_NE(nullptr, mean_node.reduction_indices());

  mean_node.input(nullptr);
  mean_node.reduction_indices(nullptr);
  ASSERT_EQ(nullptr, mean_node.input());
  ASSERT_EQ(nullptr, mean_node.reduction_indices());

  mean_node.keep_dims(true);
  ASSERT_TRUE(mean_node.keep_dims());
}

TEST(CircleMeanTest, arity_NEG)
{
  luci::CircleMean mean_node;

  ASSERT_NO_THROW(mean_node.arg(1));
  ASSERT_THROW(mean_node.arg(2), std::out_of_range);
}

TEST(CircleMeanTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMean mean_node;

  TestVisitor tv;
  ASSERT_THROW(mean_node.accept(&tv), std::exception);
}

TEST(CircleMeanTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMean mean_node;

  TestVisitor tv;
  ASSERT_THROW(mean_node.accept(&tv), std::exception);
}
