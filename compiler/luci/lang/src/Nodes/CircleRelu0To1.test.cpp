/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleRelu0To1.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleRelu0ToTest, constructor)
{
  luci::CircleRelu0To1 relu0to1_node;

  ASSERT_EQ(luci::CircleDialect::get(), relu0to1_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::RELU_0_TO_1, relu0to1_node.opcode());

  ASSERT_EQ(nullptr, relu0to1_node.features());
}

TEST(CircleRelu0ToTest, input_NEG)
{
  luci::CircleRelu0To1 relu0to1_node;
  luci::CircleRelu0To1 node;

  relu0to1_node.features(&node);
  ASSERT_NE(nullptr, relu0to1_node.features());

  relu0to1_node.features(nullptr);
  ASSERT_EQ(nullptr, relu0to1_node.features());
}

TEST(CircleRelu0ToTest, arity_NEG)
{
  luci::CircleRelu0To1 relu0to1_node;

  ASSERT_NO_THROW(relu0to1_node.arg(0));
  ASSERT_THROW(relu0to1_node.arg(1), std::out_of_range);
}

TEST(CircleRelu0ToTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleRelu0To1 relu0to1_node;

  TestVisitor tv;
  ASSERT_THROW(relu0to1_node.accept(&tv), std::exception);
}

TEST(CircleRelu0ToTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleRelu0To1 relu0to1_node;

  TestVisitor tv;
  ASSERT_THROW(relu0to1_node.accept(&tv), std::exception);
}
