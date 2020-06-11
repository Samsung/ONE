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

#include "luci/IR/Nodes/CircleReluN1To1.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleReluN1To1Test, constructor)
{
  luci::CircleReluN1To1 relu_node;

  ASSERT_EQ(luci::CircleDialect::get(), relu_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::RELU_N1_TO_1, relu_node.opcode());

  ASSERT_EQ(nullptr, relu_node.features());
}

TEST(CircleReluN1To1Test, input_NEG)
{
  luci::CircleReluN1To1 relu_node;
  luci::CircleReluN1To1 node;

  relu_node.features(&node);
  ASSERT_NE(nullptr, relu_node.features());

  relu_node.features(nullptr);
  ASSERT_EQ(nullptr, relu_node.features());
}

TEST(CircleReluN1To1Test, arity_NEG)
{
  luci::CircleReluN1To1 relu_node;

  ASSERT_NO_THROW(relu_node.arg(0));
  ASSERT_THROW(relu_node.arg(1), std::out_of_range);
}

TEST(CircleReluN1To1Test, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleReluN1To1 relu_node;

  TestVisitor tv;
  ASSERT_THROW(relu_node.accept(&tv), std::exception);
}

TEST(CircleReluN1To1Test, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleReluN1To1 relu_node;

  TestVisitor tv;
  ASSERT_THROW(relu_node.accept(&tv), std::exception);
}
