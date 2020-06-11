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

#include "luci/IR/Nodes/CircleRelu6.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleRelu6Test, constructor_P)
{
  luci::CircleRelu6 relu6_node;

  ASSERT_EQ(luci::CircleDialect::get(), relu6_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::RELU6, relu6_node.opcode());

  ASSERT_EQ(nullptr, relu6_node.features());
}

TEST(CircleRelu6Test, input_NEG)
{
  luci::CircleRelu6 relu6_node;
  luci::CircleRelu6 node;

  relu6_node.features(&node);
  ASSERT_NE(nullptr, relu6_node.features());

  relu6_node.features(nullptr);
  ASSERT_EQ(nullptr, relu6_node.features());
}

TEST(CircleRelu6Test, arity_NEG)
{
  luci::CircleRelu6 relu6_node;

  ASSERT_NO_THROW(relu6_node.arg(0));
  ASSERT_THROW(relu6_node.arg(1), std::out_of_range);
}

TEST(CircleRelu6Test, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleRelu6 relu6_node;

  TestVisitor tv;
  ASSERT_THROW(relu6_node.accept(&tv), std::exception);
}

TEST(CircleRelu6Test, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleRelu6 relu6_node;

  TestVisitor tv;
  ASSERT_THROW(relu6_node.accept(&tv), std::exception);
}
