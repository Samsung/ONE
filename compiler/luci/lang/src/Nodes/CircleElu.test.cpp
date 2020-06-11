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

#include "luci/IR/Nodes/CircleElu.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleEluTest, constructor_P)
{
  luci::CircleElu elu_node;

  ASSERT_EQ(luci::CircleDialect::get(), elu_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ELU, elu_node.opcode());

  ASSERT_EQ(nullptr, elu_node.features());
}

TEST(CircleEluTest, input_NEG)
{
  luci::CircleElu elu_node;
  luci::CircleElu node;

  elu_node.features(&node);
  ASSERT_NE(nullptr, elu_node.features());

  elu_node.features(nullptr);
  ASSERT_EQ(nullptr, elu_node.features());
}

TEST(CircleEluTest, arity_NEG)
{
  luci::CircleElu elu_node;

  ASSERT_NO_THROW(elu_node.arg(0));
  ASSERT_THROW(elu_node.arg(1), std::out_of_range);
}

TEST(CircleEluTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleElu elu_node;

  TestVisitor tv;
  ASSERT_THROW(elu_node.accept(&tv), std::exception);
}

TEST(CircleEluTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleElu elu_node;

  TestVisitor tv;
  ASSERT_THROW(elu_node.accept(&tv), std::exception);
}
