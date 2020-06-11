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

#include "luci/IR/Nodes/CircleConcatenation.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleConcatenationTest, constructor_P)
{
  luci::CircleConcatenation concat_node(3);

  ASSERT_EQ(luci::CircleDialect::get(), concat_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CONCATENATION, concat_node.opcode());

  ASSERT_EQ(3, concat_node.numValues());
  ASSERT_EQ(nullptr, concat_node.values(0));
  ASSERT_EQ(nullptr, concat_node.values(1));
  ASSERT_EQ(nullptr, concat_node.values(2));
  ASSERT_EQ(luci::FusedActFunc::UNDEFINED, concat_node.fusedActivationFunction());
}

TEST(CircleConcatenationTest, input_NEG)
{
  luci::CircleConcatenation concat_node(2);
  luci::CircleConcatenation node(2);

  concat_node.values(0, &node);
  concat_node.values(1, &node);
  ASSERT_NE(nullptr, concat_node.values(0));
  ASSERT_NE(nullptr, concat_node.values(1));

  concat_node.values(0, nullptr);
  concat_node.values(1, nullptr);
  ASSERT_EQ(nullptr, concat_node.values(0));
  ASSERT_EQ(nullptr, concat_node.values(1));
}

TEST(CircleConcatenationTest, arity_NEG)
{
  luci::CircleConcatenation concat_node(5);

  ASSERT_NO_THROW(concat_node.arg(4));
  ASSERT_THROW(concat_node.arg(5), std::out_of_range);
}

TEST(CircleConcatenationTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleConcatenation concat_node(2);

  TestVisitor tv;
  ASSERT_THROW(concat_node.accept(&tv), std::exception);
}

TEST(CircleConcatenationTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleConcatenation concat_node(2);

  TestVisitor tv;
  ASSERT_THROW(concat_node.accept(&tv), std::exception);
}
