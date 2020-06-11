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

#include "luci/IR/Nodes/CircleReshape.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleReshapeTest, constructor_P)
{
  luci::CircleReshape reshape;

  ASSERT_EQ(luci::CircleDialect::get(), reshape.dialect());
  ASSERT_EQ(luci::CircleOpcode::RESHAPE, reshape.opcode());

  ASSERT_EQ(nullptr, reshape.tensor());
  ASSERT_EQ(nullptr, reshape.shape());
  ASSERT_EQ(0, reshape.newShape()->rank());
}

TEST(CircleReshapeTest, alloc_new_shape_P)
{
  luci::CircleReshape reshape;

  reshape.newShape()->rank(2);
  ASSERT_EQ(2, reshape.newShape()->rank());

  reshape.newShape()->dim(0) = 0;
  reshape.newShape()->dim(1) = 1;

  auto &const_reshape = const_cast<const luci::CircleReshape &>(reshape);
  ASSERT_EQ(0, const_reshape.newShape()->dim(0));
  ASSERT_EQ(1, const_reshape.newShape()->dim(1));
}

TEST(CircleReshapeTest, input_NEG)
{
  luci::CircleReshape reshape_node;
  luci::CircleReshape node;

  reshape_node.tensor(&node);
  reshape_node.shape(&node);
  ASSERT_NE(nullptr, reshape_node.tensor());
  ASSERT_NE(nullptr, reshape_node.shape());

  reshape_node.tensor(nullptr);
  reshape_node.shape(nullptr);
  ASSERT_EQ(nullptr, reshape_node.tensor());
  ASSERT_EQ(nullptr, reshape_node.shape());
}

TEST(CircleReshapeTest, arity_NEG)
{
  luci::CircleReshape reshape_node;

  ASSERT_NO_THROW(reshape_node.arg(1));
  ASSERT_THROW(reshape_node.arg(2), std::out_of_range);
}

TEST(CircleReshapeTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleReshape reshape_node;

  TestVisitor tv;
  ASSERT_THROW(reshape_node.accept(&tv), std::exception);
}

TEST(CircleReshapeTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleReshape reshape_node;

  TestVisitor tv;
  ASSERT_THROW(reshape_node.accept(&tv), std::exception);
}
