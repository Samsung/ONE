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

#include "luci/IR/Nodes/CircleSparseToDense.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSparseToDenseTest, constructor)
{
  luci::CircleSparseToDense stb_node;

  ASSERT_EQ(luci::CircleDialect::get(), stb_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SPARSE_TO_DENSE, stb_node.opcode());

  ASSERT_EQ(nullptr, stb_node.indices());
  ASSERT_EQ(nullptr, stb_node.output_shape());
  ASSERT_EQ(nullptr, stb_node.values());
  ASSERT_EQ(nullptr, stb_node.default_value());

  ASSERT_EQ(true, stb_node.validate_indices());
}

TEST(CircleSparseToDenseTest, input_NEG)
{
  luci::CircleSparseToDense stb_node;
  luci::CircleSparseToDense node;

  stb_node.indices(&node);
  stb_node.output_shape(&node);
  stb_node.values(&node);
  stb_node.default_value(&node);
  ASSERT_NE(nullptr, stb_node.indices());
  ASSERT_NE(nullptr, stb_node.output_shape());
  ASSERT_NE(nullptr, stb_node.values());
  ASSERT_NE(nullptr, stb_node.default_value());

  stb_node.indices(nullptr);
  stb_node.output_shape(nullptr);
  stb_node.values(nullptr);
  stb_node.default_value(nullptr);
  ASSERT_EQ(nullptr, stb_node.indices());
  ASSERT_EQ(nullptr, stb_node.output_shape());
  ASSERT_EQ(nullptr, stb_node.values());
  ASSERT_EQ(nullptr, stb_node.default_value());
}

TEST(CircleSparseToDenseTest, arity_NEG)
{
  luci::CircleSparseToDense stb_node;

  ASSERT_NO_THROW(stb_node.arg(3));
  ASSERT_THROW(stb_node.arg(4), std::out_of_range);
}

TEST(CircleSparseToDenseTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSparseToDense stb_node;

  TestVisitor tv;
  ASSERT_THROW(stb_node.accept(&tv), std::exception);
}

TEST(CircleSparseToDenseTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSparseToDense stb_node;

  TestVisitor tv;
  ASSERT_THROW(stb_node.accept(&tv), std::exception);
}
