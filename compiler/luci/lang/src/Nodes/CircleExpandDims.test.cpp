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

#include "luci/IR/Nodes/CircleExpandDims.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleExpandDimsTest, constructor_P)
{
  luci::CircleExpandDims expand_dims;

  ASSERT_EQ(luci::CircleDialect::get(), expand_dims.dialect());
  ASSERT_EQ(luci::CircleOpcode::EXPAND_DIMS, expand_dims.opcode());

  ASSERT_EQ(nullptr, expand_dims.input());
  ASSERT_EQ(nullptr, expand_dims.axis());
}

TEST(CircleExpandDimsTest, input_NEG)
{
  luci::CircleExpandDims expand_dims;
  luci::CircleExpandDims node;

  expand_dims.input(&node);
  expand_dims.axis(&node);
  ASSERT_NE(nullptr, expand_dims.input());
  ASSERT_NE(nullptr, expand_dims.axis());

  expand_dims.input(nullptr);
  expand_dims.axis(nullptr);
  ASSERT_EQ(nullptr, expand_dims.input());
  ASSERT_EQ(nullptr, expand_dims.axis());
}

TEST(CircleExpandDimsTest, arity_NEG)
{
  luci::CircleExpandDims expand_dims;

  ASSERT_NO_THROW(expand_dims.arg(1));
  ASSERT_THROW(expand_dims.arg(2), std::out_of_range);
}

TEST(CircleExpandDimsTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleExpandDims expand_dims;

  TestVisitor tv;
  ASSERT_THROW(expand_dims.accept(&tv), std::exception);
}

TEST(CircleExpandDimsTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleExpandDims expand_dims;

  TestVisitor tv;
  ASSERT_THROW(expand_dims.accept(&tv), std::exception);
}
