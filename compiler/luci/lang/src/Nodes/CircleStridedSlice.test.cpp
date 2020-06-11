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

#include "luci/IR/Nodes/CircleStridedSlice.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleStridedSliceTest, constructor)
{
  luci::CircleStridedSlice ss_node;

  ASSERT_EQ(luci::CircleDialect::get(), ss_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::STRIDED_SLICE, ss_node.opcode());

  ASSERT_EQ(nullptr, ss_node.input());
  ASSERT_EQ(nullptr, ss_node.begin());
  ASSERT_EQ(nullptr, ss_node.end());
  ASSERT_EQ(nullptr, ss_node.strides());

  ASSERT_EQ(0, ss_node.begin_mask());
  ASSERT_EQ(0, ss_node.end_mask());
  ASSERT_EQ(0, ss_node.ellipsis_mask());
  ASSERT_EQ(0, ss_node.new_axis_mask());
  ASSERT_EQ(0, ss_node.shrink_axis_mask());
}

TEST(CircleStridedSliceTest, input_NEG)
{
  luci::CircleStridedSlice ss_node;
  luci::CircleStridedSlice node;

  ss_node.input(&node);
  ss_node.begin(&node);
  ss_node.end(&node);
  ss_node.strides(&node);
  ASSERT_NE(nullptr, ss_node.input());
  ASSERT_NE(nullptr, ss_node.begin());
  ASSERT_NE(nullptr, ss_node.end());
  ASSERT_NE(nullptr, ss_node.strides());

  ss_node.input(nullptr);
  ss_node.begin(nullptr);
  ss_node.end(nullptr);
  ss_node.strides(nullptr);
  ASSERT_EQ(nullptr, ss_node.input());
  ASSERT_EQ(nullptr, ss_node.begin());
  ASSERT_EQ(nullptr, ss_node.end());
  ASSERT_EQ(nullptr, ss_node.strides());

  ss_node.begin_mask(1);
  ss_node.end_mask(1);
  ss_node.ellipsis_mask(1);
  ss_node.new_axis_mask(1);
  ss_node.shrink_axis_mask(1);
  ASSERT_NE(0, ss_node.begin_mask());
  ASSERT_NE(0, ss_node.end_mask());
  ASSERT_NE(0, ss_node.ellipsis_mask());
  ASSERT_NE(0, ss_node.new_axis_mask());
  ASSERT_NE(0, ss_node.shrink_axis_mask());
}

TEST(CircleStridedSliceTest, arity_NEG)
{
  luci::CircleStridedSlice ss_node;

  ASSERT_NO_THROW(ss_node.arg(3));
  ASSERT_THROW(ss_node.arg(4), std::out_of_range);
}

TEST(CircleStridedSliceTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleStridedSlice ss_node;

  TestVisitor tv;
  ASSERT_THROW(ss_node.accept(&tv), std::exception);
}

TEST(CircleStridedSliceTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleStridedSlice ss_node;

  TestVisitor tv;
  ASSERT_THROW(ss_node.accept(&tv), std::exception);
}
