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

#include "luci/IR/Nodes/CircleResizeNearestNeighbor.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleResizeNearestNeightborTest, constructor)
{
  luci::CircleResizeNearestNeighbor resize;

  ASSERT_EQ(luci::CircleDialect::get(), resize.dialect());
  ASSERT_EQ(luci::CircleOpcode::RESIZE_NEAREST_NEIGHBOR, resize.opcode());

  ASSERT_EQ(nullptr, resize.input());
  ASSERT_EQ(nullptr, resize.size());
  ASSERT_FALSE(resize.align_corners());
}

TEST(CircleResizeNearestNeightborTest, input_NEG)
{
  luci::CircleResizeNearestNeighbor resize_node;
  luci::CircleResizeNearestNeighbor node;

  resize_node.input(&node);
  resize_node.size(&node);
  ASSERT_NE(nullptr, resize_node.input());
  ASSERT_NE(nullptr, resize_node.size());

  resize_node.input(nullptr);
  resize_node.size(nullptr);
  ASSERT_EQ(nullptr, resize_node.input());
  ASSERT_EQ(nullptr, resize_node.size());

  resize_node.align_corners(true);
  ASSERT_TRUE(resize_node.align_corners());
}

TEST(CircleResizeNearestNeightborTest, arity_NEG)
{
  luci::CircleResizeNearestNeighbor resize_node;

  ASSERT_NO_THROW(resize_node.arg(1));
  ASSERT_THROW(resize_node.arg(2), std::out_of_range);
}

TEST(CircleResizeNearestNeightborTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleResizeNearestNeighbor resize_node;

  TestVisitor tv;
  ASSERT_THROW(resize_node.accept(&tv), std::exception);
}

TEST(CircleResizeNearestNeightborTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleResizeNearestNeighbor resize_node;

  TestVisitor tv;
  ASSERT_THROW(resize_node.accept(&tv), std::exception);
}
