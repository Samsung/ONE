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

#include "luci/IR/Nodes/CircleMaxPool2D.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMaxPool2DTest, constructor_P)
{
  luci::CircleMaxPool2D maxpool2d_node;

  ASSERT_EQ(luci::CircleDialect::get(), maxpool2d_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MAX_POOL_2D, maxpool2d_node.opcode());

  ASSERT_EQ(nullptr, maxpool2d_node.value());
  ASSERT_EQ(luci::Padding::UNDEFINED, maxpool2d_node.padding());
  ASSERT_EQ(1, maxpool2d_node.filter()->h());
  ASSERT_EQ(1, maxpool2d_node.filter()->w());
  ASSERT_EQ(1, maxpool2d_node.stride()->h());
  ASSERT_EQ(1, maxpool2d_node.stride()->w());
}

TEST(CircleMaxPool2DTest, input_NEG)
{
  luci::CircleMaxPool2D maxpool2d_node;
  luci::CircleMaxPool2D node;

  maxpool2d_node.value(&node);
  ASSERT_NE(nullptr, maxpool2d_node.value());

  maxpool2d_node.value(nullptr);
  ASSERT_EQ(nullptr, maxpool2d_node.value());

  maxpool2d_node.filter()->h(2);
  maxpool2d_node.filter()->w(2);
  maxpool2d_node.stride()->h(2);
  maxpool2d_node.stride()->w(2);
  ASSERT_NE(1, maxpool2d_node.filter()->h());
  ASSERT_NE(1, maxpool2d_node.filter()->w());
  ASSERT_NE(1, maxpool2d_node.stride()->h());
  ASSERT_NE(1, maxpool2d_node.stride()->w());
}

TEST(CircleMaxPool2DTest, arity_NEG)
{
  luci::CircleMaxPool2D maxpool2d_node;

  ASSERT_NO_THROW(maxpool2d_node.arg(0));
  ASSERT_THROW(maxpool2d_node.arg(1), std::out_of_range);
}

TEST(CircleMaxPool2DTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMaxPool2D maxpool2d_node;

  TestVisitor tv;
  ASSERT_THROW(maxpool2d_node.accept(&tv), std::exception);
}

TEST(CircleMaxPool2DTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMaxPool2D maxpool2d_node;

  TestVisitor tv;
  ASSERT_THROW(maxpool2d_node.accept(&tv), std::exception);
}
