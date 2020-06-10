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

#include "luci/IR/Nodes/CircleAbs.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

#include <memory>

TEST(CircleAbsTest, constructor)
{
  luci::CircleAbs abs_node;

  ASSERT_EQ(luci::CircleDialect::get(), abs_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::ABS, abs_node.opcode());

  ASSERT_EQ(nullptr, abs_node.x());
}

TEST(CircleAbsTest, common_NEG)
{
  luci::CircleAbs abs_node;

  abs_node.name("name");
  ASSERT_EQ("name", abs_node.name());

  auto q = std::make_unique<luci::CircleQuantParam>();
  abs_node.quantparam(std::move(q));
  ASSERT_NE(nullptr, abs_node.quantparam());

  ASSERT_EQ(luci::ShapeStatus::UNDEFINED, abs_node.shape_status());
  abs_node.shape_status(luci::ShapeStatus::NOSHAPE);
  ASSERT_NE(luci::ShapeStatus::UNDEFINED, abs_node.shape_status());
}

TEST(CircleAbsTest, input_NEG)
{
  luci::CircleAbs abs_node;
  luci::CircleAbs node;

  abs_node.x(&node);
  ASSERT_NE(nullptr, abs_node.x());

  abs_node.x(nullptr);
  ASSERT_EQ(nullptr, abs_node.x());
}

TEST(CircleAbsTest, arity_NEG)
{
  luci::CircleAbs abs_node;

  ASSERT_NO_THROW(abs_node.arg(0));
  ASSERT_THROW(abs_node.arg(1), std::out_of_range);
}

TEST(CircleAbsTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleAbs abs_node;

  TestVisitor tv;
  ASSERT_THROW(abs_node.accept(&tv), std::exception);
}

TEST(CircleAbsTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleAbs abs_node;

  TestVisitor tv;
  ASSERT_THROW(abs_node.accept(&tv), std::exception);
}
