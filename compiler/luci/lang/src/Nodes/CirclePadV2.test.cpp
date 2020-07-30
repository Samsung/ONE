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

#include "luci/IR/Nodes/CirclePadV2.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CirclePadV2Test, constructor_P)
{
  luci::CirclePadV2 node;

  ASSERT_EQ(luci::CircleDialect::get(), node.dialect());
  ASSERT_EQ(luci::CircleOpcode::PADV2, node.opcode());

  ASSERT_EQ(nullptr, node.input());
  ASSERT_EQ(nullptr, node.paddings());
  ASSERT_EQ(nullptr, node.constant_values());
}

TEST(CirclePadV2Test, input_NEG)
{
  luci::CirclePadV2 pad_node;
  luci::CirclePadV2 node;

  pad_node.input(&node);
  pad_node.paddings(&node);
  pad_node.constant_values(&node);
  ASSERT_NE(nullptr, pad_node.input());
  ASSERT_NE(nullptr, pad_node.paddings());
  ASSERT_NE(nullptr, pad_node.constant_values());

  pad_node.input(nullptr);
  pad_node.paddings(nullptr);
  pad_node.constant_values(nullptr);
  ASSERT_EQ(nullptr, pad_node.input());
  ASSERT_EQ(nullptr, pad_node.paddings());
  ASSERT_EQ(nullptr, pad_node.constant_values());
}

TEST(CirclePadV2Test, arity_NEG)
{
  luci::CirclePadV2 pad_node;

  ASSERT_NO_THROW(pad_node.arg(2));
  ASSERT_THROW(pad_node.arg(3), std::out_of_range);
}

TEST(CirclePadV2Test, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CirclePadV2 pad_node;

  TestVisitor tv;
  ASSERT_THROW(pad_node.accept(&tv), std::exception);
}

TEST(CirclePadV2Test, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CirclePadV2 pad_node;

  TestVisitor tv;
  ASSERT_THROW(pad_node.accept(&tv), std::exception);
}
