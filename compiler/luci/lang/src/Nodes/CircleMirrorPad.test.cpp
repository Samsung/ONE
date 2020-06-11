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

#include "luci/IR/Nodes/CircleMirrorPad.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMirrorPadTest, constructor_P)
{
  luci::CircleMirrorPad pad_node;

  ASSERT_EQ(luci::CircleDialect::get(), pad_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MIRROR_PAD, pad_node.opcode());

  ASSERT_EQ(nullptr, pad_node.input());
  ASSERT_EQ(nullptr, pad_node.paddings());

  ASSERT_EQ(luci::MirrorPadMode::REFLECT, pad_node.mode());
}

TEST(CircleMirrorPadTest, input_NEG)
{
  luci::CircleMirrorPad pad_node;
  luci::CircleMirrorPad node;

  pad_node.input(&node);
  pad_node.paddings(&node);
  ASSERT_NE(nullptr, pad_node.input());
  ASSERT_NE(nullptr, pad_node.paddings());

  pad_node.input(nullptr);
  pad_node.paddings(nullptr);
  ASSERT_EQ(nullptr, pad_node.input());
  ASSERT_EQ(nullptr, pad_node.paddings());

  pad_node.mode(luci::MirrorPadMode::SYMMETRIC);
  ASSERT_NE(luci::MirrorPadMode::REFLECT, pad_node.mode());
}

TEST(CircleMirrorPadTest, arity_NEG)
{
  luci::CircleMirrorPad pad_node;

  ASSERT_NO_THROW(pad_node.arg(1));
  ASSERT_THROW(pad_node.arg(2), std::out_of_range);
}

TEST(CircleMirrorPadTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMirrorPad pad_node;

  TestVisitor tv;
  ASSERT_THROW(pad_node.accept(&tv), std::exception);
}

TEST(CircleMirrorPadTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMirrorPad pad_node;

  TestVisitor tv;
  ASSERT_THROW(pad_node.accept(&tv), std::exception);
}
