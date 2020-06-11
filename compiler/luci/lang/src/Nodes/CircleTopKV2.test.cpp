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

#include "luci/IR/Nodes/CircleTopKV2.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleTopKV2Test, constructor)
{
  luci::CircleTopKV2 topkv2_node;

  ASSERT_EQ(luci::CircleDialect::get(), topkv2_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::TOPK_V2, topkv2_node.opcode());

  ASSERT_EQ(nullptr, topkv2_node.input());
  ASSERT_EQ(nullptr, topkv2_node.k());
}

TEST(CircleTopKV2Test, input_NEG)
{
  luci::CircleTopKV2 topkv2_node;
  luci::CircleTopKV2 node;

  topkv2_node.input(&node);
  topkv2_node.k(&node);
  ASSERT_NE(nullptr, topkv2_node.input());
  ASSERT_NE(nullptr, topkv2_node.k());

  topkv2_node.input(nullptr);
  topkv2_node.k(nullptr);
  ASSERT_EQ(nullptr, topkv2_node.input());
  ASSERT_EQ(nullptr, topkv2_node.k());
}

TEST(CircleTopKV2Test, arity_NEG)
{
  luci::CircleTopKV2 topkv2_node;

  ASSERT_NO_THROW(topkv2_node.arg(1));
  ASSERT_THROW(topkv2_node.arg(2), std::out_of_range);
}

TEST(CircleTopKV2Test, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleTopKV2 topkv2_node;

  TestVisitor tv;
  ASSERT_THROW(topkv2_node.accept(&tv), std::exception);
}

TEST(CircleTopKV2Test, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleTopKV2 topkv2_node;

  TestVisitor tv;
  ASSERT_THROW(topkv2_node.accept(&tv), std::exception);
}
