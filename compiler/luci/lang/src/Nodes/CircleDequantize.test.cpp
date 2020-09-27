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

#include "luci/IR/Nodes/CircleDequantize.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

#include <memory>

TEST(CircleDequantizeTest, constructor)
{
  luci::CircleDequantize dequant_node;

  ASSERT_EQ(luci::CircleDialect::get(), dequant_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::DEQUANTIZE, dequant_node.opcode());

  ASSERT_EQ(nullptr, dequant_node.input());
}

TEST(CircleDequantizeTest, common_NEG)
{
  luci::CircleDequantize dequant_node;

  dequant_node.name("name");
  ASSERT_EQ("name", dequant_node.name());

  auto q = std::make_unique<luci::CircleQuantParam>();
  dequant_node.quantparam(std::move(q));
  ASSERT_NE(nullptr, dequant_node.quantparam());

  ASSERT_EQ(luci::ShapeStatus::UNDEFINED, dequant_node.shape_status());
  dequant_node.shape_status(luci::ShapeStatus::NOSHAPE);
  ASSERT_NE(luci::ShapeStatus::UNDEFINED, dequant_node.shape_status());
}

TEST(CircleDequantizeTest, input_NEG)
{
  luci::CircleDequantize dequant_node;
  luci::CircleDequantize node;

  dequant_node.input(&node);
  ASSERT_NE(nullptr, dequant_node.input());

  dequant_node.input(nullptr);
  ASSERT_EQ(nullptr, dequant_node.input());
}

TEST(CircleDequantizeTest, arity_NEG)
{
  luci::CircleDequantize dequant_node;

  ASSERT_NO_THROW(dequant_node.arg(0));
  ASSERT_THROW(dequant_node.arg(1), std::out_of_range);
}

TEST(CircleDequantizeTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleDequantize dequant_node;

  TestVisitor tv;
  ASSERT_THROW(dequant_node.accept(&tv), std::exception);
}

TEST(CircleDequantizeTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleDequantize dequant_node;

  TestVisitor tv;
  ASSERT_THROW(dequant_node.accept(&tv), std::exception);
}
