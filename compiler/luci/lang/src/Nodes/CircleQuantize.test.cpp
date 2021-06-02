/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleQuantize.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

#include <memory>

TEST(CircleQuantizeTest, constructor)
{
  luci::CircleQuantize quant_node;

  ASSERT_EQ(luci::CircleDialect::get(), quant_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::QUANTIZE, quant_node.opcode());

  ASSERT_EQ(nullptr, quant_node.input());
}

TEST(CircleQuantizeTest, common_NEG)
{
  luci::CircleQuantize quant_node;

  quant_node.name("name");
  ASSERT_EQ("name", quant_node.name());

  auto q = std::make_unique<luci::CircleQuantParam>();
  quant_node.quantparam(std::move(q));
  ASSERT_NE(nullptr, quant_node.quantparam());

  ASSERT_EQ(luci::ShapeStatus::UNDEFINED, quant_node.shape_status());
  quant_node.shape_status(luci::ShapeStatus::NOSHAPE);
  ASSERT_NE(luci::ShapeStatus::UNDEFINED, quant_node.shape_status());
}

TEST(CircleQuantizeTest, input_NEG)
{
  luci::CircleQuantize quant_node;
  luci::CircleQuantize node;

  quant_node.input(&node);
  ASSERT_NE(nullptr, quant_node.input());

  quant_node.input(nullptr);
  ASSERT_EQ(nullptr, quant_node.input());
}

TEST(CircleQuantizeTest, arity_NEG)
{
  luci::CircleQuantize quant_node;

  ASSERT_NO_THROW(quant_node.arg(0));
  ASSERT_THROW(quant_node.arg(1), std::out_of_range);
}

TEST(CircleQuantizeTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleQuantize quant_node;

  TestVisitor tv;
  ASSERT_THROW(quant_node.accept(&tv), std::exception);
}

TEST(CircleQuantizeTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleQuantize quant_node;

  TestVisitor tv;
  ASSERT_THROW(quant_node.accept(&tv), std::exception);
}
