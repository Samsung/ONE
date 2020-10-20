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

#include "luci/IR/Nodes/CircleMaxPoolWithArgMax.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMaxPoolWithArgMaxTest, constructor_P)
{
  luci::CircleMaxPoolWithArgMax maxpoolwithargmax_node;

  ASSERT_EQ(luci::CircleDialect::get(), maxpoolwithargmax_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MAX_POOL_WITH_ARG_MAX, maxpoolwithargmax_node.opcode());

  ASSERT_EQ(nullptr, maxpoolwithargmax_node.input());
  ASSERT_EQ(luci::Padding::UNDEFINED, maxpoolwithargmax_node.padding());
  ASSERT_EQ(1, maxpoolwithargmax_node.filter()->h());
  ASSERT_EQ(1, maxpoolwithargmax_node.filter()->w());
  ASSERT_EQ(1, maxpoolwithargmax_node.stride()->h());
  ASSERT_EQ(1, maxpoolwithargmax_node.stride()->w());
  ASSERT_EQ(luci::FusedActFunc::UNDEFINED, maxpoolwithargmax_node.fusedActivationFunction());
}

TEST(CircleMaxPoolWithArgMaxTest, input_NEG)
{
  luci::CircleMaxPoolWithArgMax maxpoolwithargmax_node;
  luci::CircleMaxPoolWithArgMax node;

  maxpoolwithargmax_node.input(&node);
  ASSERT_NE(nullptr, maxpoolwithargmax_node.input());

  maxpoolwithargmax_node.input(nullptr);
  ASSERT_EQ(nullptr, maxpoolwithargmax_node.input());

  maxpoolwithargmax_node.filter()->h(2);
  maxpoolwithargmax_node.filter()->w(2);
  maxpoolwithargmax_node.stride()->h(2);
  maxpoolwithargmax_node.stride()->w(2);
  ASSERT_NE(1, maxpoolwithargmax_node.filter()->h());
  ASSERT_NE(1, maxpoolwithargmax_node.filter()->w());
  ASSERT_NE(1, maxpoolwithargmax_node.stride()->h());
  ASSERT_NE(1, maxpoolwithargmax_node.stride()->w());

  maxpoolwithargmax_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  ASSERT_NE(luci::FusedActFunc::UNDEFINED, maxpoolwithargmax_node.fusedActivationFunction());
}

TEST(CircleMaxPoolWithArgMaxTest, arity_NEG)
{
  luci::CircleMaxPoolWithArgMax maxpoolwithargmax_node;

  ASSERT_NO_THROW(maxpoolwithargmax_node.arg(0));
  ASSERT_THROW(maxpoolwithargmax_node.arg(1), std::out_of_range);
}

TEST(CircleMaxPoolWithArgMaxTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMaxPoolWithArgMax maxpoolwithargmax_node;

  TestVisitor tv;
  ASSERT_THROW(maxpoolwithargmax_node.accept(&tv), std::exception);
}

TEST(CircleMaxPoolWithArgMaxTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMaxPoolWithArgMax maxpoolwithargmax_node;

  TestVisitor tv;
  ASSERT_THROW(maxpoolwithargmax_node.accept(&tv), std::exception);
}
