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

#include "luci/IR/Nodes/CircleFullyConnected.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleFullyConnectedTest, constructor)
{
  luci::CircleFullyConnected fc_node;

  ASSERT_EQ(luci::CircleDialect::get(), fc_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::FULLY_CONNECTED, fc_node.opcode());

  ASSERT_EQ(nullptr, fc_node.input());
  ASSERT_EQ(nullptr, fc_node.weights());
  ASSERT_EQ(nullptr, fc_node.bias());
  ASSERT_EQ(luci::FusedActFunc::UNDEFINED, fc_node.fusedActivationFunction());
  ASSERT_EQ(false, fc_node.keep_num_dims());
}

TEST(CircleFullyConnectedTest, input_NEG)
{
  luci::CircleFullyConnected fc_node;
  luci::CircleFullyConnected node;

  fc_node.input(&node);
  fc_node.weights(&node);
  fc_node.bias(&node);
  ASSERT_NE(nullptr, fc_node.input());
  ASSERT_NE(nullptr, fc_node.weights());
  ASSERT_NE(nullptr, fc_node.bias());

  fc_node.input(nullptr);
  fc_node.weights(nullptr);
  fc_node.bias(nullptr);
  ASSERT_EQ(nullptr, fc_node.input());
  ASSERT_EQ(nullptr, fc_node.weights());
  ASSERT_EQ(nullptr, fc_node.bias());

  fc_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  ASSERT_NE(luci::FusedActFunc::UNDEFINED, fc_node.fusedActivationFunction());
}

TEST(CircleFullyConnectedTest, arity_NEG)
{
  luci::CircleFullyConnected fc_node;

  ASSERT_NO_THROW(fc_node.arg(2));
  ASSERT_THROW(fc_node.arg(3), std::out_of_range);
}

TEST(CircleFullyConnectedTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleFullyConnected fc_node;

  TestVisitor tv;
  ASSERT_THROW(fc_node.accept(&tv), std::exception);
}

TEST(CircleFullyConnectedTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleFullyConnected fc_node;

  TestVisitor tv;
  ASSERT_THROW(fc_node.accept(&tv), std::exception);
}
