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

#include "luci/IR/Nodes/CircleLogSoftmax.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleLogSoftmaxTest, constructor)
{
  luci::CircleLogSoftmax log_softmax_node;

  ASSERT_EQ(luci::CircleDialect::get(), log_softmax_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::LOG_SOFTMAX, log_softmax_node.opcode());

  ASSERT_EQ(nullptr, log_softmax_node.logits());
}

TEST(CircleLogSoftmaxTest, input_NEG)
{
  luci::CircleLogSoftmax log_softmax_node;
  luci::CircleLogSoftmax node;

  log_softmax_node.logits(&node);
  ASSERT_NE(nullptr, log_softmax_node.logits());

  log_softmax_node.logits(nullptr);
  ASSERT_EQ(nullptr, log_softmax_node.logits());
}

TEST(CircleLogSoftmaxTest, arity_NEG)
{
  luci::CircleLogSoftmax log_softmax_node;

  ASSERT_NO_THROW(log_softmax_node.arg(0));
  ASSERT_THROW(log_softmax_node.arg(1), std::out_of_range);
}

TEST(CircleLogSoftmaxTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleLogSoftmax log_softmax_node;

  TestVisitor tv;
  ASSERT_THROW(log_softmax_node.accept(&tv), std::exception);
}

TEST(CircleLogSoftmaxTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleLogSoftmax log_softmax_node;

  TestVisitor tv;
  ASSERT_THROW(log_softmax_node.accept(&tv), std::exception);
}
